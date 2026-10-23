# -*- coding: utf-8 -*-
"""Expose an AgentScope agent as an A2A 1.0 server with A2UI support.

An ``AgentExecutor`` bridges the two event models: the agent's streamed
``TextBlockDeltaEvent`` chunks are published as artifact chunks, so an A2A
client sees the reply arrive incrementally. Each A2A ``context_id`` gets its
own agent, which is what keeps the conversations independent.

When a client negotiates the A2UI extension (``X-A2A-Extensions`` header or
``message.extensions`` + ``metadata.a2uiClientCapabilities``), the reply flows
through the A2UI incremental parser: ``<a2ui-json>`` blocks are validated and
streamed to the client as ``DataPart`` artifacts the renderer turns into UI
components, while surrounding text keeps streaming as text artifacts. See
``AgentUI/a2ui/python/a2ui_agent/agent_development.md``. Run with::

    export ARK_API_KEY=sk-...
    python server.py [--model ark-code-latest] [--port 9999]
"""
import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentInterface,
    AgentSkill,
    Part,
    Task,
    TaskState,
    TaskStatus,
)
from google.protobuf.json_format import MessageToDict, ParseDict
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware

from pydantic import SecretStr

from a2ui.core.basic_catalog import BasicCatalog
from a2ui.core.schema import ProtocolVersion
from a2ui.inference_formats import to_message_dicts
from a2ui.parser import A2uiPart, TextPart
from a2ui.processor import A2uiGenerator, CatalogConfig

from agentscope.agent import Agent
from agentscope.state import AgentState
from agentscope.credential import VolcengineCredential, DeepSeekCredential
from agentscope.event import (
    ConfirmResult,
    RequireUserConfirmEvent,
    TextBlockDeltaEvent,
    UserConfirmResultEvent,
)
from agentscope.message import Msg, ToolCallBlock, UserMsg
from agentscope.mcp import MCPClient, StdioMCPConfig
from agentscope.model import DeepSeekChatModel, VolcengineChatModel
from agentscope.permission import (
    PermissionBehavior,
    PermissionContext,
    PermissionRule,
)
from agentscope.tool import Toolkit

logger = logging.getLogger(__name__)

A2UI_EXTENSION_BASE_URI = "https://a2ui.org/a2a-extension/a2ui"
A2UI_MIME_TYPE = "application/a2ui+json"
A2UI_CAPABILITIES_METADATA_KEY = "a2uiClientCapabilities"

A2UI_CATALOG_DIR = Path(__file__).parent / "a2uiCatalog"

# The demo's own catalog lives next to the surface it renders. Registering it
# advertises it to renderers without forking the standard basic catalog.

CATALOG_CONFIGS = [
    CatalogConfig(BasicCatalog(ProtocolVersion.V0_9)),
    # CatalogConfig(BasicCatalog(ProtocolVersion.V1_0))
]

# The version of every A2UI message we emit, taken from the catalog we
# advertise so the wire format and the catalog can never drift apart.
A2UI_PROTOCOL_VERSION = CATALOG_CONFIGS[0].transformed_catalog.protocol_version

# The confirmation surface tree, kept as data so it can be rendered and
# edited on its own. Placeholders are filled in by ``_confirmation_surface``.
_CONFIRM_SURFACE_PATH = A2UI_CATALOG_DIR / "confirm.json"

A2UI_OPEN_TAG = "<a2ui-json>"
A2UI_CLOSE_TAG = "</a2ui-json>"


def _inside_block(text: str) -> bool:
    """Whether the reply so far has an unterminated A2UI JSON block."""
    return text.count(A2UI_OPEN_TAG) > text.count(A2UI_CLOSE_TAG)


CONFIRM_ACTION = "a2ui_confirm"
REJECT_ACTION = "a2ui_reject"


def _confirmation_surface(
    surface_id: str,
    reply_id: str,
    tool_names: list[str],
) -> list[dict]:
    """A2UI messages asking the user to allow or reject tool calls.

    The tree itself lives in ``a2uiCatalog/confirm.json``, so the UI can be
    edited or rendered on its own; the placeholders are filled in here. The
    buttons dispatch ``a2ui_confirm`` / ``a2ui_reject`` back to the server,
    and the ``replyId`` travels in the action context so the parked reply can
    be resumed.
    """
    template = json.loads(_CONFIRM_SURFACE_PATH.read_text(encoding="utf-8"))
    return _fill_placeholders(
        template,
        {
            "${surfaceId}": surface_id,
            "${catalogId}": CATALOG_CONFIGS[0].transformed_catalog.catalog_id,
            "${detail}": "即将运行：" + "、".join(tool_names),
            "${replyId}": reply_id,
            "${confirmAction}": CONFIRM_ACTION,
            "${rejectAction}": REJECT_ACTION,
        },
    )


def _fill_placeholders(node: Any, values: dict[str, str]) -> Any:
    """Replaces placeholder strings (``${name}``) in a parsed JSON tree."""
    if isinstance(node, dict):
        return {key: _fill_placeholders(item, values) for key, item in node.items()}
    if isinstance(node, list):
        return [_fill_placeholders(item, values) for item in node]
    if isinstance(node, str):
        return values.get(node, node)
    return node


async def _report_confirmation_required(
    event: "RequireUserConfirmEvent",
    updater: TaskUpdater,
    *,
    a2ui: bool,
) -> None:
    """Asks the client to confirm the parked tool calls.

    With A2UI the question is an interactive surface with Allow / Reject
    buttons; otherwise it degrades to a plain message. Either way the task
    ends as ``input-required`` and the client's reply resumes the reply.
    """
    names = [call.name for call in event.tool_calls]
    logger.info("Agent requires confirmation for: %s", ", ".join(names))

    if a2ui:
        surface_id = f"confirm-{event.reply_id[:8]}"
        for message in _confirmation_surface(
            surface_id,
            event.reply_id,
            names,
        ):
            await updater.add_artifact(
                [_data_part_static(message)],
                append=False,
                last_chunk=True,
            )

    text = (
        "The agent needs your approval before running: "
        f"{', '.join(names)}. Reply to confirm."
    )
    await updater.requires_input(updater.new_agent_message([Part(text=text)]))


def _data_part_static(data: dict) -> Part:
    """Builds an A2UI data Part (module-level twin of the class helper)."""
    part = Part()
    ParseDict(
        {
            "data": data,
            "metadata": {"mimeType": A2UI_MIME_TYPE},
        },
        part,
        ignore_unknown_fields=True,
    )
    return part


# Fence languages whose content is a JSON payload: models frequently wrap
# A2UI blocks in markdown code fences instead of the a2ui tags.
_JSON_FENCE_LANGS = {"", "json", "jsonc", "a2ui", "a2ui+json"}


class _FenceNormalizer:
    """Rewrites markdown JSON code fences into a2ui tags in a token stream.

    Without this, a model that wraps its payload in ```` ```json ```` fences
    leaks the raw JSON into the chat as plain text, because the A2UI parser
    only recognizes ``<a2ui-json>`` tags.
    """

    _OPEN_FENCE = re.compile(r"^`{3,}[ \t]*(\w*)[ \t]*\r?\n$")
    _CLOSE_FENCE = re.compile(r"^`{3,}[ \t]*\r?\n?$")
    # A partial line that could still grow into an opening fence line.
    _FENCE_IN_PROGRESS = re.compile(r"^`{0,3}[ \t\r]*\w{0,12}[ \t\r]*$")

    def __init__(self) -> None:
        self._buf = ""
        self._in_fence = False
        self._at_line_start = True

    def feed(self, chunk: str) -> list[str]:
        """Splits a streamed chunk into fence-normalized pieces."""
        self._buf += chunk
        out: list[str] = []
        while True:
            newline = self._buf.find("\n")
            if newline == -1:
                # No complete line: hold the buffer only while it might
                # still become a fence line; anything else passes straight
                # through so the typewriter effect stays responsive.
                if self._in_fence:
                    break
                if self._at_line_start and self._FENCE_IN_PROGRESS.match(
                    self._buf,
                ):
                    break
                if self._buf:
                    rest, self._buf = self._buf, ""
                    self._at_line_start = False
                    out.append(rest)
                break
            line, self._buf = self._buf[: newline + 1], self._buf[newline + 1 :]
            self._at_line_start = True
            if self._in_fence:
                if self._CLOSE_FENCE.match(line):
                    self._in_fence = False
                    out.append(A2UI_CLOSE_TAG)
                else:
                    out.append(line)
            else:
                match = self._OPEN_FENCE.match(line)
                if match and match.group(1).lower() in _JSON_FENCE_LANGS:
                    # A JSON fence is almost certainly an A2UI payload that
                    # the model mislabeled; non-JSON fences pass untouched.
                    self._in_fence = True
                    out.append(A2UI_OPEN_TAG)
                else:
                    out.append(line)
        return out

    def close(self) -> list[str]:
        """Flushes the held remainder, closing a fence left open."""
        out = []
        rest, self._buf = self._buf, ""
        if rest:
            out.append(rest)
        if self._in_fence:
            self._in_fence = False
            out.append(A2UI_CLOSE_TAG)
        return out


class AgentScopeExecutor(AgentExecutor):
    """Run one AgentScope agent per A2A context."""

    def __init__(self, model: str, api_key: str) -> None:
        """Remember how to build an agent for a new conversation."""
        self._model = model
        self._api_key = api_key
        self._agents: dict[str, dict[str, Agent | None]] = {}
        self._generator = A2uiGenerator(CATALOG_CONFIGS)
        # One browser for every conversation; connected lazily on the first
        # reply that needs the toolkit.
        self._playwright_client = MCPClient(
            name="playwright",
            mcp_config=StdioMCPConfig(
                command="npx",
                args=["@playwright/mcp@latest"],
            ),
            is_stateful=True,
        )
        # context_id -> (reply_id, tool calls) for a reply parked on a
        # confirmation prompt, so the client's answer can resume it.
        self._pending_confirms: dict[
            str,
            tuple[str, list[ToolCallBlock]],
        ] = {}

    def _text_system_prompt(self) -> str:
        """The prompt used when the client does not negotiate A2UI."""
        return "You are a helpful assistant named Friday."

    def _build_toolkit(self) -> Toolkit:
        """Register the shared Playwright MCP client's tools in a toolkit."""
        return Toolkit(mcps=[self._playwright_client])

    async def _allow_rules_for(self, toolkit: Toolkit) -> dict[str, list]:
        """Exact-name allow rules for every tool the toolkit carries.

        The permission engine matches rule keys by exact tool name (no
        wildcards), so the rules are generated from the live tool list.
        Without them a headless call parks the reply stream waiting for a
        confirmation nobody can answer.
        """
        rules: dict[str, list] = {}
        for group in toolkit.tool_groups:
            for client in group.mcps:
                # ``list_tools`` both filters and caches the tool list.
                for tool in await client.list_tools():
                    name = tool.name
                    rules[name] = [
                        PermissionRule(
                            tool_name=name,
                            rule_content=None,
                            behavior=PermissionBehavior.ASK,
                            source="userSettings",
                        ),
                    ]
        return rules

    async def _build_agent(
        self,
        context_id: str,
        system_prompt: str,
    ) -> Agent:
        """Build one agent bound to a prompt and a conversation."""
        toolkit = None
        try:
            # A stateful MCP client must be connected before the Toolkit
            # accepts it. Connecting is idempotent, so one browser process
            # can serve every conversation.
            await self._playwright_client.connect()
            toolkit = self._build_toolkit()
            logger.info(
                "Playwright MCP tools registered for context %s.",
                context_id,
            )
        except Exception as exc:
            # A missing MCP server must not take the whole agent down.
            logger.warning(
                "Playwright MCP unavailable; continuing without browser "
                "tools: %s",
                exc,
            )
        if toolkit is not None:
            system_prompt = (
                f"{system_prompt}\n\n"
                "# Tools\n"
                "Browser tools are available: you can open, inspect and "
                "interact with web pages in an isolated browser. Use them "
                "when the answer benefits from real web content."
            )
        allow_rules = (
            await self._allow_rules_for(toolkit) if toolkit is not None else {}
        )
        logger.info(
            "Auto-allow rules for %d tool(s) in context %s.",
            len(allow_rules),
            context_id,
        )
        return Agent(
            name="Friday",
            system_prompt=system_prompt,
            toolkit=toolkit,
            state=AgentState(
                permission_context=PermissionContext(
                    # The demo runs headless; nobody can answer a
                    # confirmation prompt, and an unanswered one parks the
                    # reply stream until the next message — surfacing as
                    # "waiting for 1 tool calls and received no event".
                    # Browsers run in an isolated Playwright profile, so
                    # auto-allowing the registered browser tools here is
                    # acceptable.
                    allow_rules=allow_rules,
                ),
            ),
            model=VolcengineChatModel(
                credential=VolcengineCredential(
                    api_key=SecretStr(self._api_key),
                    base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
                ),
                model=self._model,
                stream=True,
            ),
        )

    @staticmethod
    def _data_part(data: dict) -> Part:
        """Build an A2A Part carrying A2UI data (proto construction)."""
        return _data_part_static(data)

    def _activate_a2ui(self, context: RequestContext) -> bool:
        """Negotiates the A2UI extension a client requests and we advertise."""
        requested = {
            uri
            for uri in context.requested_extensions
            if isinstance(uri, str) and uri.startswith(A2UI_EXTENSION_BASE_URI)
        }
        advertised = {
            f"{A2UI_EXTENSION_BASE_URI}/{config.transformed_catalog.protocol_version}"
            for config in CATALOG_CONFIGS
        }
        return bool(requested & advertised)

    @staticmethod
    def _client_event_json(context: RequestContext) -> dict | None:
        """Returns the A2UI client event carried by a data part, if any."""
        message = context.message
        if message is None:
            return None
        for part in message.parts:
            if part.WhichOneof("content") == "data":
                data = MessageToDict(
                    part,
                    preserving_proto_field_name=True,
                ).get("data")
                return data if isinstance(data, dict) else None
        return None

    @staticmethod
    def _client_event_text(context: RequestContext) -> str | None:
        """Detects an A2UI client event part (e.g. a button action)."""
        data = AgentScopeExecutor._client_event_json(context)
        return json.dumps(data) if data is not None else None

    @staticmethod
    def _file_text(context: RequestContext) -> str:
        """Reads text-like file parts (e.g. a pasted attachment).

        Clients send pasted text and attachments as file parts; without this
        the agent would receive an empty message for them.
        """
        message = context.message
        if message is None:
            return ""
        chunks = []
        for part in message.parts:
            if part.WhichOneof("content") != "raw":
                continue
            media_type = (part.media_type or "").lower()
            if not (
                media_type.startswith("text/")
                or "json" in media_type
                or media_type == ""
            ):
                logger.info("Skipping binary file part (%s).", media_type)
                continue
            try:
                chunks.append(part.raw.decode("utf-8"))
            except UnicodeDecodeError:
                logger.warning("File part is not valid UTF-8; skipping.")
        return "\n".join(chunks)

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Stream the agent's reply back as A2A artifact chunks."""
        if context.task_id is None or context.context_id is None:
            raise RuntimeError("A2A server did not assign Task/context IDs.")

        # Negotiate A2UI: the extension must be both requested by the client
        # and advertised in our agent card, and the client's renderer
        # capabilities must match one of our catalogs.
        processor = None
        if self._activate_a2ui(context):
            capabilities = {}
            message = context.message
            if message is not None and message.metadata.ListFields():
                capabilities = MessageToDict(
                    message.metadata,
                    preserving_proto_field_name=True,
                ).get(A2UI_CAPABILITIES_METADATA_KEY) or {}
            if not capabilities:
                capabilities = context.metadata.get(
                    A2UI_CAPABILITIES_METADATA_KEY,
                ) or {}
            try:
                processor = self._generator.create_processor(capabilities)
            except Exception:
                # Capability negotiation failed; fall back to plain text.
                processor = None

        conversation = self._agents.setdefault(
            context.context_id,
            {"text": None, "a2ui": None},
        )
        agent_key = "a2ui" if processor is not None else "text"
        if conversation[agent_key] is None:
            system_prompt = self._text_system_prompt()
            if processor is not None:
                system_prompt = "\n\n".join(
                    [
                        system_prompt,
                        (
                            "# A2UI\n"
                            "When a response benefits from interactive UI, "
                            "emit one or more <a2ui-json> blocks following "
                            "the format below. Combine conversational text "
                            "with UI blocks when that helps the user."
                        ),
                        processor.prompt_snippet,
                    ],
                )
            conversation[agent_key] = await self._build_agent(
                context.context_id,
                system_prompt,
            )
        agent = conversation[agent_key]
        assert agent is not None

        # The input is the plain text of the message; A2UI client events
        # (renderer actions) ride in data parts and are forwarded as JSON so
        # the agent can continue the UI conversation.
        user_input = context.get_user_input()
        file_text = self._file_text(context)
        client_event = self._client_event_text(context)
        if file_text:
            user_input = (
                f"{user_input}\n\n{file_text}" if user_input else file_text
            )
        content = (
            f"{user_input}\n{client_event}"
            if processor is not None and client_event and user_input
            else (client_event or user_input)
        )
        if not content.strip():
            logger.warning("Empty message received; nothing to send to agent.")

        # A button on the confirmation surface resumes the parked reply
        # with the user's decision instead of a new chat message.
        inputs: Msg | UserConfirmResultEvent = UserMsg(
            name="user",
            content=content,
        )
        event_json = self._client_event_json(context)
        action_name = (
            (event_json or {}).get("action", {}).get("name")
            if event_json
            else None
        )
        if action_name in (CONFIRM_ACTION, REJECT_ACTION):
            pending = self._pending_confirms.pop(context.context_id, None)
            if pending is None:
                logger.warning(
                    "Confirmation action %s with no parked reply.",
                    action_name,
                )
            else:
                reply_id, tool_calls = pending
                inputs = UserConfirmResultEvent(
                    reply_id=reply_id,
                    confirm_results=[
                        ConfirmResult(
                            confirmed=action_name == CONFIRM_ACTION,
                            tool_call=call,
                            rules=(
                                call.suggested_rules
                                if action_name == CONFIRM_ACTION
                                else None
                            ),
                        )
                        for call in tool_calls
                    ],
                )
                logger.info(
                    "Resuming parked reply %s with %s.",
                    reply_id,
                    action_name,
                )

        # The Task itself must reach the client before any update to it.
        await event_queue.enqueue_event(
            Task(
                id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
            ),
        )
        updater = TaskUpdater(
            event_queue,
            task_id=context.task_id,
            context_id=context.context_id,
        )
        await updater.start_work()

        parser = processor.create_parser() if processor else None

        text_stream_state = {"started": False}

        async def _emit_text(text: str, *, last_chunk: bool) -> None:
            """Send one streamed text chunk as an artifact."""
            await updater.add_artifact(
                [Part(text=text)],
                artifact_id=f"{context.task_id}-reply",
                append=text_stream_state["started"],
                last_chunk=last_chunk,
            )
            text_stream_state["started"] = True

        full_reply = ""  # Raw model output, kept for diagnostics.
        norm_reply = ""  # Fence-normalized output, drives block detection.
        normalizer = _FenceNormalizer() if parser is not None else None
        json_guard_done = False

        async def _emit_parsed(parts) -> None:
            """Forward parser output: text artifacts and A2UI data artifacts."""
            for part in parts:
                if isinstance(part, TextPart):
                    if part.text and "</a2ui-json>" not in part.text:
                        await _emit_text(part.text, last_chunk=False)
                elif isinstance(part, A2uiPart):
                    # Each renderer-bound message is its own artifact so
                    # the client applies updates in order.
                    for message in to_message_dicts(part.a2ui):
                        logger.info("Emitting A2UI message: %s", message)
                        await updater.add_artifact(
                            [self._data_part(message)],
                            append=False,
                            last_chunk=True,
                        )

        async for event in agent.reply_stream(inputs):
            if isinstance(event, RequireUserConfirmEvent):
                # The agent parked the reply awaiting a human. Dropping
                # this event would silently end the stream and leave the
                # tool call ASKING, so the next message would fail with
                # "waiting for 1 tool calls ... received no event". Ask the
                # client instead, remembering what to resume.
                self._pending_confirms[context.context_id] = (
                    event.reply_id,
                    list(event.tool_calls),
                )
                await _report_confirmation_required(
                    event,
                    updater,
                    a2ui=processor is not None,
                )
                if text_stream_state["started"]:
                    await _emit_text("", last_chunk=True)
                return
            if not isinstance(event, TextBlockDeltaEvent):
                continue
            delta = event.delta
            if parser is None:
                # Plain text mode: pipeline the chunks as-is.
                await _emit_text(delta, last_chunk=False)
                continue
            full_reply += delta
            pieces = normalizer.feed(delta) if normalizer else [delta]
            for piece in pieces:
                # A reply opening with bare JSON is an A2UI block without
                # its tags; wrap it so the parser sees one. Prose replies
                # never start with an opening brace, so this stays safe.
                if not json_guard_done and piece.strip():
                    json_guard_done = True
                    if piece.lstrip()[:1] in ("{", "["):
                        piece = A2UI_OPEN_TAG + piece
                norm_reply += piece
                # A2UI mode: text passes through, <a2ui-json> blocks are
                # parsed and validated incrementally as they complete.
                try:
                    await _emit_parsed(parser.parse_chunk(piece))
                except Exception as exc:
                    # A malformed block must not kill the reply. Emitting
                    # the raw piece from inside a block would leak JSON and
                    # tag fragments into the chat, so those are dropped.
                    logger.warning("A2UI chunk rejected, skipping: %s", exc)
                    if not _inside_block(norm_reply):
                        await _emit_text(piece, last_chunk=False)

        # Full model output, logged so a dropped block can be diagnosed.
        logger.info("Raw model reply: %r", full_reply)

        if parser is not None:
            if normalizer is not None:
                for piece in normalizer.close():
                    norm_reply += piece
                    try:
                        await _emit_parsed(parser.parse_chunk(piece))
                    except Exception as exc:
                        logger.warning("A2UI tail rejected: %s", exc)
            # The model may forget the closing tag or the closing fence.
            # Only then do we inject one; injecting it unconditionally would
            # leak the tag into the reply text.
            if _inside_block(norm_reply):
                try:
                    await _emit_parsed(parser.parse_chunk(A2UI_CLOSE_TAG))
                except Exception:
                    # The reply must complete even if the tail is invalid.
                    pass
            # Flush text the parser held back as a possible tag prefix.
            try:
                await _emit_parsed(parser.parse_chunk(""))
            except Exception:
                pass
        if text_stream_state["started"]:
            await _emit_text("", last_chunk=True)
        await updater.complete()

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Mark the Task canceled at the client's request."""
        if context.task_id is None or context.context_id is None:
            raise RuntimeError("Cannot cancel a Task without IDs.")
        await TaskUpdater(
            event_queue,
            task_id=context.task_id,
            context_id=context.context_id,
        ).cancel()


def create_app(base_url: str, model: str, api_key: str) -> Starlette:
    """Build the A2A 1.0 application serving the agent."""
    supported_catalog_ids = [
        config.transformed_catalog.catalog_id for config in CATALOG_CONFIGS
    ]
    a2ui_extension = AgentExtension()
    ParseDict(
        {
            "uri": f"{A2UI_EXTENSION_BASE_URI}/{A2UI_PROTOCOL_VERSION}",
            "description": (
                "Provides agent driven UI using the A2UI JSON format."
            ),
            "params": {"supportedCatalogIds": supported_catalog_ids},
        },
        a2ui_extension,
        ignore_unknown_fields=True,
    )
    card = AgentCard(
        name="Friday",
        description="An AgentScope assistant exposed over A2A 1.0.",
        version="1.0.0",
        supported_interfaces=[
            AgentInterface(
                url=base_url,
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            ),
        ],
        capabilities=AgentCapabilities(
            streaming=True,
            extensions=[a2ui_extension],
        ),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="chat",
                name="Chat",
                description="Answer questions in a multi-turn conversation.",
                tags=["chat"],
            ),
        ],
    )
    handler = DefaultRequestHandler(
        agent_executor=AgentScopeExecutor(model=model, api_key=api_key),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    app = Starlette(
        routes=[
            *create_agent_card_routes(card),
            *create_jsonrpc_routes(
                handler,
                rpc_url="/",
                # Browser/mobile clients in this repo speak the A2A v0.3
                # wire method names ("message/send", "message/stream", ...).
                enable_v0_3_compat=True,
            ),
        ],
    )
    # Web clients (e.g. the Flutter A2A demo running in a browser) send an
    # OPTIONS preflight before cross-origin POSTs; without this middleware
    # the server answers 405 and the browser blocks the request.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        max_age=3600,
    )
    return app


async def main() -> None:
    """The main entry point of the server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ark-code-latest")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    args = parser.parse_args()

    api_key = os.environ.get('ARK_API_KEY')
    if not api_key:
        raise RuntimeError(
            "Set the DS_API_KEY environment variable before "
            "running this demo.",
        )

    app = create_app(
        base_url=f"http://{args.host}:{args.port}",
        model=args.model,
        api_key=api_key,
    )
    await uvicorn.Server(
        uvicorn.Config(app, host=args.host, port=args.port),
    ).serve()


if __name__ == "__main__":
    asyncio.run(main())
