# -*- coding: utf-8 -*-
"""Expose an AgentScope agent as an A2A 1.0 server with A2UI 1.0 output.

The reply streams as A2A messages (agent ``Message`` objects attached to
``working`` status updates) instead of artifacts: plain text goes out as text
parts, and any ``<a2ui-json>`` block the model emits is parsed with the
``a2ui`` SDK and sent as an A2UI 1.0 DataPart. Agent activity (thinking,
tool calls, human confirmation) is also published as message parts using the
custom-catalog surfaces. Human confirmation pauses the A2A Task in
``input-required`` until the renderer sends an A2UI action. Each A2A
``context_id`` gets its own AgentScope agent. Run with::

    export DEEPSAKE_API_KEY=...
    python server.py [--model deepsake-flash] [--port 9999]
"""
import argparse
import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from google.protobuf import json_format, struct_pb2
from pydantic import SecretStr
import uvicorn

from a2a.compat.v0_3 import types as types_v03
from a2a.compat.v0_3.conversions import to_core_part
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
    Message,
    Part,
    Task,
    TaskState,
    TaskStatus,
)
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from agentscope.agent import Agent
from agentscope.model import DeepSeekChatModel,VolcengineChatModel
from agentscope.credential import DeepSeekCredential,VolcengineCredential

from agentscope.event import (
    ConfirmResult,
    RequireUserConfirmEvent,
    UserConfirmResultEvent,
    TextBlockDeltaEvent,
    ThinkingBlockStartEvent,
    ThinkingBlockDeltaEvent,
    ThinkingBlockEndEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultDataDeltaEvent,
    ToolResultEndEvent,
    ToolResultStartEvent,
    ToolResultTextDeltaEvent,

)
from agentscope.mcp import MCPClient, StdioMCPConfig
from agentscope.message import Msg, ToolCallBlock, ToolCallState, UserMsg
from agentscope.tool import Bash, Edit, Glob, Read, Toolkit, Write

from a2ui.constants import A2UI_ACTIONS_KEY, A2UI_CLIENT_CAPABILITIES_KEY
from a2ui.core.basic_catalog import BasicCatalog
from a2ui.core.schema import ProtocolVersion
from a2ui.inference_formats import to_message_dicts
from a2ui.parser import A2uiPart, Parser, ResponsePart, TextPart
from a2ui.processor import A2uiGenerator, CatalogConfig


A2UI_VERSION = "v1.0"
A2UI_MIME_TYPE = "application/a2ui+json"
# v0.8 and v0.9 renderers tag their parts with this deprecated type.
A2UI_DEPRECATED_MIME_TYPE = "application/json+a2ui"
A2UI_MIME_TYPES = (A2UI_MIME_TYPE, A2UI_DEPRECATED_MIME_TYPE)
A2UI_EXTENSION_URI = f"https://a2ui.org/a2a-extension/a2ui/{A2UI_VERSION}"
CATALOG_IDS_KEY = "supportedCatalogIds"
RENDERER_CAPABILITIES_KEYS = (
    "a2uiRendererCapabilities",
    A2UI_CLIENT_CAPABILITIES_KEY,
)
CONFIRM_ACTION = "agentscope.tool.confirm"
REJECT_ACTION = "agentscope.tool.reject"

logger = logging.getLogger(__name__)

_BASIC_CATALOG = BasicCatalog(ProtocolVersion.V1_0)
_AGENT_CATALOG = CatalogConfig.from_path(
    str(Path(__file__).parent / "a2uiCatalog" / "catalog.json"),
)
AGENT_CATALOG_ID = _AGENT_CATALOG.catalog.catalog_id
SUPPORTED_CATALOG_IDS = [_BASIC_CATALOG.catalog_id, AGENT_CATALOG_ID]

_processor = A2uiGenerator(
    [CatalogConfig(_BASIC_CATALOG), _AGENT_CATALOG],
).create_processor({A2UI_VERSION: {CATALOG_IDS_KEY: SUPPORTED_CATALOG_IDS}})

INSTRUCTION = "\n\n".join([
    "You are a helpful assistant named Friday.",
    _processor.prompt_snippet,
])


def _a2ui_to_part(a2ui: Any) -> Part:
    """Encode one or more A2UI 1.0 messages as an A2A DataPart.

    ``ParseDict`` accepts a dict or a list at runtime, but its type stub
    only declares ``dict[str, Any]``.
    """
    return Part(
        data=json_format.ParseDict(a2ui, struct_pb2.Value()),
        metadata=json_format.ParseDict(
            {"mimeType": A2UI_MIME_TYPE},
            struct_pb2.Struct(),
        ),
        media_type=A2UI_MIME_TYPE,
    )


def _a2ui_payload(part: Part | types_v03.Part | None) -> Any:
    """Return the JSON payload of a 1.0 or legacy 0.3 A2UI DataPart."""
    if part is None:
        return None
    if isinstance(part, types_v03.Part):
        part = to_core_part(part)
    if part.WhichOneof("content") != "data":
        return None
    metadata = json_format.MessageToDict(part.metadata)
    mime_type = metadata.get("mimeType") or part.media_type
    if mime_type not in A2UI_MIME_TYPES:
        return None
    return json_format.MessageToDict(part.data)


def _to_a2a_parts(response_parts: Sequence[ResponsePart]) -> list[Part]:
    """Convert parsed model output into text parts and A2UI DataParts."""
    parts: list[Part] = []
    for response_part in response_parts:
        if isinstance(response_part, TextPart):
            if response_part.text:
                parts.append(Part(text=response_part.text))
        elif isinstance(response_part, A2uiPart):
            for message in to_message_dicts(response_part.a2ui):
                parts.append(_a2ui_to_part(message))
    return parts


def _create_parser() -> Parser:
    """Create a fresh parser: one parser reads exactly one response.

    ``create_parser`` exists in the installed SDK; newer SDKs expose the
    parser as a ``parser`` property instead, so both are accepted.
    """
    factory = getattr(_processor, "create_parser", None)
    if callable(factory):
        return cast(Parser, factory())
    return cast(Parser, getattr(_processor, "parser"))


def _a2ui_extension() -> AgentExtension:
    """Advertise the A2UI 1.0 extension and the catalogs this agent produces."""
    return AgentExtension(
        uri=A2UI_EXTENSION_URI,
        description="Provides agent driven UI using the A2UI JSON format.",
        required=False,
        params=json_format.ParseDict(
            {CATALOG_IDS_KEY: SUPPORTED_CATALOG_IDS},
            struct_pb2.Struct(),
        ),
    )


def _create_agent_card(base_url: str) -> AgentCard:
    """Build the public AgentCard, including A2UI 1.0 extension support."""
    return AgentCard(
        name="Friday",
        description=(
            "An AgentScope assistant exposed over A2A 1.0 with A2UI 1.0 "
            "activity and confirmation surfaces."
        ),
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
            extensions=[_a2ui_extension()],
        ),
        default_input_modes=["text/plain", A2UI_MIME_TYPE],
        default_output_modes=["text/plain", A2UI_MIME_TYPE],
        skills=[
            AgentSkill(
                id="chat",
                name="Chat",
                description="Answer questions in a multi-turn conversation.",
                tags=["chat"],
            ),
        ],
    )


def _a2a_message(
    context_or_message: RequestContext | Message,
) -> Message | None:
    """Return the A2A message behind a request context, or the message."""
    if isinstance(context_or_message, Message):
        return context_or_message
    return context_or_message.message


def _renderer_capabilities(
    context_or_message: RequestContext | Message,
) -> dict[str, Any] | None:
    """Read the renderer's own A2UI capabilities from the message metadata."""
    message = _a2a_message(context_or_message)
    if message is None:
        return None
    metadata = json_format.MessageToDict(message.metadata)
    for key in RENDERER_CAPABILITIES_KEYS:
        capabilities = metadata.get(key)
        if capabilities is None:
            continue
        if isinstance(capabilities, str):
            try:
                capabilities = json.loads(capabilities)
            except ValueError:
                return None
        if isinstance(capabilities, dict):
            return capabilities
    return None


def _renderer_supports_a2ui(
    context_or_message: RequestContext | Message,
) -> bool:
    """Use A2UI only when the renderer negotiates the extension or the catalog."""
    if isinstance(context_or_message, RequestContext) and (
        A2UI_EXTENSION_URI in (context_or_message.requested_extensions or [])
    ):
        return True

    capabilities = _renderer_capabilities(context_or_message)
    if capabilities is None:
        return False
    version_capabilities = capabilities.get(A2UI_VERSION, {})
    supported = (
        version_capabilities.get(CATALOG_IDS_KEY, [])
        if isinstance(version_capabilities, dict)
        else []
    )
    return AGENT_CATALOG_ID in supported


def _extract_a2ui_actions(
    context_or_message: RequestContext | Message,
) -> list[dict[str, Any]]:
    """Read renderer actions from the A2UI DataParts in the A2A message."""
    message = _a2a_message(context_or_message)
    if message is None:
        return []

    actions: list[dict[str, Any]] = []
    for part in message.parts:
        payload = _a2ui_payload(part)
        if payload is None:
            continue
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            # v1.0 renderers send `action`, older ones `userAction`.
            action = item.get(A2UI_ACTIONS_KEY) or item.get("userAction")
            if not isinstance(action, dict) or not isinstance(
                action.get("name"),
                str,
            ):
                continue
            action_context = action.get("context", {})
            actions.append({
                "name": action["name"],
                "context": action_context if isinstance(action_context, dict) else {},
            })
    return actions


def _action_query(actions: list[dict[str, Any]], text_input: str) -> str:
    """Describe renderer actions as text the agent can answer."""
    lines = [
        f'A2UI action "{action["name"]}" with context '
        f"{json.dumps(action['context'], ensure_ascii=False)}"
        for action in actions
    ]
    if text_input.strip():
        lines.append(text_input.strip())
    return "\n".join(lines)



def _stable_surface_id(kind: str, context_id: str, item_id: str = "") -> str:
    """Create a stable, renderer-safe ID without exposing arbitrary tool IDs."""
    key = f"{kind}:{context_id}:{item_id}"
    return f"agentscope-{kind}-{uuid.uuid5(uuid.NAMESPACE_URL, key).hex}"


def _tool_component(
    *,
    name: str,
    status: str,
    arguments: str = "",
    result: str = "",
    status_text: str | None = None,
) -> dict[str, Any]:
    """Build a ToolCall component using the custom catalog's property names."""
    component: dict[str, Any] = {
        "id": "root",
        "component": "ToolCall",
        "name": name,
        "status": status,
    }
    if arguments:
        component["arguments"] = arguments
    if result:
        component["result"] = result
    if status_text:
        component["statusText"] = status_text
    return component


def _thinking_component(text: str, status: str) -> dict[str, Any]:
    """Build a ThinkingBar component."""
    return {
        "id": "root",
        "component": "ThinkingBar",
        "text": text,
        "status": status,
    }


def _human_confirm_component(tool_call: ToolCallBlock) -> dict[str, Any]:
    """Build a confirmation card whose actions identify the pending tool call."""
    return {
        "id": "root",
        "component": "HumanConfirm",
        "title": f"Allow {tool_call.name}?",
        "message": (
            f"The agent wants to run {tool_call.name} with these arguments:\n"
            f"{tool_call.input or '{}'}"
        ),
        "severity": "warning",
        "confirmLabel": "Allow",
        "cancelLabel": "Reject",
        "confirmAction": {
            "event": {
                "name": CONFIRM_ACTION,
                "context": {"toolCallId": tool_call.id},
            },
        },
        "cancelAction": {
            "event": {
                "name": REJECT_ACTION,
                "context": {"toolCallId": tool_call.id},
            },
        },
    }


async def _emit_agent_message(
    updater: TaskUpdater,
    parts: Sequence[Part],
    extensions: Sequence[str] = (),
) -> None:
    """Send agent output as an A2A Message on a working status update."""
    message = updater.new_agent_message(list(parts))
    if extensions:
        message.extensions.extend(extensions)
    await updater.update_status(
        TaskState.TASK_STATE_WORKING,
        message=message,
    )



class AgentScopeExecutor(AgentExecutor):
    """Run one AgentScope agent per A2A context."""

    def __init__(
        self,
        model: str,
        api_key: str,
        card: AgentCard | None = None,
    ) -> None:
        """Remember how to build an agent for a new conversation."""
        self._card = card
        self._model = model
        self._api_key = api_key
        self._agents: dict[str, Agent] = {}
        self._mcp_clients: dict[str, MCPClient] = {}
        self._a2ui_surfaces: dict[str, set[str]] = {}
        self._a2ui_contexts: set[str] = set()



    async def _emit_component(
        self,
        context_id: str,
        updater: TaskUpdater,
        surface_id: str,
        component: dict[str, Any],
    ) -> None:
        """Create or update a custom-catalog component surface."""
        surfaces = self._a2ui_surfaces.setdefault(context_id, set())
        envelope = {}
        if surface_id in surfaces:
            envelope = {
                "version": A2UI_VERSION,
                "updateComponents": {
                    "surfaceId": surface_id,
                    "components": [component],
                },
            }
        else:
            envelope = {
                "version": A2UI_VERSION,
                "createSurface": {
                    "surfaceId": surface_id,
                    "catalogId": AGENT_CATALOG_ID,
                    "components": [component],
                },
            }
        await _emit_agent_message(
            updater,
            [_a2ui_to_part(envelope)],
            extensions=[A2UI_EXTENSION_URI],
        )
        self._a2ui_surfaces[context_id].add(surface_id)

    async def _emit_confirmation(
        self,
        context_id: str,
        updater: TaskUpdater,
        confirmation: RequireUserConfirmEvent,
        supports_a2ui: bool,
    ) -> None:
        """Present the confirmation and park the A2A task at input-required."""
        if supports_a2ui:
            for tool_call in confirmation.tool_calls:
                await self._emit_component(
                    context_id,
                    updater,
                    _stable_surface_id("confirm", context_id, tool_call.id),
                    _human_confirm_component(tool_call),
                )
            await updater.requires_input()
        else:
            tool_call = confirmation.tool_calls[0]
            prompt = (
                f"Allow tool {tool_call.name} with arguments "
                f"{tool_call.input or '{}'}? Reply yes or no."
            )
            await updater.requires_input(
                updater.new_agent_message([Part(text=prompt)]),
            )

    async def _emit_delete_surfaces(
        self,
        context_id: str,
        updater: TaskUpdater,
        confirmresultevent: UserConfirmResultEvent | None,
    ) -> None:
        """Remove confirmation cards after their result is accepted."""
        surfaces = self._a2ui_surfaces.setdefault(context_id, set())
        for result in confirmresultevent.confirm_results if confirmresultevent is not None else []:
            surface_id = _stable_surface_id(
                "confirm",
                context_id,
                result.tool_call.id,
            )
            if surface_id not in surfaces:
                continue
            await _emit_agent_message(
                updater,
                [
                    _a2ui_to_part([
                        {
                            "version": A2UI_VERSION,
                            "deleteSurface": {"surfaceId": surface_id},
                        },
                    ]),
                ],
                extensions=[A2UI_EXTENSION_URI],
            )
            surfaces.remove(surface_id)





    async def _build_agent(
        self,
        context_id: str,
        supports_a2ui: bool,
    ) -> Agent:
        """Build an agent with local tools and an isolated Playwright MCP."""
        workdir = str(Path.cwd().resolve())
        try:
            toolkit = Toolkit(
                tools=[
                    Bash(cwd=workdir),
                    Read(),
                    Write(),
                    Edit(),
                    Glob(),
                ],
            )
            agent = Agent(
                name="Friday",
                system_prompt="\n\n".join([
                    (
                        INSTRUCTION
                        if supports_a2ui
                        else "You are a helpful assistant named Friday."
                    ),
                    f"The local working directory is {workdir}.",
                    (
                        "Use Bash, Read, Write, Edit, and Glob for local "
                        "development tasks. Use Playwright only for websites "
                        "and applications the user is authorized to test. "
                        "Do not submit forms or perform consequential actions "
                        "without the user's explicit approval."
                    ),
                ]),
                model=DeepSeekChatModel(
                    credential=DeepSeekCredential(
                        api_key=SecretStr(self._api_key),
                        # base_url= "https://api.deepseek.com",
                    ),
                    model=self._model,
                    stream=True,
                ),
                toolkit=toolkit,
            )
        except BaseException:
            raise

        return agent


    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Stream text/activity and suspend for explicit human confirmations."""
        if context.task_id is None or context.context_id is None:
            raise RuntimeError("A2A server did not assign Task/context IDs.")

        context_id = context.context_id
        task_id = context.task_id
        # A follow-up message (an A2UI action, say) may omit the capabilities,
        # so once a conversation uses A2UI it keeps using it.
        supports_a2ui = _renderer_supports_a2ui(context) or (
            context_id in self._a2ui_contexts
        )
        if supports_a2ui:
            self._a2ui_contexts.add(context_id)
        parser = _create_parser() if supports_a2ui else None
        tool_progress: dict[str, dict[str, str]] = {}
        agent = self._agents.get(context_id)
        if agent is None:
            agent = await self._build_agent(context_id, supports_a2ui)
            self._agents[context_id] = agent
        if context.current_task is None:
            await event_queue.enqueue_event(
                Task(
                    id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                ),
            )
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.start_work()
        actions = _extract_a2ui_actions(context)
        user_input = context.get_user_input()
        if actions:
                # The renderer clicked a surface the model generated, so the
                # action is a normal user turn, not a tool decision.
            logger.info(
                "A2UI action(s) without a pending confirmation: %s",
                 [action["name"] for action in actions],
                )
            user_input = _action_query(actions, user_input)
        agent_input = UserMsg(name="user", content=user_input)
        streaming = bool(
            parser is not None
            and getattr(parser, "parse_chunk", None)
            and getattr(parser, "supports_streaming", False)
        )
        buffered = ""

        async def emit_part(part: Part) -> None:
            """Send one response part as an agent message."""
            if part.WhichOneof("content") == "text":
                if not part.text:
                    return
                await _emit_agent_message(updater, [part])
                return
            await _emit_agent_message(
                updater,
                [part],
                extensions=[A2UI_EXTENSION_URI],
            )

        async def emit_text(text: str) -> None:
            """Stream model text, splitting out the A2UI blocks it carries."""
            nonlocal buffered
            if parser is None:
                await emit_part(Part(text=text))
                return
            if not streaming:
                buffered += text
                return
            for part in _to_a2a_parts(parser.parse_chunk(text)):
                await emit_part(part)

        async def emit_tool(tool_call_id: str) -> None:
            state = tool_progress[tool_call_id]
            await self._emit_component(
                context_id,
                updater,
                _stable_surface_id("tool", context_id, tool_call_id),
                _tool_component(
                    name=state["name"],
                    status=state["status"],
                    arguments=state["arguments"],
                    result=state["result"],
                    status_text=state.get("statusText"),
                ),
            )


  
        async for event in agent.reply_stream(agent_input):
            if isinstance(event, TextBlockDeltaEvent):
                await emit_text(event.delta)
            elif isinstance(event, ThinkingBlockStartEvent):
                if supports_a2ui:
                    await self._emit_component(
                        context_id,
                        updater,
                        _stable_surface_id(event.block_id, context_id,event.reply_id),
                        _thinking_component("Thinking…", "thinking"),
                    )
            elif isinstance(event, ThinkingBlockDeltaEvent):
                if supports_a2ui:
                    # await self._emit_component(
                    #     context_id,
                    #     updater,
                    #     _stable_surface_id(event.block_id, context_id,event.reply_id),
                    #     _thinking_component(event.delta, "thinking"),
                    #     )
                    await emit_text(event.delta)
            elif isinstance(event, ThinkingBlockEndEvent):
                if supports_a2ui:
                    await self._emit_component(
                        context_id,
                        updater,
                        _stable_surface_id(event.block_id, context_id,event.reply_id),
                        _thinking_component("Thinking Done", "done"),
                        )           
            elif isinstance(event, ToolCallStartEvent):
                tool_progress[event.tool_call_id] = {
                    "name": event.tool_call_name,
                    "status": "pending",
                    "arguments": "",
                    "result": "",
                }
                if supports_a2ui:
                    await emit_tool(event.tool_call_id)
            elif isinstance(event, ToolCallDeltaEvent):
                tool_progress[event.tool_call_id]["arguments"] += event.delta
            elif isinstance(event, ToolCallEndEvent):
                tool_progress[event.tool_call_id]["status"] = "running"
                if supports_a2ui:
                    await emit_tool(event.tool_call_id)
            elif isinstance(event, ToolResultStartEvent):
                state = tool_progress.setdefault(
                    event.tool_call_id,
                    {
                        "name": event.tool_call_name,
                        "status": "running",
                        "arguments": "",
                        "result": "",
                    },
                )
                state["name"] = event.tool_call_name
                state["status"] = "running"
                if supports_a2ui:
                    await emit_tool(event.tool_call_id)
            elif isinstance(event, ToolResultTextDeltaEvent):
                if event.tool_call_id in tool_progress:
                    tool_progress[event.tool_call_id]["result"] += event.delta
            elif isinstance(event, ToolResultDataDeltaEvent):
                if event.tool_call_id in tool_progress:
                    tool_progress[event.tool_call_id]["result"] += (
                        f"[{event.media_type} content omitted]"
                    )
            elif isinstance(event, ToolResultEndEvent):
                if event.tool_call_id in tool_progress:
                    state = tool_progress[event.tool_call_id]
                    result_state = getattr(event.state, "value", event.state)
                    state["status"] = (
                        "success" if result_state == "success" else "error"
                    )
                    if state["status"] == "error":
                        state["statusText"] = str(result_state).title()
                    if supports_a2ui:
                        await emit_tool(event.tool_call_id)
            elif isinstance(event, RequireUserConfirmEvent):
                for tool_call in event.tool_calls:
                    tool_progress[tool_call.id] = {
                        "name": tool_call.name,
                        "status": "pending",
                        "arguments": tool_call.input,
                        "result": "",
                    }
                    if supports_a2ui:
                        await emit_tool(tool_call.id)
                await self._emit_confirmation(
                    context_id,
                    updater,
                    event,
                    supports_a2ui,
                )
            elif isinstance(event, UserConfirmResultEvent):
                if supports_a2ui:
                    await self._emit_delete_surfaces(
                        updater=updater,
                        context_id=context_id, 
                        confirmresultevent=event )
               



        if parser is not None and not streaming and buffered:
            for part in _to_a2a_parts(_processor.parse_response(buffered)):
                await emit_part(part)


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
            context.task_id,
            context.context_id,
        ).cancel()

    async def aclose(self) -> None:
        """Close all per-conversation MCP connections."""
        clients = list(self._mcp_clients.values())
        self._mcp_clients.clear()
        for client in clients:
            await client.close()



def create_app(
    base_url: str,
    model: str,
    api_key: str,
    cors_origins: list[str] | None = None,
) -> Starlette:
    """Build the A2A 1.0 application serving the agent.

    Browser renderers send a CORS preflight ``OPTIONS`` before posting a
    JSON-RPC request, so the app answers preflights for the origins it is
    configured with.
    """
    card = _create_agent_card(base_url)
    executor = AgentScopeExecutor(card=card, model=model, api_key=api_key)

    @asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncGenerator[None, None]:
        try:
            yield
        finally:
            await executor.aclose()

    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    return Starlette(
        routes=[
            *create_agent_card_routes(card),
            *create_jsonrpc_routes(
                handler,
                rpc_url="/",
                enable_v0_3_compat=True,
            ),
        ],
        middleware=[
            Middleware(
                CORSMiddleware,
                allow_origins=cors_origins or ["*"],
                allow_methods=["GET", "POST", "OPTIONS"],
                allow_headers=["*"],
                expose_headers=["X-A2A-Extensions"],
                max_age=600,
            ),
        ],
        lifespan=lifespan,
    )


async def main() -> None:
    """The main entry point of the server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument(
        "--cors-origin",
        action="append",
        dest="cors_origins",
        metavar="ORIGIN",
        help=(
            "Allow this origin in CORS preflights (repeatable, e.g. "
            "http://localhost:5173). Defaults to every origin."
        ),
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key: 
        raise RuntimeError(
            "Set the DEEPSEEK_API_KEY environment variable before running "
            "this demo.",
        )

    app = create_app(
        base_url=f"http://{args.host}:{args.port}",
        model=args.model,
        api_key=api_key,
        cors_origins=args.cors_origins,
    )
    await uvicorn.Server(
        uvicorn.Config(app, host=args.host, port=args.port),
    ).serve()


if __name__ == "__main__":
    asyncio.run(main())
