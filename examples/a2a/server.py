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
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes,create_rest_routes,add_a2a_routes_to_fastapi
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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


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
from a2ui.core import A2uiError
from a2ui.parser import (
    A2uiPart,
    ErrorPart,
    Parser,
    ResponsePart,
    StreamPart,
    TextPart,
)
from a2ui.processor import A2uiGenerator, CatalogConfig, A2uiRequestProcessor


A2UI_VERSION = "1.0"
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


_BASIC_CATALOG = BasicCatalog(ProtocolVersion.V1_0)
_AGENT_CATALOG = CatalogConfig.from_path(
    str(Path(__file__).parent / "a2uiCatalog" / "catalog.json"),
)
AGENT_CATALOG_ID = _AGENT_CATALOG.catalog.catalog_id
SUPPORTED_CATALOG_IDS = [_BASIC_CATALOG.catalog_id, AGENT_CATALOG_ID]

_processor = A2uiGenerator(
    [CatalogConfig(_BASIC_CATALOG), _AGENT_CATALOG],
).create_processor({A2UI_VERSION: {CATALOG_IDS_KEY: SUPPORTED_CATALOG_IDS}})

# The processor exposes a stateless cached ``parser`` property. A parser
# holds no state, so reusing it across responses is safe.
parser = cast(Parser, _processor.parser)

logger = logging.getLogger(__name__)

def _a2ui_to_part(a2ui: Any) -> Part:
    """Encode a single A2UI 1.0 message as an A2A DataPart.

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


def _part_to_a2ui(part: Part | types_v03.Part | None) -> Any:
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


def _stream_part_to_a2a_parts(stream_part: StreamPart) -> list[Part]:
    """Convert one response-stream part into A2A parts.

    Stream TextParts are skipped: ``emit_text`` already forwards the raw model
    text, so only the compiled A2UI blocks need to be sent towards the end.
    """
    if isinstance(stream_part, A2uiPart):
        return [
            _a2ui_to_part(message)
            for message in to_message_dicts(stream_part.a2ui)
        ]
    if isinstance(stream_part, ErrorPart):
        logger.warning("A2UI stream rejected a block: %s", stream_part.error)
    return []



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


def _renderer_is_supports_a2ui(
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
        payload = _part_to_a2ui(part)
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


def _action_to_query(
    actions: list[dict[str, Any]],
    text_input: str,
    pending_confirmation: RequireUserConfirmEvent | None = None,
) -> str | UserConfirmResultEvent:
    """Describe renderer actions as text the agent can answer.

    CONFIRM_ACTION/REJECT_ACTION actions that resolve a pending
    :class:`RequireUserConfirmEvent` are converted into a
    :class:`UserConfirmResultEvent` instead of a text turn, so the agent
    can resume the parked reply with the user's per-tool-call decision.
    """
    if pending_confirmation is not None:
        confirm_ids = {
            action["context"].get("toolCallId")
            for action in actions
            if action["name"] == CONFIRM_ACTION
        }
        reject_ids = {
            action["context"].get("toolCallId")
            for action in actions
            if action["name"] == REJECT_ACTION
        }
        if confirm_ids or reject_ids:
            confirm_results = []
            for tool_call in pending_confirmation.tool_calls:
                if tool_call.id in confirm_ids:
                    confirm_results.append(
                        ConfirmResult(confirmed=True, tool_call=tool_call),
                    )
                elif tool_call.id in reject_ids:
                    confirm_results.append(
                        ConfirmResult(confirmed=False, tool_call=tool_call),
                    )
            if confirm_results:
                return UserConfirmResultEvent(
                    reply_id=pending_confirmation.reply_id,
                    confirm_results=confirm_results,
                )
        tool_call_ids = {t.id for t in pending_confirmation.tool_calls}
        if any(tool_call_ids & (confirm_ids | reject_ids)):
            logger.warning(
                "Ignored A2UI confirm/reject actions for pending tool calls.",
            )
    lines = [
        f'A2UI action "{action["name"]}" with context '
        f"{json.dumps(action['context'], ensure_ascii=False)}"
        for action in actions
    ]
    if text_input.strip():
        lines.append(text_input.strip())
    return "\n".join(lines)



def _stable_surface_id(kind: str, task_id: str, item_id: str = "") -> str:
    """Create a stable, renderer-safe ID without exposing arbitrary tool IDs."""
    key = f"{kind}:{task_id}:{item_id}"
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


def _thinking_component(
    text: str,
    status: str,
    append: str | None = None,
    details: str | None = None,
) -> dict[str, Any]:
    """Build a ThinkingBar component."""
    component: dict[str, Any] = {
        "id": "root",
        "component": "ThinkingBar",
        "text": text,
        "status": status,
    }
    # Only carry `append`/`details` when set: null values would fail the
    # properties' string schemas and make the renderer drop the component.
    if append:
        component["append"] = append
    if details:
        component["details"] = details
    return component


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
        self._mcp_clients: dict[str, MCPClient] = {}
        self._agents: dict[str, Agent] = {}


    SURFACES_KEY = "a2uiSurfaces"
    PENDING_CONFIRM_KEY = "pendingConfirmation"

    @classmethod
    def _task_metadata(cls, task: Task) -> dict[str, Any]:
        """Read the task metadata as a plain dict."""
        return json_format.MessageToDict(task.metadata)

    @classmethod
    def _update_task_metadata(
        cls,
        task: Task,
        updates: dict[str, Any],
    ) -> None:
        """Merge ``updates`` into the task metadata, keeping other keys."""
        merged = {**cls._task_metadata(task), **updates}
        task.metadata.CopyFrom(
            json_format.ParseDict(merged, struct_pb2.Struct()),
        )

    @classmethod
    def _task_surfaces(cls, task: Task) -> set[str]:
        """Read the set of emitted A2UI surface IDs from the task metadata."""
        return set(cls._task_metadata(task).get(cls.SURFACES_KEY, []))

    @classmethod
    def _save_task_surfaces(cls, task: Task, surfaces: set[str]) -> None:
        """Persist the surface IDs on the task so they survive the request."""
        cls._update_task_metadata(
            task,
            {cls.SURFACES_KEY: sorted(surfaces)},
        )

    @classmethod
    def _task_pending_confirm(
        cls,
        task: Task,
    ) -> RequireUserConfirmEvent | None:
        """Rebuild the parked confirmation from the task metadata."""
        data = cls._task_metadata(task).get(cls.PENDING_CONFIRM_KEY)
        if not data:
            return None
        return RequireUserConfirmEvent(
            reply_id=data["reply_id"],
            tool_calls=[
                ToolCallBlock(**tool_call) for tool_call in data["tool_calls"]
            ],
        )

    @classmethod
    def _set_task_pending_confirm(
        cls,
        task: Task,
        confirmation: RequireUserConfirmEvent | None,
    ) -> None:
        """Persist (or clear) the parked confirmation on the task metadata."""
        if confirmation is None:
            cls._update_task_metadata(task, {cls.PENDING_CONFIRM_KEY: None})
            return
        cls._update_task_metadata(
            task,
            {
                cls.PENDING_CONFIRM_KEY: {
                    "reply_id": confirmation.reply_id,
                    "tool_calls": [
                        tool_call.model_dump(mode="json")
                        for tool_call in confirmation.tool_calls
                    ],
                },
            },
        )




    async def _emit_component(
        self,
        task: Task,
        updater: TaskUpdater,
        surface_id: str,
        component: dict[str, Any],
    ) -> None:
        """Create or update a custom-catalog component surface."""
        surfaces = self._task_surfaces(task)
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
        self._save_task_surfaces(task, surfaces | {surface_id})

    async def _emit_confirmation(
        self,
        task: Task,
        updater: TaskUpdater,
        confirmation: RequireUserConfirmEvent,
        is_supports_a2ui: bool,
    ) -> None:
        """Present the confirmation and park the A2A task at input-required."""
        if is_supports_a2ui:
            for tool_call in confirmation.tool_calls:
                await self._emit_component(
                    task,
                    updater,
                    _stable_surface_id("confirm", task.id, tool_call.id),
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
        task: Task,
        updater: TaskUpdater,
        confirmresultevent: UserConfirmResultEvent | None,
    ) -> None:
        """Remove confirmation cards after their result is accepted."""
        surfaces = self._task_surfaces(task)
        for result in confirmresultevent.confirm_results if confirmresultevent is not None else []:
            surface_id = _stable_surface_id(
                "confirm",
                task.id,
                result.tool_call.id,
            )
            if surface_id not in surfaces:
                continue
            await updater.add_artifact(
                [
                    _a2ui_to_part({
                        "version": A2UI_VERSION,
                        "deleteSurface": {"surfaceId": surface_id},
                    }),
                ],
                extensions=[A2UI_EXTENSION_URI],
            )
            surfaces.remove(surface_id)
        self._save_task_surfaces(task, surfaces)


    async def _build_agent(self,) -> Agent:
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
                      "You are a helpful assistant named Friday.",
                    f"The local working directory is {workdir}.",
                    (
                        "Use Bash, Read, Write, Edit, and Glob for local "
                        "development tasks. Use Playwright only for websites "
                        "and applications the user is authorized to test. "
                        "Do not submit forms or perform consequential actions "
                        "without the user's explicit approval."
                    ),
                    _processor.prompt_snippet,
                ]),
                model=VolcengineChatModel(
                    credential=VolcengineCredential(
                        api_key=SecretStr(self._api_key),
                        base_url= "https://ark.cn-beijing.volces.com/api/coding/v3",
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
        # The agent is what remembers the parked reply awaiting confirmation,
        # so it must be reused across requests of the same context.
        agent = self._agents.get(context_id)
        if agent is None:
            agent = await self._build_agent()
            self._agents[context_id] = agent
            logger.info("Context %s: built new agent", context_id)
        else:
            logger.info("Context %s: reusing parked agent", context_id)


        awaiting_confirmation = False

        task = context.current_task
        if task is None:
            # The task manager saves this exact object, so metadata
            # mutations (A2UI surface tracking) survive across requests.
            task = Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
            )
            await event_queue.enqueue_event(task)

        # The Task must be enqueued before any status update, otherwise the
        # handler rejects events with "Agent should enqueue Task before
        # TaskStatusUpdateEvent".
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.start_work()

        user_input = context.get_user_input()
        is_supports_a2ui = _renderer_is_supports_a2ui(context)
        actions = _extract_a2ui_actions(context)
        if actions:
                # The renderer clicked a surface the model generated, so the
                # action is a normal user turn, not a tool decision.
            logger.info(
                "Received A2UI action(s): %s",
                 [action["name"] for action in actions],
                )
            action_result = _action_to_query(
                actions,
                user_input,
                self._task_pending_confirm(task),
            )
            if isinstance(action_result, UserConfirmResultEvent):
                logger.info("Resume pending tool confirmation: %s", [
                    r.tool_call.id for r in action_result.confirm_results
                ])
                self._set_task_pending_confirm(task, None)
            user_input = action_result


        if isinstance(user_input, UserConfirmResultEvent):
            agent_input = user_input
        else:
            agent_input = UserMsg(name="user", content=user_input)
        streaming = bool(parser is not None and parser.supports_streaming)
  

        buffered = ""
        a2ui_stream = (
            # "continue" keeps the stream alive after a rejected block: a bad
            # DSL block yields an ErrorPart instead of closing the stream and
            # crashing the task on the next text delta.
            parser.open_stream(after_rejected_block="continue")
            if parser is not None
            else None
        )
  
        tool_progress: dict[str, dict[str, Any]] = {}
        # Reasoning accumulated server-side, keyed by thinking block id, so
        # the terminal update can carry a full `details` snapshot.
        thinking_text: dict[str, str] = {}   
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

        async def handle_a2ui(text: str) -> None:
            """Stream model text, splitting out the A2UI blocks it carries."""
            nonlocal buffered
            if not streaming:
                buffered += text
                return
            elif a2ui_stream is not None:
                try:
                    for part in a2ui_stream.feed(text):
                        if isinstance(part,A2uiPart):
                            for message in to_message_dicts(part.a2ui):
                                await emit_part(_a2ui_to_part(message))
                                print(f"Emitting A2UI message: {message}")
                                print(f"Emitting A2UI message as dict: {_a2ui_to_part(message)}")
                            # print(f"Emitting A2UI part: {message}")
                            # print(f"Emitting A2UI part as dict: {to_message_dicts(part.a2ui)}")
                            # print(f"Emitting A2UI part as A2A Part: {_a2ui_to_part(to_message_dicts(part.a2ui))}")
                            # await emit_part(_a2ui_to_part(to_message_dicts(part.a2ui)))

                except A2uiError as e:
                    logger.warning("A2UI parse failed, sending raw text: %s", e)
                    buffered += text
                except ValueError as e:
                    # The stream is closed (an earlier block was rejected and
                    # after_rejected_block did not apply); keep the raw text.
                    logger.warning("A2UI stream no longer accepts text: %s", e)


        async def emit_tool(tool_call_id: str) -> None:
            state = tool_progress[tool_call_id]
            await self._emit_component(
                task,
                updater,
                _stable_surface_id("tool", task.id, tool_call_id),
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
                await emit_part(Part(text=event.delta))
                await handle_a2ui(event.delta)
            elif isinstance(event, ThinkingBlockStartEvent):
                if is_supports_a2ui:
                    await self._emit_component(
                        task,
                        updater,
                        _stable_surface_id('thinking', task.id,event.block_id),
                        _thinking_component("Thinking…", "thinking"),
                    )
                print(f"Thinking block started: {event.block_id}")

            elif isinstance(event, ThinkingBlockDeltaEvent):
                if is_supports_a2ui:
                    thinking_text[event.block_id] = (
                        thinking_text.get(event.block_id, "") + event.delta
                    )
                    # Delta only: the client accumulates locally, so the
                    # payload stays O(chunk) regardless of reasoning length.
                    await handle_a2ui(event.delta)
                    await self._emit_component(
                        task,
                        updater,
                        _stable_surface_id('thinking', task.id,event.block_id),
                        _thinking_component("Thinking…", "thinking", event.delta),
                    )


            elif isinstance(event, ThinkingBlockEndEvent):
                if is_supports_a2ui:
                    # Terminal full snapshot: an explicit `details` restores
                    # the reasoning for clients that joined or replayed past
                    # any of the deltas above.
                    await self._emit_component(
                        task,
                        updater,
                        _stable_surface_id('thinking', task.id,event.block_id),
                        _thinking_component(
                            "Thinking Done",
                            "done",
                            details=thinking_text.pop(event.block_id, None),
                        ),
                    )
            elif isinstance(event, ToolCallStartEvent):
                tool_progress[event.tool_call_id] = {
                    "name": event.tool_call_name,
                    "status": "pending",
                    "arguments": "",
                    "result": "",
                }
                if is_supports_a2ui:
                    await emit_tool(event.tool_call_id)
            elif isinstance(event, ToolCallDeltaEvent):
                tool_progress[event.tool_call_id]["arguments"] += event.delta
            elif isinstance(event, ToolCallEndEvent):
                tool_progress[event.tool_call_id]["status"] = "running"
                if is_supports_a2ui:
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
                if is_supports_a2ui:
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
                    if is_supports_a2ui:
                        await emit_tool(event.tool_call_id)
            elif isinstance(event, RequireUserConfirmEvent):
                awaiting_confirmation = True
                self._set_task_pending_confirm(task, event)
                for tool_call in event.tool_calls:
                    tool_progress[tool_call.id] = {
                        "name": tool_call.name,
                        "status": "pending",
                        "arguments": tool_call.input,
                        "result": "",
                    }
                    if is_supports_a2ui:
                        await emit_tool(tool_call.id)
                await self._emit_confirmation(
                    task,
                    updater,
                    event,
                    is_supports_a2ui,
                )



        # if a2ui_stream is not None and not awaiting_confirmation:
        #     # The stream holds back text that might start a tag; close it to
        #     # get the final parts of the reply.
        #     try:
        #         for stream_part in a2ui_stream.close():
        #             for part in _stream_part_to_a2a_parts(stream_part):
        #                 await emit_part(part)
        #     except A2uiError as e:
        #         logger.warning("A2UI stream close failed: %s", e)

        # if parser is not None and not streaming and buffered:
        #     try:
        #         parts = _to_a2a_parts(_processor.parse_response(buffered))
        #     except A2uiError as e:
        #         # Fall back to plain text when the model emits invalid DSL
        #         # instead of failing the whole task.
        #         logger.warning("A2UI parse failed, sending raw text: %s", e)
        #         parts = [Part(text=buffered)]
        #     for part in parts:
        #         await emit_part(part)

        if not awaiting_confirmation:
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
            context.task_id,
            context.context_id,
        ).cancel()

    async def aclose(self) -> None:
        """Close all per-conversation MCP connections."""
        clients = list(self._mcp_clients.values())
        self._mcp_clients.clear()
        for client in clients:
            await client.close()




def create_fastapi_app(
    base_url: str,
    model: str,
    api_key: str,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Build the A2A 1.0 application serving the agent using FastAPI.

    Browser renderers send a CORS preflight ``OPTIONS`` before posting a
    JSON-RPC request, so the app answers preflights for the origins it is
    configured with.
    """
    card = _create_agent_card(base_url)
    executor = AgentScopeExecutor(card=card, model=model, api_key=api_key)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
        try:
            yield
        finally:
            await executor.aclose()

    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    app = FastAPI()
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        # jsonrpc_routes_v0_3=create_jsonrpc_routes(handler, rpc_url='/v0.3',enable_v0_3_compat=True,),
        # rest_routes_v0_3=create_rest_routes(handler, path_prefix="/rest/v0.3", enable_v0_3_compat=True),
        jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url='/',enable_v0_3_compat=True),
        rest_routes=create_rest_routes(handler, path_prefix="/rest",enable_v0_3_compat=True),
   )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-A2A-Extensions"],
        max_age=600,
    )
    return app

    


async def main() -> None:
    """The main entry point of the server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ark-code-latest")
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

    api_key = os.environ.get("ARK_API_KEY")
    if not api_key: 
        raise RuntimeError(
            "Set the DEEPSEEK_API_KEY environment variable before running "
            "this demo.",
        )

    fast_app = create_fastapi_app(
        base_url=f"http://{args.host}:{args.port}",
        model=args.model,
        api_key=api_key,
        cors_origins=args.cors_origins,
    )


    await uvicorn.Server(
        uvicorn.Config(fast_app, host=args.host, port=args.port),
    ).serve()


if __name__ == "__main__":
    asyncio.run(main())
