# -*- coding: utf-8 -*-
"""Unit tests for the A2A server's A2UI event and confirmation mapping."""
from __future__ import annotations

from typing import Any, cast
from unittest import IsolatedAsyncioTestCase

from google.protobuf import json_format, struct_pb2
from a2a.types import (
    Message,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
)
from a2a.server.agent_execution import RequestContext

from agentscope.agent import Agent
from agentscope.event import (
    RequireUserConfirmEvent,
    TextBlockDeltaEvent,
    ThinkingBlockStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultEndEvent,
    ToolResultStartEvent,
    ToolResultTextDeltaEvent,
    UserConfirmResultEvent,
)
from agentscope.message import (
    Base64Source,
    DataBlock,
    TextBlock,
    ToolCallBlock,
    ToolCallState,
    ToolResultState,
    URLSource,
)
from examples.a2a.server import (
    A2UI_EXTENSION_URI,
    A2UI_MIME_TYPE,
    A2UI_VERSION,
    AGENT_CATALOG_ID,
    CONFIRM_ACTION,
    REJECT_ACTION,
    AgentScopeExecutor,
    _create_agent_card,
    _extract_a2ui_actions,
    _extract_user_content,
    _human_confirm_component,
    _part_to_a2ui,
)


def _a2ui_to_part(a2ui: dict[str, Any] | list[Any]) -> Part:
    """Encode an A2UI message (or message list) as an A2A DataPart."""
    return Part(
        data=json_format.ParseDict(cast(Any, a2ui), struct_pb2.Value()),
        metadata={"mimeType": A2UI_MIME_TYPE},
        media_type=A2UI_MIME_TYPE,
    )


def _renderer_metadata() -> struct_pb2.Struct:
    return json_format.ParseDict(
        {
            "a2uiRendererCapabilities": {
                # Renderer capabilities are keyed by the A2UI spec version.
                A2UI_VERSION.value: {
                    "supportedCatalogIds": [AGENT_CATALOG_ID],
                },
            },
        },
        struct_pb2.Struct(),
    )


def _a2ui_messages(part: Part) -> list[dict[str, Any]]:
    payload = cast(Any, json_format.MessageToDict(part.data))
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise AssertionError("Expected an A2UI message list.")
    return payload


def _emitted_a2ui_messages(events: list[Any]) -> list[dict[str, Any]]:
    """Flatten the A2UI messages carried by the executor's events.

    Component surfaces are emitted as a single message object, while
    model-generated A2UI arrives as a list of messages, so both shapes are
    normalized here.
    """
    messages: list[dict[str, Any]] = []
    for event in events:
        if isinstance(event, TaskArtifactUpdateEvent):
            parts = event.artifact.parts
        elif isinstance(event, TaskStatusUpdateEvent):
            if not event.status.HasField("message"):
                continue
            parts = event.status.message.parts
        else:
            continue
        for part in parts:
            payload = _part_to_a2ui(part)
            if isinstance(payload, dict):
                messages.append(payload)
            elif isinstance(payload, list):
                messages.extend(
                    item for item in payload if isinstance(item, dict)
                )
    return messages


class _FakeContext:
    def __init__(
        self,
        *,
        message: Message,
        current_task: Task | None = None,
        text_input: str = "",
    ) -> None:
        self.task_id = "task-1"
        self.context_id = "context-1"
        self.message = message
        self.current_task = current_task
        self._text_input = text_input

    def get_user_input(self) -> str:
        return self._text_input


class _FakeEventQueue:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def enqueue_event(self, event: Any) -> None:
        self.events.append(event)


class _FakeAgent:
    def __init__(self) -> None:
        self.inputs: list[Any] = []
        self.tool_call = ToolCallBlock(
            id="call-1",
            name="send_email",
            input='{"recipient":"user@example.com"}',
        )

    async def reply_stream(self, inputs: Any, yield_final_msg: bool = False):
        self.inputs.append(inputs)
        if isinstance(inputs, UserConfirmResultEvent):
            yield ToolResultStartEvent(
                reply_id="reply-1",
                tool_call_id="call-1",
                tool_call_name="send_email",
            )
            yield ToolResultTextDeltaEvent(
                reply_id="reply-1",
                tool_call_id="call-1",
                delta="Sent",
            )
            yield ToolResultEndEvent(
                reply_id="reply-1",
                tool_call_id="call-1",
                state=ToolResultState.SUCCESS,
            )
            yield TextBlockDeltaEvent(
                reply_id="reply-2",
                block_id="text-1",
                delta="Done",
            )
            return

        yield ThinkingBlockStartEvent(reply_id="reply-1", block_id="think-1")
        yield ToolCallStartEvent(
            reply_id="reply-1",
            tool_call_id="call-1",
            tool_call_name="send_email",
        )
        yield ToolCallDeltaEvent(
            reply_id="reply-1",
            tool_call_id="call-1",
            delta=self.tool_call.input,
        )
        yield ToolCallEndEvent(
            reply_id="reply-1",
            tool_call_id="call-1",
        )
        yield RequireUserConfirmEvent(
            reply_id="reply-1",
            tool_calls=[self.tool_call],
        )


class A2AServerTest(IsolatedAsyncioTestCase):
    def test_agent_card_advertises_a2ui_catalogs(self) -> None:
        card = _create_agent_card("http://localhost:9999")

        self.assertEqual(len(card.capabilities.extensions), 1)
        extension = card.capabilities.extensions[0]
        self.assertEqual(extension.uri, A2UI_EXTENSION_URI)
        params = json_format.MessageToDict(extension.params)
        self.assertIn(AGENT_CATALOG_ID, params["supportedCatalogIds"])
        self.assertIn("application/a2ui+json", card.default_output_modes)
        self.assertIn("application/a2ui+json", card.default_input_modes)

    def test_a2ui_to_part_encodes_dict_and_list_payloads(self) -> None:
        message = {
            "version": "v1.0",
            "createSurface": {
                "surfaceId": "s-1",
                "catalogId": AGENT_CATALOG_ID,
                "components": [{"component": "Text", "text": "hi"}],
            },
        }

        part = _a2ui_to_part(message)
        self.assertEqual(part.WhichOneof("content"), "data")
        self.assertEqual(part.media_type, A2UI_MIME_TYPE)
        self.assertEqual(
            json_format.MessageToDict(part.metadata),
            {"mimeType": A2UI_MIME_TYPE},
        )
        self.assertEqual(json_format.MessageToDict(part.data), message)
        self.assertEqual(_part_to_a2ui(part), message)

        batch = [
            message,
            {"version": "v1.0", "deleteSurface": {"surfaceId": "s-1"}},
        ]
        batch_part = _a2ui_to_part(batch)
        self.assertEqual(json_format.MessageToDict(batch_part.data), batch)
        self.assertEqual(_part_to_a2ui(batch_part), batch)

    def test_extract_user_content_converts_all_part_kinds(self) -> None:
        message = Message(
            role=Role.ROLE_USER,
            parts=[
                Part(text="look at these"),
                Part(raw=b"abc", media_type="image/png", filename="a.png"),
                Part(
                    url="https://example.com/report.pdf",
                    media_type="application/pdf",
                    filename="report.pdf",
                ),
                Part(
                    data=json_format.ParseDict(
                        {"count": 2}, struct_pb2.Value()
                    ),
                    media_type="application/json",
                ),
            ],
        )

        content = _extract_user_content(message)

        self.assertEqual(content.actions, [])
        self.assertEqual(
            [type(block) for block in content.content],
            [TextBlock, DataBlock, DataBlock, TextBlock],
        )
        self.assertEqual(content.content[0].text, "look at these")
        self.assertIsInstance(content.content[1].source, Base64Source)
        self.assertEqual(content.content[1].source.data, "YWJj")
        self.assertEqual(content.content[1].source.media_type, "image/png")
        self.assertEqual(content.content[1].name, "a.png")
        self.assertIsInstance(content.content[2].source, URLSource)
        self.assertEqual(
            str(content.content[2].source.url),
            "https://example.com/report.pdf",
        )
        self.assertEqual(content.content[3].text, '{"count": 2.0}')

    def test_extract_user_content_keeps_a2ui_actions_out_of_content(
        self,
    ) -> None:
        action_part = _a2ui_to_part(
            [
                {
                    "version": "v1.0",
                    "action": {"name": CONFIRM_ACTION, "context": {}},
                },
            ],
        )
        message = Message(
            role=Role.ROLE_USER,
            parts=[
                Part(text="approve it"),
                action_part,
            ],
        )

        content = _extract_user_content(message)

        self.assertEqual(
            [action["name"] for action in content.actions],
            [
                CONFIRM_ACTION,
            ],
        )
        self.assertEqual(len(content.content), 1)
        self.assertEqual(content.content[0].text, "approve it")

    def test_confirmation_component_and_action_data_part(self) -> None:
        tool_call = ToolCallBlock(
            id="call-1",
            name="send_email",
            input='{"recipient":"user@example.com"}',
        )
        component = _human_confirm_component(tool_call)
        self.assertEqual(component["component"], "HumanConfirm")
        self.assertEqual(
            component["confirmAction"]["event"]["name"],
            CONFIRM_ACTION,
        )
        self.assertEqual(
            component["cancelAction"]["event"]["name"],
            REJECT_ACTION,
        )

        part = _a2ui_to_part(
            [
                {
                    "version": "v1.0",
                    "action": {
                        "name": CONFIRM_ACTION,
                        "context": {"toolCallId": "call-1"},
                    },
                },
            ],
        )
        self.assertEqual(
            json_format.MessageToDict(part.metadata)["mimeType"],
            A2UI_MIME_TYPE,
        )
        self.assertEqual(
            _extract_a2ui_actions(
                Message(role=Role.ROLE_USER, parts=[part]),
            )[0]["context"]["toolCallId"],
            "call-1",
        )


    async def test_confirmation_pauses_and_resumes_a2a_task(self) -> None:
        executor = AgentScopeExecutor(model="test", api_key="unused")
        fake_agent = _FakeAgent()
        executor._agents["context-1"] = cast(Agent, fake_agent)
        queue = _FakeEventQueue()
        first_context = _FakeContext(
            message=Message(
                role=Role.ROLE_USER,
                metadata=_renderer_metadata(),
            ),
            text_input="send an email",
        )

        await executor.execute(cast(RequestContext, first_context), cast(Any, queue))

        first_statuses = [
            event
            for event in queue.events
            if isinstance(event, TaskStatusUpdateEvent)
        ]
        self.assertEqual(
            first_statuses[-1].status.state,
            TaskState.TASK_STATE_INPUT_REQUIRED,
        )
        confirm_surface_ids = [
            message["createSurface"]["surfaceId"]
            for message in _emitted_a2ui_messages(queue.events)
            if any(
                component["component"] == "HumanConfirm"
                for component in message.get("createSurface", {}).get(
                    "components",
                    [],
                )
            )
        ]
        self.assertEqual(len(confirm_surface_ids), 1)

        action_part = _a2ui_to_part(
            [
                {
                    "version": "v1.0",
                    "action": {
                        "name": CONFIRM_ACTION,
                        "context": {"toolCallId": "call-1"},
                    },
                },
            ],
        )
        # The parked confirmation is stored on the task metadata, which is the
        # same task object the task store hands back on the next request.
        parked_task = next(
            event for event in queue.events if isinstance(event, Task)
        )
        resumed_context = _FakeContext(
            message=Message(
                role=Role.ROLE_USER,
                parts=[action_part],
                metadata=_renderer_metadata(),
            ),
            current_task=parked_task,
        )

        await executor.execute(
            cast(RequestContext, resumed_context),
            cast(Any, queue),
        )

        self.assertIsInstance(fake_agent.inputs[-1], UserConfirmResultEvent)
        self.assertTrue(fake_agent.inputs[-1].confirm_results[0].confirmed)
        self.assertEqual(
            [
                message["deleteSurface"]
                for message in _emitted_a2ui_messages(queue.events)
                if "deleteSurface" in message
            ],
            [{"surfaceId": confirm_surface_ids[0]}],
        )
        final_statuses = [
            event
            for event in queue.events
            if isinstance(event, TaskStatusUpdateEvent)
        ]
        self.assertEqual(
            final_statuses[-1].status.state,
            TaskState.TASK_STATE_COMPLETED,
        )
        rendered_components = [
            component
            for message in _emitted_a2ui_messages(queue.events)
            for update in (
                message.get("createSurface", {}).get("components", [])
                + message.get("updateComponents", {}).get("components", [])
            )
            for component in [update]
        ]
        self.assertTrue(
            any(
                component.get("component") == "ToolCall"
                and component.get("status") == "success"
                and component.get("result") == "Sent"
                for component in rendered_components
            ),
        )

    async def test_execute_passes_attachments_to_the_agent(self) -> None:
        executor = AgentScopeExecutor(model="test", api_key="unused")
        fake_agent = _FakeAgent()
        executor._agents["context-1"] = cast(Agent, fake_agent)
        queue = _FakeEventQueue()
        context = _FakeContext(
            message=Message(
                role=Role.ROLE_USER,
                parts=[
                    Part(text="what is this?"),
                    Part(raw=b"abc", media_type="image/png", filename="a.png"),
                ],
                metadata=_renderer_metadata(),
            ),
            text_input="what is this?",
        )

        await executor.execute(cast(RequestContext, context), cast(Any, queue))

        agent_input = fake_agent.inputs[-1]
        self.assertEqual(
            [type(block) for block in agent_input.content],
            [TextBlock, DataBlock],
        )
        self.assertEqual(agent_input.content[0].text, "what is this?")
        self.assertEqual(agent_input.content[1].source.data, "YWJj")


if __name__ == "__main__":
    import unittest

    unittest.main()
