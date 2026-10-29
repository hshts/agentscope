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
    TaskStatus,
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
from agentscope.message import ToolCallBlock, ToolResultState
from examples.a2a.server import (
    A2UI_EXTENSION_URI,
    A2UI_MIME_TYPE,
    AGENT_CATALOG_ID,
    CONFIRM_ACTION,
    REJECT_ACTION,
    AgentScopeExecutor,
    _PendingConfirmation,
    _a2ui_part,
    _confirmation_result,
    _create_agent_card,
    _extract_a2ui_actions,
    _human_confirm_component,
)


def _renderer_metadata() -> struct_pb2.Struct:
    return json_format.ParseDict(
        {
            "a2uiRendererCapabilities": {
                "v1.0": {
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

    async def reply_stream(self, inputs: Any):
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

        part = _a2ui_part(
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

    def test_text_confirmation_maps_to_agentscope_result(self) -> None:
        tool_call = ToolCallBlock(
            id="call-1",
            name="send_email",
            input="{}",
        )
        pending = _PendingConfirmation(
            event=RequireUserConfirmEvent(
                reply_id="reply-1",
                tool_calls=[tool_call],
            ),
            supports_a2ui=False,
        )

        result = _confirmation_result(pending, None, "yes")

        self.assertTrue(result.confirmed)
        self.assertEqual(result.tool_call.id, "call-1")

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
        a2ui_parts = [
            part
            for event in queue.events
            if isinstance(event, TaskArtifactUpdateEvent)
            for part in event.artifact.parts
            if part.HasField("data")
        ]
        self.assertTrue(a2ui_parts)
        self.assertTrue(
            any(
                component["component"] == "HumanConfirm"
                for part in a2ui_parts
                for message in _a2ui_messages(part)
                for surface in [message.get("createSurface", {})]
                for component in surface.get("components", [])
            ),
        )

        action_part = _a2ui_part(
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
        resumed_context = _FakeContext(
            message=Message(
                role=Role.ROLE_USER,
                parts=[action_part],
                metadata=_renderer_metadata(),
            ),
            current_task=Task(
                id="task-1",
                context_id="context-1",
                status=TaskStatus(state=TaskState.TASK_STATE_INPUT_REQUIRED),
            ),
        )

        await executor.execute(
            cast(RequestContext, resumed_context),
            cast(Any, queue),
        )

        self.assertIsInstance(fake_agent.inputs[-1], UserConfirmResultEvent)
        self.assertTrue(fake_agent.inputs[-1].confirm_results[0].confirmed)
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
            for event in queue.events
            if isinstance(event, TaskArtifactUpdateEvent)
            for part in event.artifact.parts
            if part.HasField("data")
            for message in _a2ui_messages(part)
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


if __name__ == "__main__":
    import unittest

    unittest.main()
