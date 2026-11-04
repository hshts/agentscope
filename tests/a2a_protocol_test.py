# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Test cases for A2A protocol middleware."""

import json
from typing import AsyncGenerator
from unittest.async_case import IsolatedAsyncioTestCase
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from agentscope.app.middleware import A2AProtocolMiddleware
from agentscope.event import (
    DataBlockStartEvent,
    ExceedMaxItersEvent,
    ModelCallEndEvent,
    ModelCallStartEvent,
    ReplyEndEvent,
    ReplyStartEvent,
    RequireUserConfirmEvent,
    TextBlockDeltaEvent,
    TextBlockEndEvent,
    TextBlockStartEvent,
    ThinkingBlockDeltaEvent,
    ToolCallStartEvent,
    ToolResultDataDeltaEvent,
    ToolResultEndEvent,
    ToolResultStartEvent,
    ToolResultTextDeltaEvent,
    ReplyFinishedReason,
)
from agentscope.message import ToolCallBlock, ToolResultState
from agentscope.model import FinishedReason
from tests.utils import AnyString


async def _collect_stream(
    mw: A2AProtocolMiddleware,
    chunks: list[str],
) -> str:
    """Collect converted stream chunks as text."""

    async def _stream() -> AsyncGenerator[str, None]:
        """Yield the provided chunks."""
        for chunk in chunks:
            yield chunk

    out: list[str] = []
    async for item in mw._convert_stream(_stream()):
        out.append(item.decode("utf-8"))
    return "".join(out)


class A2AProtocolStreamTest(IsolatedAsyncioTestCase):
    """Test stream-level conversion behavior."""

    async def asyncSetUp(self) -> None:
        """The async setup method."""
        self.mw = A2AProtocolMiddleware(app=MagicMock())

    async def test_raw_json_stream_is_converted(self) -> None:
        """Test raw AgentEvent JSON stream conversion."""
        event = ReplyStartEvent(
            session_id="sess_1",
            reply_id="reply_1",
            name="agent",
        )

        body = await _collect_stream(self.mw, [event.model_dump_json()])
        data = json.loads(body)

        self.assertEqual(
            data,
            {
                "taskId": "reply_1",
                "contextId": "sess_1",
                "status": {"state": "TASK_STATE_WORKING"},
            },
        )

    async def test_sse_data_frame_is_converted(self) -> None:
        """Test AgentEvent JSON inside an SSE data frame is converted."""
        event = ReplyStartEvent(
            session_id="sess_1",
            reply_id="reply_1",
            name="agent",
        )

        body = await _collect_stream(
            self.mw,
            [f"data: {event.model_dump_json()}\n\n"],
        )
        self.assertTrue(body.startswith("data: "))

        data = json.loads(body.removeprefix("data: ").strip())
        self.assertEqual(data["taskId"], "reply_1")
        self.assertEqual(data["contextId"], "sess_1")

    async def test_sse_heartbeat_is_passed_through(self) -> None:
        """Test SSE heartbeat frames are not modified."""
        self.assertEqual(
            await _collect_stream(self.mw, [":\n\n"]),
            ":\n\n",
        )

    async def test_fastapi_sse_response_is_converted(self) -> None:
        """Test middleware converts a real FastAPI SSE response."""
        app = FastAPI()
        app.add_middleware(A2AProtocolMiddleware)

        @app.get("/sessions/sess_1/stream")
        async def stream() -> StreamingResponse:
            event = ReplyStartEvent(
                session_id="sess_1",
                reply_id="reply_1",
                name="agent",
            )

            async def gen() -> AsyncGenerator[str, None]:
                yield f"data: {event.model_dump_json()}\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        client = TestClient(app)
        with client.stream("GET", "/sessions/sess_1/stream") as response:
            body = "".join(response.iter_text())

        self.assertTrue(body.startswith("data: "))
        data = json.loads(body.removeprefix("data: ").strip())
        self.assertEqual(data["taskId"], "reply_1")
        self.assertEqual(data["contextId"], "sess_1")
        self.assertEqual(data["status"]["state"], "TASK_STATE_WORKING")

    async def test_fastapi_json_response_is_not_converted(self) -> None:
        """Test non-SSE responses are outside protocol conversion scope."""
        app = FastAPI()
        app.add_middleware(A2AProtocolMiddleware)

        @app.get("/event")
        def event() -> JSONResponse:
            agent_event = ReplyStartEvent(
                session_id="sess_1",
                reply_id="reply_1",
                name="agent",
            )
            return JSONResponse(agent_event.model_dump(mode="json"))

        client = TestClient(app)
        response = client.get("/event")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["type"], "REPLY_START")
        self.assertNotIn("taskId", data)

    async def asyncTearDown(self) -> None:
        """The async teardown method."""


class A2AProtocolLifecycleTest(IsolatedAsyncioTestCase):
    """Test lifecycle event conversions."""

    async def asyncSetUp(self) -> None:
        """The async setup method."""
        self.mw = A2AProtocolMiddleware(app=MagicMock())

    async def test_reply_start_to_working_status(self) -> None:
        """Test ReplyStartEvent -> status update with working state."""
        event = ReplyStartEvent(
            session_id="sess_1",
            reply_id="reply_1",
            name="agent",
        )
        result = self.mw._convert_to_protocol(event)

        self.assertEqual(
            result,
            {
                "taskId": "reply_1",
                "contextId": "sess_1",
                "status": {"state": "TASK_STATE_WORKING"},
            },
        )

    async def test_reply_end_completed(self) -> None:
        """Test ReplyEndEvent with COMPLETED -> completed state."""
        event = ReplyEndEvent(
            session_id="sess_1",
            reply_id="reply_1",
            finished_reason=ReplyFinishedReason.COMPLETED,
        )
        result = self.mw._convert_to_protocol(event)

        self.assertEqual(
            result,
            {
                "taskId": "reply_1",
                "contextId": "sess_1",
                "status": {"state": "TASK_STATE_COMPLETED"},
            },
        )

    async def test_reply_end_exceed_max_iters_failed(self) -> None:
        """Test ReplyEndEvent with EXCEED_MAX_ITERS -> failed state."""
        event = ReplyEndEvent(
            session_id="sess_1",
            reply_id="reply_1",
            finished_reason=ReplyFinishedReason.EXCEED_MAX_ITERS,
        )
        result = self.mw._convert_to_protocol(event)

        self.assertEqual(
            result,
            {
                "taskId": "reply_1",
                "contextId": "sess_1",
                "status": {"state": "TASK_STATE_FAILED"},
            },
        )

    async def test_reply_end_interrupted_canceled(self) -> None:
        """Test ReplyEndEvent with INTERRUPTED -> canceled state."""
        event = ReplyEndEvent(
            session_id="sess_1",
            reply_id="reply_1",
            finished_reason=ReplyFinishedReason.INTERRUPTED,
        )
        result = self.mw._convert_to_protocol(event)

        self.assertEqual(
            result,
            {
                "taskId": "reply_1",
                "contextId": "sess_1",
                "status": {"state": "TASK_STATE_CANCELED"},
            },
        )

    async def test_deprecated_exceed_max_iters_wrapped_in_data(
        self,
    ) -> None:
        """Test the deprecated ExceedMaxItersEvent is carried in data."""
        event = ExceedMaxItersEvent(
            reply_id="reply_1",
            name="my_agent",
        )
        result = self.mw._convert_to_protocol(event)

        self.assertEqual(result["taskId"], "reply_1")
        parts = result["artifact"]["parts"]
        self.assertEqual(parts[0]["data"]["event"], "exceed_max_iters")

    async def asyncTearDown(self) -> None:
        """The async teardown method."""


class A2AProtocolArtifactTest(IsolatedAsyncioTestCase):
    """Test text block -> artifact event conversions."""

    async def asyncSetUp(self) -> None:
        """The async setup method."""
        self.mw = A2AProtocolMiddleware(app=MagicMock())
        self.mw._convert_to_protocol(
            ReplyStartEvent(
                session_id="sess_1",
                reply_id="reply_1",
                name="agent",
            ),
        )

    async def test_text_block_start(self) -> None:
        """Test TextBlockStartEvent -> artifact update start."""
        event = TextBlockStartEvent(reply_id="reply_1", block_id="block_1")
        result = self.mw._convert_to_protocol(event)

        self.assertEqual(
            result,
            {
                "taskId": "reply_1",
                "contextId": "sess_1",
                "artifact": {"artifactId": "block_1"},
            },
        )

    async def test_text_block_delta(self) -> None:
        """Test TextBlockDeltaEvent -> streamed artifact text chunk."""
        event = TextBlockDeltaEvent(
            reply_id="reply_1",
            block_id="block_1",
            delta="Hello, ",
        )
        result = self.mw._convert_to_protocol(event)

        self.assertEqual(
            result,
            {
                "taskId": "reply_1",
                "contextId": "sess_1",
                "append": True,
                "artifact": {
                    "artifactId": "block_1",
                    "parts": [{"text": "Hello, "}],
                },
            },
        )

    async def test_text_block_end(self) -> None:
        """Test TextBlockEndEvent -> last artifact chunk."""
        event = TextBlockEndEvent(reply_id="reply_1", block_id="block_1")
        result = self.mw._convert_to_protocol(event)

        self.assertEqual(
            result,
            {
                "taskId": "reply_1",
                "contextId": "sess_1",
                "lastChunk": True,
                "artifact": {"artifactId": "block_1"},
            },
        )

    async def asyncTearDown(self) -> None:
        """The async teardown method."""


class A2AProtocolToolTest(IsolatedAsyncioTestCase):
    """Test tool call and tool result event conversions."""

    async def asyncSetUp(self) -> None:
        """The async setup method."""
        self.mw = A2AProtocolMiddleware(app=MagicMock())
        self.mw._convert_to_protocol(
            ReplyStartEvent(
                session_id="sess_1",
                reply_id="reply_1",
                name="agent",
            ),
        )

    async def test_tool_call_start_wrapped_in_data(self) -> None:
        """Test ToolCallStartEvent is carried in a data artifact."""
        event = ToolCallStartEvent(
            reply_id="reply_1",
            tool_call_id="tc_1",
            tool_call_name="search",
        )
        result = self.mw._convert_to_protocol(event)

        self.assertEqual(
            result,
            {
                "taskId": "reply_1",
                "contextId": "sess_1",
                "artifact": {
                    "artifactId": "reply_1:tool_call_start",
                    "parts": [
                        {
                            "data": {
                                "event": "tool_call_start",
                                "value": {
                                    "type": "TOOL_CALL_START",
                                    "id": AnyString(),
                                    "created_at": AnyString(),
                                    "metadata": {},
                                    "reply_id": "reply_1",
                                    "tool_call_id": "tc_1",
                                    "tool_call_name": "search",
                                },
                            },
                        },
                    ],
                },
            },
        )

    async def test_tool_result_end_with_buffered_content(self) -> None:
        """Test that ToolResultEndEvent carries accumulated text content."""
        for delta in ("partial ", "result"):
            self.mw._convert_to_protocol(
                ToolResultTextDeltaEvent(
                    reply_id="reply_1",
                    tool_call_id="tc_1",
                    delta=delta,
                ),
            )

        result = self.mw._convert_to_protocol(
            ToolResultEndEvent(
                reply_id="reply_1",
                tool_call_id="tc_1",
                state=ToolResultState.SUCCESS,
            ),
        )

        self.assertEqual(
            result,
            {
                "taskId": "reply_1",
                "contextId": "sess_1",
                "lastChunk": True,
                "artifact": {
                    "artifactId": "reply_1:tc_1:result",
                    "parts": [{"text": "partial result"}],
                },
            },
        )
        self.assertDictEqual(self.mw._tool_result_buffers, {})

    async def test_tool_result_end_fallback_to_state(self) -> None:
        """Test that ToolResultEndEvent falls back to state when no buffer."""
        result = self.mw._convert_to_protocol(
            ToolResultEndEvent(
                reply_id="reply_1",
                tool_call_id="tc_1",
                state=ToolResultState.ERROR,
            ),
        )

        self.assertEqual(
            result,
            {
                "taskId": "reply_1",
                "contextId": "sess_1",
                "lastChunk": True,
                "artifact": {
                    "artifactId": "reply_1:tc_1:result",
                    "parts": [{"text": "error"}],
                },
            },
        )

    async def test_multiple_tool_results_use_unique_artifact_ids(self) -> None:
        """Test that two tool results of one reply get distinct ids."""
        results = []
        for tool_call_id, delta in (
            ("tc_1", "result A"),
            ("tc_2", "result B"),
        ):
            self.mw._convert_to_protocol(
                ToolResultTextDeltaEvent(
                    reply_id="reply_1",
                    tool_call_id=tool_call_id,
                    delta=delta,
                ),
            )
            results.append(
                self.mw._convert_to_protocol(
                    ToolResultEndEvent(
                        reply_id="reply_1",
                        tool_call_id=tool_call_id,
                        state=ToolResultState.SUCCESS,
                    ),
                ),
            )

        self.assertEqual(
            results,
            [
                {
                    "taskId": "reply_1",
                    "contextId": "sess_1",
                    "lastChunk": True,
                    "artifact": {
                        "artifactId": "reply_1:tc_1:result",
                        "parts": [{"text": "result A"}],
                    },
                },
                {
                    "taskId": "reply_1",
                    "contextId": "sess_1",
                    "lastChunk": True,
                    "artifact": {
                        "artifactId": "reply_1:tc_2:result",
                        "parts": [{"text": "result B"}],
                    },
                },
            ],
        )
        self.assertDictEqual(self.mw._tool_result_buffers, {})

    async def test_tool_result_data_delta_to_data_artifact(self) -> None:
        """Test ToolResultDataDeltaEvent -> streamed data artifact."""
        event = ToolResultDataDeltaEvent(
            reply_id="reply_1",
            tool_call_id="tc_1",
            media_type="image/png",
            data="base64chunk",
        )
        result = self.mw._convert_to_protocol(event)

        self.assertEqual(
            result,
            {
                "taskId": "reply_1",
                "contextId": "sess_1",
                "append": True,
                "artifact": {
                    "artifactId": "reply_1:tc_1:data",
                    "parts": [
                        {
                            "data": {
                                "type": "TOOL_RESULT_DATA_DELTA",
                                "id": AnyString(),
                                "created_at": AnyString(),
                                "metadata": {},
                                "reply_id": "reply_1",
                                "tool_call_id": "tc_1",
                                "block_id": AnyString(),
                                "media_type": "image/png",
                                "data": "base64chunk",
                            },
                        },
                    ],
                },
            },
        )

    async def asyncTearDown(self) -> None:
        """The async teardown method."""


class A2AProtocolWrappedEventTest(IsolatedAsyncioTestCase):
    """Test events without direct A2A counterparts are wrapped."""

    async def asyncSetUp(self) -> None:
        """The async setup method."""
        self.mw = A2AProtocolMiddleware(app=MagicMock())
        self.mw._convert_to_protocol(
            ReplyStartEvent(
                session_id="sess_1",
                reply_id="reply_1",
                name="agent",
            ),
        )

    async def test_thinking_block_delta_wrapped_in_data(self) -> None:
        """Test ThinkingBlockDeltaEvent is carried in a data artifact."""
        event = ThinkingBlockDeltaEvent(
            reply_id="reply_1",
            block_id="think_1",
            delta="Let me think...",
        )
        result = self.mw._convert_to_protocol(event)

        data = result["artifact"]["parts"][0]["data"]
        self.assertEqual(result["artifact"]["artifactId"], "think_1")
        self.assertEqual(data["event"], "thinking_block_delta")
        self.assertEqual(
            data["value"]["delta"],
            "Let me think...",
        )

    async def test_data_block_start_wrapped_in_data(self) -> None:
        """Test DataBlockStartEvent is carried in a data artifact."""
        event = DataBlockStartEvent(
            reply_id="reply_1",
            block_id="db_1",
            media_type="image/png",
        )
        result = self.mw._convert_to_protocol(event)

        data = result["artifact"]["parts"][0]["data"]
        self.assertEqual(result["artifact"]["artifactId"], "db_1")
        self.assertEqual(data["event"], "data_block_start")
        self.assertEqual(
            data["value"]["media_type"],
            "image/png",
        )

    async def test_require_user_confirm_wrapped_in_data(self) -> None:
        """Test RequireUserConfirmEvent is carried in a data artifact."""
        event = RequireUserConfirmEvent(
            reply_id="reply_1",
            tool_calls=[
                ToolCallBlock(
                    id="tc_1",
                    name="bash",
                    input='{"command": "ls"}',
                ),
            ],
        )
        result = self.mw._convert_to_protocol(event)

        data = result["artifact"]["parts"][0]["data"]
        self.assertEqual(result["taskId"], "reply_1")
        self.assertEqual(data["event"], "require_user_confirm")
        self.assertEqual(data["state"], "input_required")
        self.assertIn("tool_calls", data["value"])

    async def test_model_call_events_wrapped_in_data(self) -> None:
        """Test model call events are carried in data artifacts."""
        start = ModelCallStartEvent(
            reply_id="reply_1",
            model_name="gpt-4",
        )
        result_start = self.mw._convert_to_protocol(start)
        data_start = result_start["artifact"]["parts"][0]["data"]
        self.assertEqual(data_start["event"], "model_call_start")
        self.assertEqual(
            data_start["value"]["model_name"],
            "gpt-4",
        )

        end = ModelCallEndEvent(
            reply_id="reply_1",
            input_tokens=100,
            output_tokens=50,
            finished_reason=FinishedReason.COMPLETED,
        )
        result_end = self.mw._convert_to_protocol(end)
        data_end = result_end["artifact"]["parts"][0]["data"]
        self.assertEqual(data_end["event"], "model_call_end")

    async def asyncTearDown(self) -> None:
        """The async teardown method."""
