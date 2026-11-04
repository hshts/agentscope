# -*- coding: utf-8 -*-
"""The A2A protocol middleware class."""
from typing import Any

from starlette.types import ASGIApp

from ._base import ProtocolMiddlewareBase
from ....event import (
    AgentEvent,
    DataBlockDeltaEvent,
    DataBlockEndEvent,
    DataBlockStartEvent,
    ExceedMaxItersEvent,
    ExternalExecutionResultEvent,
    ModelCallEndEvent,
    ModelCallStartEvent,
    ReplyEndEvent,
    ReplyFinishedReason,
    ReplyStartEvent,
    RequireExternalExecutionEvent,
    RequireUserConfirmEvent,
    TextBlockDeltaEvent,
    TextBlockEndEvent,
    TextBlockStartEvent,
    ThinkingBlockDeltaEvent,
    ThinkingBlockEndEvent,
    ThinkingBlockStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultDataDeltaEvent,
    ToolResultEndEvent,
    ToolResultStartEvent,
    ToolResultTextDeltaEvent,
    UserConfirmResultEvent,
)


class A2AProtocolMiddleware(ProtocolMiddlewareBase):
    """The middleware that converts the AgentScope events into A2A protocol
    streaming events."""

    def __init__(self, app: ASGIApp) -> None:
        """Initialize the A2A protocol middleware.

        Args:
            app: The ASGI application to wrap.
        """
        super().__init__(app)
        # Per-instance state; safe under typical single-stream usage
        # but not across concurrent requests.  Use contextvars if
        # concurrency is needed.
        self._context_id: str = "context"
        self._tool_result_buffers: dict[tuple[str, str], list[str]] = {}

    def _convert_to_protocol(self, event: AgentEvent) -> dict:
        """Convert the AgentScope events into A2A protocol."""
        from google.protobuf import json_format

        return json_format.MessageToDict(self._to_a2a_event(event))

    # pylint: disable=too-many-return-statements,too-many-branches
    def _to_a2a_event(self, event: AgentEvent) -> Any:  # noqa: C901
        """Convert an AgentScope event to an A2A streaming event."""

        from a2a.types import (
            Artifact as A2AArtifact,
            Part as A2APart,
            TaskState as A2ATaskState,
            TaskStatus as A2ATaskStatus,
            TaskArtifactUpdateEvent as A2ATaskArtifactUpdateEvent,
            TaskStatusUpdateEvent as A2ATaskStatusUpdateEvent,
        )

        if isinstance(event, ReplyStartEvent):
            self._context_id = event.session_id
            return A2ATaskStatusUpdateEvent(
                task_id=event.reply_id,
                context_id=self._context_id,
                status=A2ATaskStatus(
                    state=A2ATaskState.TASK_STATE_WORKING,
                ),
            )

        if isinstance(event, ReplyEndEvent):
            if event.finished_reason in (
                ReplyFinishedReason.EXCEED_MAX_ITERS,
                ReplyFinishedReason.ERROR,
            ):
                final_state = A2ATaskState.TASK_STATE_FAILED
            elif event.finished_reason == ReplyFinishedReason.INTERRUPTED:
                final_state = A2ATaskState.TASK_STATE_CANCELED
            else:
                final_state = A2ATaskState.TASK_STATE_COMPLETED
            return A2ATaskStatusUpdateEvent(
                task_id=event.reply_id,
                context_id=event.session_id,
                status=A2ATaskStatus(state=final_state),
            )

        if isinstance(event, ExceedMaxItersEvent):
            # Deprecated event, still emitted for backward compatibility.
            return self._wrap_in_data("exceed_max_iters", event)

        if isinstance(event, (ModelCallStartEvent, ModelCallEndEvent)):
            return self._wrap_in_data(event.type.lower(), event)

        # A2A streams agent output as artifacts, so the text blocks are
        # converted into ``TaskArtifactUpdateEvent`` chunks.
        if isinstance(event, TextBlockStartEvent):
            return A2ATaskArtifactUpdateEvent(
                task_id=event.reply_id,
                context_id=self._context_id,
                artifact=A2AArtifact(artifact_id=event.block_id),
            )

        if isinstance(event, TextBlockDeltaEvent):
            return A2ATaskArtifactUpdateEvent(
                task_id=event.reply_id,
                context_id=self._context_id,
                artifact=A2AArtifact(
                    artifact_id=event.block_id,
                    parts=[A2APart(text=event.delta)],
                ),
                append=True,
            )

        if isinstance(event, TextBlockEndEvent):
            return A2ATaskArtifactUpdateEvent(
                task_id=event.reply_id,
                context_id=self._context_id,
                artifact=A2AArtifact(artifact_id=event.block_id),
                last_chunk=True,
            )

        # Reasoning blocks have no direct A2A counterpart; carry them
        # in a data artifact.
        if isinstance(
            event,
            (
                ThinkingBlockStartEvent,
                ThinkingBlockDeltaEvent,
                ThinkingBlockEndEvent,
            ),
        ):
            return self._wrap_in_data(event.type.lower(), event)

        if isinstance(
            event,
            (
                ToolCallStartEvent,
                ToolCallDeltaEvent,
                ToolCallEndEvent,
                ToolResultStartEvent,
            ),
        ):
            return self._wrap_in_data(event.type.lower(), event)

        if isinstance(event, ToolResultTextDeltaEvent):
            # Buffer text deltas so the final artifact carries the entire
            # tool result content.
            self._tool_result_buffers.setdefault(
                (event.reply_id, event.tool_call_id),
                [],
            ).append(event.delta)
            return self._wrap_in_data(event.type.lower(), event)

        if isinstance(event, ToolResultDataDeltaEvent):
            part = A2APart()
            self._fill_struct(
                part.data,
                event.model_dump(mode="json", exclude_none=True),
            )
            return A2ATaskArtifactUpdateEvent(
                task_id=event.reply_id,
                context_id=self._context_id,
                artifact=A2AArtifact(
                    # ``reply_id`` is shared by every tool result of one
                    # reply, so qualify it with ``tool_call_id`` to keep
                    # each result its own artifact.
                    artifact_id=f"{event.reply_id}:{event.tool_call_id}:data",
                    parts=[part],
                ),
                append=True,
            )

        if isinstance(event, ToolResultEndEvent):
            content = "".join(
                self._tool_result_buffers.pop(
                    (event.reply_id, event.tool_call_id),
                    [],
                ),
            )
            return A2ATaskArtifactUpdateEvent(
                task_id=event.reply_id,
                context_id=self._context_id,
                artifact=A2AArtifact(
                    artifact_id=(
                        f"{event.reply_id}:{event.tool_call_id}:result"
                    ),
                    parts=[A2APart(text=content or str(event.state))],
                ),
                last_chunk=True,
            )

        # Permission requests carry an input-required marker so the
        # caller knows the task is waiting for external input.
        if isinstance(
            event,
            (
                RequireUserConfirmEvent,
                RequireExternalExecutionEvent,
                UserConfirmResultEvent,
                ExternalExecutionResultEvent,
            ),
        ):
            return self._wrap_in_data(
                event.type.lower(),
                event,
                state="input_required",
            )

        # Events without a direct A2A counterpart (data blocks, ...)
        # are carried in a data artifact, similar to AGUI's custom
        # events.
        return self._wrap_in_data(event.type.lower(), event)

    def _wrap_in_data(
        self,
        name: str,
        event: AgentEvent,
        state: Any = None,
    ) -> Any:
        """Wrap an AgentScope event into an A2A artifact data part.

        Args:
            name: The name describing the wrapped AgentScope event.
            event: The AgentScope event to wrap.
            state: The optional A2A ``TaskState`` to record in the data,
                e.g. ``TASK_STATE_INPUT_REQUIRED`` for permission events.

        Returns:
            The A2A ``TaskArtifactUpdateEvent`` carrying the AgentScope
            event in a data part.
        """
        from a2a.types import (
            Artifact as A2AArtifact,
            Part as A2APart,
            TaskArtifactUpdateEvent as A2ATaskArtifactUpdateEvent,
        )

        data: dict = {
            "event": name,
            "value": event.model_dump(mode="json", exclude_none=True),
        }
        if state is not None:
            data["state"] = state
        part = A2APart()
        self._fill_struct(part.data, data)
        artifact = A2AArtifact(
            artifact_id=getattr(
                event,
                "block_id",
                f"{getattr(event, 'reply_id', event.id)}:{name}",
            ),
        )
        artifact.parts.append(part)
        return A2ATaskArtifactUpdateEvent(
            task_id=getattr(event, "reply_id", event.id),
            context_id=self._context_id,
            artifact=artifact,
        )

    @staticmethod
    def _fill_struct(struct: Any, data: dict) -> None:
        """Fill a protobuf struct with the given JSON-compatible dict.

        Args:
            struct: The protobuf ``Struct`` to fill.
            data: The JSON-compatible dictionary to fill into the struct.
        """
        from google.protobuf import json_format

        json_format.ParseDict(data, struct)
