# A2A Agent

This example puts AgentScope on both ends of the
[A2A 1.0](https://a2a-protocol.org/) protocol:

- `server.py` wraps an AgentScope `Agent` in an A2A `AgentExecutor` and serves
  it over JSON-RPC/SSE.
- `client.py` connects to it with `A2AAgent` and hands that agent to
  `launch_console`, so the remote agent is chatted with exactly like a local
  one.

`A2AAgent` deliberately exposes Agent-like methods without inheriting from
`Agent`. A local `Agent` owns a model, toolkit, state and reasoning loop;
`A2AAgent` delegates all of that to the remote server and owns only the remote
conversation (`context_id`) and the Task the next message continues
(`task_id`).

## Prerequisites

- Python 3.11 or newer
- AgentScope with the A2A extra:

  ```bash
  pip install "agentscope[a2a]"
  ```

  From an AgentScope source checkout, use `uv sync --extra a2a` instead.
- A `DEEPSAKE_API_KEY` for the server's configured model.

## Run it

In one terminal:

```bash
export DEEPSAKE_API_KEY=...
python server.py                 # http://127.0.0.1:9999
python server.py --port 8080 --model deepsake-flash
```

In another:

```bash
python client.py
python client.py --url http://127.0.0.1:8080 --verbosity debug
```

Type a message and the reply streams back through A2A artifact chunks. `exit`,
`quit` or Ctrl+D leaves the console.

## A2UI activity and human confirmation

The AgentCard advertises the A2UI v1.0 extension
(`https://a2ui.org/a2a-extension/a2ui/v1.0`) and the basic and AgentScope
catalogs. The server emits A2UI DataParts (`application/a2ui+json`) only when
the renderer advertises the AgentScope catalog in
`message.metadata.a2uiRendererCapabilities.v1.0.supportedCatalogIds`.
Otherwise, tool confirmations use a plain-text prompt.

Thinking events update a `ThinkingBar` with short status text; model-provided
thinking text is not sent to the renderer. Tool-call and tool-result events
update `ToolCall` surfaces with the call arguments, result, and status. When
AgentScope raises `RequireUserConfirmEvent`, the server pauses the A2A Task in
`input-required` and displays one `HumanConfirm` card at a time. The renderer
resumes the Task by sending an A2UI `action` DataPart whose event name is
`agentscope.tool.confirm` or `agentscope.tool.reject` and whose context contains
the `toolCallId`. A plain-text client can instead reply exactly `yes` or `no`.

The included `A2AAgent` console adapter sends text messages; it does not yet
provide an A2UI renderer or transmit A2UI action DataParts. Use an A2UI-capable
renderer to exercise the custom activity surfaces and button-based confirmation.

## How the two sides map onto each other

**Server** (`AgentScopeExecutor`). One AgentScope `Agent` per A2A
`context_id`, so concurrent conversations stay independent. Each reply
enqueues the `Task`, moves it to `WORKING`, publishes every
`TextBlockDeltaEvent` as an artifact chunk (the last one marked
`last_chunk=True`), and completes the Task.

**Client** (`A2AAgent`). Every A2A Part of a response becomes one content
block — text Parts become `TextBlock`, raw byte and URL Parts become
`DataBlock` — streamed as ordinary AgentScope block events. Each block end
event carries the A2A ids it came from under `metadata["a2a"]`, and the final
message carries the `context_id`.

## Using `A2AAgent` directly

Resolve the card once and use the agent as an async context manager:

```python
import httpx
from a2a.client import A2ACardResolver

from agentscope.agent import A2AAgent
from agentscope.message import UserMsg


async with httpx.AsyncClient() as httpx_client:
    card = await A2ACardResolver(
        httpx_client=httpx_client,
        base_url="http://127.0.0.1:9999",
    ).get_agent_card()

async with A2AAgent(card) as agent:
    reply = await agent.reply(
        UserMsg(name="user", content="Plan a weekend in Hangzhou."),
    )
    print(reply.get_text_content())

    # The second call reuses the remote context automatically.
    reply = await agent.reply(
        UserMsg(name="user", content="Make it suitable for children."),
    )
```

Use `reply_stream()` when the application needs the events instead. Messages
passed to `observe()` are sent ahead of the next reply's own input:

```python
await agent.observe(previous_messages)
reply = await agent.reply(current_message)
```

The adapter owns its A2A client and closes it on exit, so one instance serves
one conversation and cannot be reopened.

## Task lifecycle

A remote Task suspends server-side, but nothing suspends locally, so every
response stream ends the reply. The state it ended on decides how:

- `COMPLETED`, `INPUT_REQUIRED`, `AUTH_REQUIRED` → `COMPLETED`. The Task status
  message is emitted as ordinary content, so the remote agent's question is
  part of the reply — answer it with the next `reply()`.
- `CANCELED` → `INTERRUPTED`.
- `FAILED`, `REJECTED` → `ERROR`.

`state.task_id` is kept only while the server is waiting for input
(`INPUT_REQUIRED` / `AUTH_REQUIRED`); the next message then continues that
Task. Any other outcome clears it and the next message starts a new Task
inside the same `context_id`. A Task the server has since forgotten degrades
into a new one; a Task still running is reported as an error, because sending
to it would run it a second time.

A2A carries credentials out of band, so an `AUTH_REQUIRED` Task cannot be
approved through this adapter — follow the instructions in the status message.

## Resuming a conversation

`A2AAgentState` holds everything worth persisting and is accepted by the
constructor:

```python
from agentscope.state import A2AAgentState

agent = A2AAgent(card, state=A2AAgentState(context_id=stored_context_id))
```

## Content support

Text, raw bytes and URL Parts map to AgentScope text/data blocks in both
directions. The server additionally publishes A2UI activity and confirmation
surfaces over A2A DataParts. The `A2AAgent` client adapter does not currently
interpret A2UI surfaces or submit renderer actions. Other structured-data
Parts, thinking/tool/hint blocks and push notifications are not supported by
the adapter.

`compress_context()` is intentionally a no-op: the remote server, not the
local adapter, owns and compresses its conversation context.

## Local E2E test

```bash
uv run --extra dev pytest tests/a2a_agent_e2e_test.py -v
```

The test starts an ephemeral A2A 1.0 server on localhost with the official SDK
server primitives, connects `A2AAgent` to it over real JSON-RPC/SSE, and checks
streamed artifact handling and context continuity across two turns. It needs no
model, API key or network access.
