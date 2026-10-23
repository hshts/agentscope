# -*- coding: utf-8 -*-
"""Unit tests for SessionArtifactsMiddleware and artifact blob handling."""
import json
from types import SimpleNamespace
from unittest.async_case import IsolatedAsyncioTestCase

from agentscope.message import TextBlock, ToolCallBlock, ToolResultState
from agentscope.middleware import SessionArtifactsMiddleware
from agentscope.middleware._session_artifacts import (
    extract_artifact_paths,
    is_artifact_tool_name,
    normalize_artifact_path,
)
from agentscope.state import AgentState
from agentscope.tool import ToolChunk, ToolResponse


def _tool_call(name: str, args: dict) -> ToolCallBlock:
    return ToolCallBlock(
        id="tc-1",
        name=name,
        input=json.dumps(args),
    )


def _response(
    *,
    state: ToolResultState = ToolResultState.SUCCESS,
    content: str = "ok",
    metadata: dict | None = None,
) -> ToolResponse:
    return ToolResponse(
        content=[TextBlock(text=content)],
        state=state,
        metadata=metadata or {},
    )


async def _run(
    middleware: SessionArtifactsMiddleware,
    agent: SimpleNamespace,
    tool_call: ToolCallBlock,
    items: list,
) -> list:
    """Drive on_acting with a scripted downstream generator."""

    async def next_handler():
        for item in items:
            yield item

    out = []
    async for item in middleware.on_acting(
        agent=agent,
        input_kwargs={"tool_call": tool_call},
        next_handler=next_handler,
    ):
        out.append(item)
    return out


async def _async(value):
    """Wrap a value in a coroutine (test helper)."""
    return value


class SessionArtifactHelpersTest(IsolatedAsyncioTestCase):
    """Path normalization and extraction helpers."""

    def test_normalize_absolute_path_is_kept(self) -> None:
        self.assertEqual(
            normalize_artifact_path("/tmp/ws/docs/a.md", "/tmp/ws"),
            "/tmp/ws/docs/a.md",
        )

    def test_normalize_relative_path_joins_workspace(self) -> None:
        self.assertEqual(
            normalize_artifact_path("docs/a.md", "/tmp/ws"),
            "/tmp/ws/docs/a.md",
        )

    def test_normalize_rejects_directories_and_extensionless(self) -> None:
        self.assertEqual(
            normalize_artifact_path("/tmp/ws/docs", "/tmp/ws"),
            "",
        )
        self.assertEqual(normalize_artifact_path("notes", "/tmp/ws"), "")
        self.assertEqual(normalize_artifact_path("", "/tmp/ws"), "")

    def test_normalize_rejects_internal_workspace_dirs(self) -> None:
        self.assertEqual(
            normalize_artifact_path("/tmp/ws/skills/a.md", "/tmp/ws"),
            "",
        )
        self.assertEqual(
            normalize_artifact_path("/tmp/ws/sessions/a.md", "/tmp/ws"),
            "",
        )

    def test_normalize_strips_file_url(self) -> None:
        self.assertEqual(
            normalize_artifact_path("file:///tmp/ws/a.md", "/tmp/ws"),
            "/tmp/ws/a.md",
        )

    def test_normalize_keeps_windows_absolute_path(self) -> None:
        self.assertEqual(
            normalize_artifact_path(r"C:\ws\docs\a.md", "/tmp/ws"),
            "C:/ws/docs/a.md",
        )

    def test_is_artifact_tool_name(self) -> None:
        self.assertTrue(is_artifact_tool_name("Write"))
        self.assertTrue(is_artifact_tool_name("Edit"))
        self.assertTrue(is_artifact_tool_name("mcp__filesystem__write_file"))
        self.assertTrue(is_artifact_tool_name("files/Edit"))
        self.assertFalse(is_artifact_tool_name("Read"))
        self.assertFalse(is_artifact_tool_name("Bash"))
        self.assertFalse(is_artifact_tool_name(""))

    def test_extract_prefers_metadata_over_args(self) -> None:
        self.assertEqual(
            extract_artifact_paths(
                tool_name="Write",
                args={"file_path": "/tmp/ws/a.md"},
                metadata={"file_path": "/tmp/ws/b.md"},
                workspace_dir="/tmp/ws",
            ),
            ["/tmp/ws/b.md"],
        )

    def test_extract_falls_back_to_result_text(self) -> None:
        self.assertEqual(
            extract_artifact_paths(
                tool_name="Edit",
                args={},
                content=[TextBlock(text="Updated /tmp/ws/docs/a.md")],
                workspace_dir="/tmp/ws",
            ),
            ["/tmp/ws/docs/a.md"],
        )

    def test_extract_ignores_non_artifact_tools(self) -> None:
        self.assertEqual(
            extract_artifact_paths(
                tool_name="Read",
                args={"file_path": "/tmp/ws/a.md"},
                workspace_dir="/tmp/ws",
            ),
            [],
        )


class SessionArtifactsMiddlewareTest(IsolatedAsyncioTestCase):
    """The middleware records successful file-producing tool calls."""

    async def test_records_successful_write(self) -> None:
        middleware = SessionArtifactsMiddleware(
            workspace_dir="/tmp/ws",
            agent_id="agent-1",
        )
        agent = SimpleNamespace(state=AgentState())
        tool_call = _tool_call("Write", {"file_path": "/tmp/ws/docs/a.md"})
        response = _response(
            metadata={"file_path": "/tmp/ws/docs/a.md"},
        )

        out = await _run(
            middleware,
            agent,
            tool_call,
            [ToolChunk(content=[TextBlock(text="writing")]), response],
        )

        self.assertEqual(len(out), 2)
        self.assertIs(out[-1], response)
        entries = agent.state.artifact_context.entries
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].path, "/tmp/ws/docs/a.md")
        self.assertEqual(entries[0].agent_id, "agent-1")
        self.assertEqual(
            agent.state.artifact_context.paths,
            ["/tmp/ws/docs/a.md"],
        )

    async def test_records_from_args_when_response_is_placeholder(
        self,
    ) -> None:
        middleware = SessionArtifactsMiddleware(workspace_dir="/tmp/ws")
        agent = SimpleNamespace(state=AgentState())
        tool_call = _tool_call("Edit", {"file_path": "docs/a.md"})
        placeholder = ToolResponse(
            content=[TextBlock(text="started in background")],
        )

        await _run(middleware, agent, tool_call, [placeholder])

        self.assertEqual(
            [entry.path for entry in agent.state.artifact_context.entries],
            ["/tmp/ws/docs/a.md"],
        )

    async def test_skips_failed_tool_call(self) -> None:
        middleware = SessionArtifactsMiddleware(workspace_dir="/tmp/ws")
        agent = SimpleNamespace(state=AgentState())
        tool_call = _tool_call("Write", {"file_path": "/tmp/ws/a.md"})

        await _run(
            middleware,
            agent,
            tool_call,
            [
                _response(
                    state=ToolResultState.ERROR,
                    metadata={"file_path": "/tmp/ws/a.md"},
                ),
            ],
        )

        self.assertEqual(agent.state.artifact_context.entries, [])

    async def test_dedupes_and_trims_oldest(self) -> None:
        middleware = SessionArtifactsMiddleware(
            workspace_dir="/tmp/ws",
            max_entries=2,
        )
        agent = SimpleNamespace(state=AgentState())

        for name in ("a.md", "b.md", "c.md", "a.md"):
            await _run(
                middleware,
                agent,
                _tool_call("Write", {"file_path": f"/tmp/ws/{name}"}),
                [_response(metadata={"file_path": f"/tmp/ws/{name}"})],
            )

        self.assertEqual(
            [entry.path for entry in agent.state.artifact_context.entries],
            ["/tmp/ws/c.md", "/tmp/ws/a.md"],
        )


class _FakeStore:
    """Records the calls the middleware makes to the storage hook."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple] = []
        self._fail = fail

    async def append_session_artifacts(
        self,
        user_id: str,
        agent_id: str,
        session_id: str,
        artifacts: list,
        *,
        max_entries: int = 200,
    ) -> list:
        if self._fail:
            raise RuntimeError("storage unavailable")
        self.calls.append((user_id, agent_id, session_id, list(artifacts)))
        return list(artifacts)


class SessionArtifactStorageTest(IsolatedAsyncioTestCase):
    """The optional store hook persists each newly added entry."""

    async def test_store_receives_new_entries(self) -> None:
        store = _FakeStore()
        middleware = SessionArtifactsMiddleware(
            workspace_dir="/tmp/ws",
            agent_id="agent-1",
            user_id="user-1",
            store=store,
        )
        agent = SimpleNamespace(state=AgentState(session_id="session-1"))
        tool_call = _tool_call("Write", {"file_path": "/tmp/ws/a.md"})

        await _run(
            middleware,
            agent,
            tool_call,
            [_response(metadata={"file_path": "/tmp/ws/a.md"})],
        )

        self.assertEqual(len(store.calls), 1)
        user_id, agent_id, session_id, artifacts = store.calls[0]
        self.assertEqual(
            (user_id, agent_id, session_id),
            ("user-1", "agent-1", "session-1"),
        )
        self.assertEqual(
            [(entry.path, entry.agent_id) for entry in artifacts],
            [("/tmp/ws/a.md", "agent-1")],
        )

    async def test_store_not_called_for_duplicate(self) -> None:
        store = _FakeStore()
        middleware = SessionArtifactsMiddleware(
            workspace_dir="/tmp/ws",
            agent_id="agent-1",
            user_id="user-1",
            store=store,
        )
        agent = SimpleNamespace(state=AgentState(session_id="session-1"))
        tool_call = _tool_call("Write", {"file_path": "/tmp/ws/a.md"})
        response = _response(metadata={"file_path": "/tmp/ws/a.md"})

        await _run(middleware, agent, tool_call, [response])
        await _run(middleware, agent, tool_call, [response])

        self.assertEqual(len(store.calls), 1)

    async def test_store_failure_keeps_in_memory_entry(self) -> None:
        store = _FakeStore(fail=True)
        middleware = SessionArtifactsMiddleware(
            workspace_dir="/tmp/ws",
            agent_id="agent-1",
            user_id="user-1",
            store=store,
        )
        agent = SimpleNamespace(state=AgentState(session_id="session-1"))
        tool_call = _tool_call("Write", {"file_path": "/tmp/ws/a.md"})

        with self.assertLogs("as", level="WARNING"):
            out = await _run(
                middleware,
                agent,
                tool_call,
                [_response(metadata={"file_path": "/tmp/ws/a.md"})],
            )

        self.assertEqual(
            [entry.path for entry in agent.state.artifact_context.entries],
            ["/tmp/ws/a.md"],
        )
        self.assertEqual(len(out), 1)

    async def test_storage_base_default_appends_to_state(self) -> None:
        from unittest.mock import AsyncMock

        from agentscope.app.storage import (
            SessionConfig,
            SessionRecord,
            StorageBase,
        )
        from agentscope.state import SessionArtifact

        record = SessionRecord(
            id="session-1",
            user_id="user-1",
            agent_id="agent-1",
            config=SessionConfig(workspace_id="ws-1"),
            state=AgentState(session_id="session-1"),
        )
        backend = SimpleNamespace(
            get_session=AsyncMock(return_value=record),
            update_session_state=AsyncMock(),
        )

        added = await StorageBase.append_session_artifacts(
            backend,
            "user-1",
            "agent-1",
            "session-1",
            [
                SessionArtifact(path="/tmp/ws/a.md", agent_id="agent-1"),
                SessionArtifact(path="/tmp/ws/a.md", agent_id="agent-1"),
            ],
        )

        self.assertEqual([entry.path for entry in added], ["/tmp/ws/a.md"])
        backend.update_session_state.assert_awaited_once_with(
            "user-1",
            "agent-1",
            "session-1",
            record.state,
        )

    async def test_storage_base_ignores_missing_session(self) -> None:
        from unittest.mock import AsyncMock

        from agentscope.app.storage import StorageBase
        from agentscope.state import SessionArtifact

        backend = SimpleNamespace(
            get_session=AsyncMock(return_value=None),
            update_session_state=AsyncMock(),
        )

        added = await StorageBase.append_session_artifacts(
            backend,
            "user-1",
            "agent-1",
            "missing",
            [SessionArtifact(path="/tmp/ws/a.md")],
        )

        self.assertEqual(added, [])
        backend.update_session_state.assert_not_awaited()


class SQLSessionArtifactStorageTest(IsolatedAsyncioTestCase):
    """The default storage hook round-trips through a real backend."""

    async def asyncSetUp(self) -> None:
        from agentscope.app.storage import AsyncSQLAlchemyStorage

        self._storage = AsyncSQLAlchemyStorage(
            "sqlite+aiosqlite:///:memory:",
            create_tables=True,
        )
        await self._storage.__aenter__()

    async def asyncTearDown(self) -> None:
        await self._storage.__aexit__(None, None, None)

    async def test_append_and_reload_session_artifacts(self) -> None:
        from agentscope.app.storage import SessionConfig
        from agentscope.state import SessionArtifact

        record = await self._storage.upsert_session(
            "user-1",
            "agent-1",
            SessionConfig(workspace_id="ws-1"),
        )
        added = await self._storage.append_session_artifacts(
            "user-1",
            "agent-1",
            record.id,
            [
                SessionArtifact(path="/tmp/ws/a.md", agent_id="agent-1"),
                SessionArtifact(
                    path="/workspace/b.md",
                    agent_id="agent-1",
                    storage="sandbox",
                    blob_uri="local://artifacts/s1/workspace/b.md",
                ),
            ],
        )
        self.assertEqual(
            [entry.path for entry in added],
            ["/tmp/ws/a.md", "/workspace/b.md"],
        )

        loaded = await self._storage.get_session(
            "user-1",
            "agent-1",
            record.id,
        )
        entries = loaded.state.artifact_context.entries
        self.assertEqual(
            [entry.path for entry in entries],
            ["/tmp/ws/a.md", "/workspace/b.md"],
        )
        self.assertEqual(entries[0].agent_id, "agent-1")
        self.assertEqual(entries[1].storage, "sandbox")
        self.assertEqual(
            entries[1].blob_uri,
            "local://artifacts/s1/workspace/b.md",
        )


class _RemoteBackend:
    """Minimal sandbox backend with a read-only file map."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = dict(files)

    async def read_file(self, path: str) -> bytes:
        return self.files[path]

    async def stat(self, path: str):
        from agentscope.tool._builtin._backend import DirEntry

        if path not in self.files:
            return None
        return DirEntry(
            name=path.rsplit("/", 1)[-1],
            is_dir=False,
            size_bytes=len(self.files[path]),
        )

    async def read_stream(self, path: str, chunk_size: int = 1024):
        data = self.files[path]
        for index in range(0, len(data), chunk_size):
            yield data[index : index + chunk_size]


class _FakeBlobStore:
    """Capture uploaded blobs by key."""

    def __init__(self, *, fail: bool = False) -> None:
        self.blobs: dict[str, bytes] = {}
        self.fail = fail

    async def write_stream(self, key: str, stream) -> str:
        if self.fail:
            raise RuntimeError("upload failed")
        self.blobs[key] = stream.read()
        return f"mem://{key}"


class BlobUploadTest(IsolatedAsyncioTestCase):
    """Sandbox artifacts are uploaded; local artifacts are not."""

    async def test_sandbox_artifact_is_uploaded(self) -> None:
        store = _FakeBlobStore()
        middleware = SessionArtifactsMiddleware(
            workspace_dir="/workspace",
            agent_id="agent-1",
            backend=_RemoteBackend({"/workspace/a.md": b"hello"}),
            blob_store=store,
        )
        agent = SimpleNamespace(state=AgentState(session_id="session-1"))
        tool_call = _tool_call("Write", {"file_path": "/workspace/a.md"})

        await _run(
            middleware,
            agent,
            tool_call,
            [_response(metadata={"file_path": "/workspace/a.md"})],
        )

        entry = agent.state.artifact_context.entries[0]
        self.assertEqual(entry.storage, "sandbox")
        self.assertEqual(
            entry.blob_uri,
            "mem://artifacts/session-1/workspace/a.md",
        )
        self.assertEqual(
            store.blobs["artifacts/session-1/workspace/a.md"],
            b"hello",
        )

    async def test_local_artifact_is_not_uploaded(self) -> None:
        store = _FakeBlobStore()
        middleware = SessionArtifactsMiddleware(
            workspace_dir="/tmp/ws",
            blob_store=store,
        )
        agent = SimpleNamespace(state=AgentState(session_id="session-1"))
        tool_call = _tool_call("Write", {"file_path": "/tmp/ws/a.md"})

        await _run(
            middleware,
            agent,
            tool_call,
            [_response(metadata={"file_path": "/tmp/ws/a.md"})],
        )

        self.assertEqual(store.blobs, {})
        self.assertEqual(
            agent.state.artifact_context.entries[0].blob_uri,
            "",
        )

    async def test_upload_failure_keeps_entry_readable_via_backend(
        self,
    ) -> None:
        middleware = SessionArtifactsMiddleware(
            workspace_dir="/workspace",
            backend=_RemoteBackend({"/workspace/a.md": b"hello"}),
            blob_store=_FakeBlobStore(fail=True),
        )
        agent = SimpleNamespace(state=AgentState(session_id="session-1"))
        tool_call = _tool_call("Write", {"file_path": "/workspace/a.md"})

        with self.assertLogs("as", level="WARNING"):
            await _run(
                middleware,
                agent,
                tool_call,
                [_response(metadata={"file_path": "/workspace/a.md"})],
            )

        entry = agent.state.artifact_context.entries[0]
        self.assertEqual(entry.blob_uri, "")

    async def test_oversized_file_is_not_uploaded(self) -> None:
        store = _FakeBlobStore()
        middleware = SessionArtifactsMiddleware(
            workspace_dir="/workspace",
            backend=_RemoteBackend({"/workspace/big.bin": b"x" * 32}),
            blob_store=store,
            max_blob_bytes=8,
        )
        agent = SimpleNamespace(state=AgentState(session_id="session-1"))
        tool_call = _tool_call("Write", {"file_path": "/workspace/big.bin"})

        await _run(
            middleware,
            agent,
            tool_call,
            [_response(metadata={"file_path": "/workspace/big.bin"})],
        )

        self.assertEqual(store.blobs, {})
        self.assertEqual(
            agent.state.artifact_context.entries[0].blob_uri,
            "",
        )

    def test_blob_key_is_sanitized(self) -> None:
        from agentscope.middleware._session_artifacts import blob_key

        self.assertEqual(
            blob_key("artifacts", "s/1", "/workspace/../out/a.md"),
            "artifacts/s_1/workspace/out/a.md",
        )


class BlobUploadRetryTest(IsolatedAsyncioTestCase):
    """A previously failed upload is retried on the next write."""

    async def test_retries_entry_with_empty_blob_uri(self) -> None:
        from agentscope.state import SessionArtifact

        store = _FakeBlobStore()
        middleware = SessionArtifactsMiddleware(
            workspace_dir="/workspace",
            backend=_RemoteBackend({"/workspace/a.md": b"hello"}),
            blob_store=store,
        )
        state = AgentState(session_id="session-1")
        state.artifact_context.add_entries([
            SessionArtifact(
                path="/workspace/a.md",
                storage="sandbox",
                blob_uri="",
            ),
        ])
        agent = SimpleNamespace(state=state)
        tool_call = _tool_call("Write", {"file_path": "/workspace/a.md"})

        await _run(
            middleware,
            agent,
            tool_call,
            [_response(metadata={"file_path": "/workspace/a.md"})],
        )

        self.assertEqual(len(state.artifact_context.entries), 1)
        self.assertEqual(
            state.artifact_context.entries[0].blob_uri,
            "mem://artifacts/session-1/workspace/a.md",
        )


class ArtifactStorageModeTest(IsolatedAsyncioTestCase):
    """The ``storage`` flag distinguishes host paths from sandbox paths."""

    def test_local_backend_maps_to_local(self) -> None:
        from agentscope.middleware._session_artifacts import storage_mode_for
        from agentscope.tool import LocalBackend

        self.assertEqual(storage_mode_for(LocalBackend()), "local")
        self.assertEqual(storage_mode_for(None), "local")

    def test_remote_backend_maps_to_sandbox(self) -> None:
        from agentscope.middleware._session_artifacts import storage_mode_for

        self.assertEqual(storage_mode_for(_RemoteBackend({})), "sandbox")

    async def test_middleware_stamps_sandbox_mode(self) -> None:
        backend = _RemoteBackend({})
        middleware = SessionArtifactsMiddleware(
            workspace_dir="/workspace",
            agent_id="agent-1",
            backend=backend,
        )
        agent = SimpleNamespace(state=AgentState(session_id="session-1"))
        tool_call = _tool_call("Write", {"file_path": "/workspace/a.md"})
        response = _response(metadata={"file_path": "/workspace/a.md"})

        await _run(middleware, agent, tool_call, [response])

        entry = agent.state.artifact_context.entries[0]
        self.assertEqual(entry.path, "/workspace/a.md")
        self.assertEqual(entry.storage, "sandbox")

    async def test_middleware_defaults_to_local(self) -> None:
        middleware = SessionArtifactsMiddleware(workspace_dir="/tmp/ws")
        agent = SimpleNamespace(state=AgentState())
        tool_call = _tool_call("Write", {"file_path": "/tmp/ws/a.md"})

        await _run(
            middleware,
            agent,
            tool_call,
            [_response(metadata={"file_path": "/tmp/ws/a.md"})],
        )

        self.assertEqual(
            agent.state.artifact_context.entries[0].storage,
            "local",
        )

    async def test_deduped_entry_keeps_its_storage_flag(self) -> None:
        from agentscope.state import SessionArtifact

        state = AgentState()
        state.artifact_context.add_entries([
            SessionArtifact(path="/tmp/ws/a.md", storage="sandbox"),
        ])
        added = state.artifact_context.add_entries([
            SessionArtifact(path="/tmp/ws/a.md", storage="local"),
        ])
        self.assertEqual(added, [])
        self.assertEqual(
            state.artifact_context.entries[0].storage,
            "sandbox",
        )

    async def test_add_paths_stamps_storage(self) -> None:
        state = AgentState()
        state.artifact_context.add_paths(
            ["/workspace/a.md"],
            storage="sandbox",
        )
        self.assertEqual(
            state.artifact_context.entries[0].storage,
            "sandbox",
        )


class BlobLookupTest(IsolatedAsyncioTestCase):
    """The download path prefers an uploaded copy for sandbox entries."""

    def _record(self, entry_storage: str, blob_uri: str):
        from agentscope.app.storage import SessionConfig, SessionRecord
        from agentscope.state import SessionArtifact

        record = SessionRecord(
            id="session-1",
            user_id="user-1",
            agent_id="agent-1",
            config=SessionConfig(workspace_id="ws-1"),
            state=AgentState(session_id="session-1"),
        )
        record.state.artifact_context.add_entries([
            SessionArtifact(
                path="/workspace/a.md",
                storage=entry_storage,
                blob_uri=blob_uri,
            ),
        ])
        return record

    def _service(self, record, blob_store) -> object:
        from agentscope.app._service import ArtifactBlobService

        storage = SimpleNamespace(
            get_session=lambda *args: _async(record),
        )
        return ArtifactBlobService(storage=storage, blob_store=blob_store)

    async def test_returns_blob_uri_for_sandbox_entry(self) -> None:
        service = self._service(
            self._record(
                "sandbox",
                "mem://artifacts/session-1/workspace/a.md",
            ),
            _FakeBlobStore(),
        )
        uri = await service.blob_uri(
            "user-1",
            "agent-1",
            "session-1",
            "/workspace/a.md",
            "/workspace/a.md",
        )
        self.assertEqual(
            uri,
            "mem://artifacts/session-1/workspace/a.md",
        )

    async def test_lookup_by_raw_path_without_target(self) -> None:
        service = self._service(
            self._record(
                "sandbox",
                "mem://artifacts/session-1/workspace/a.md",
            ),
            _FakeBlobStore(),
        )
        uri = await service.blob_uri(
            "user-1",
            "agent-1",
            "session-1",
            raw_path="/workspace/a.md",
        )
        self.assertEqual(
            uri,
            "mem://artifacts/session-1/workspace/a.md",
        )

    async def test_ignores_local_and_unuploaded_entries(self) -> None:
        for storage_kind, blob_uri in (
            ("local", "mem://x"),
            ("sandbox", ""),
        ):
            service = self._service(
                self._record(storage_kind, blob_uri),
                _FakeBlobStore(),
            )
            self.assertEqual(
                await service.blob_uri(
                    "user-1",
                    "agent-1",
                    "session-1",
                    "/workspace/a.md",
                    "/workspace/a.md",
                ),
                "",
            )

    async def test_missing_session_returns_empty(self) -> None:
        service = self._service(None, _FakeBlobStore())
        self.assertEqual(
            await service.blob_uri(
                "user-1",
                "agent-1",
                "session-1",
                "/workspace/a.md",
                "/workspace/a.md",
            ),
            "",
        )

    async def test_no_blob_store_short_circuits(self) -> None:
        service = self._service(
            self._record("sandbox", "mem://x"),
            None,
        )
        self.assertEqual(
            await service.blob_uri(
                "user-1",
                "agent-1",
                "session-1",
                "/workspace/a.md",
                "/workspace/a.md",
            ),
            "",
        )

    async def test_streams_uploaded_blob(self) -> None:
        import io
        import tempfile

        from agentscope.app.rag.blob_store import LocalBlobStore

        with tempfile.TemporaryDirectory() as root:
            store = LocalBlobStore(root)
            async with store:
                uri = await store.write_stream(
                    "artifacts/session-1/workspace/a.md",
                    io.BytesIO(b"uploaded"),
                )
                service = self._service(None, store)
                chunks = [
                    chunk async for chunk in service.stream(uri)
                ]
        self.assertEqual(b"".join(chunks), b"uploaded")
