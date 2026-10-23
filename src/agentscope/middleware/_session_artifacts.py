# -*- coding: utf-8 -*-
"""Middleware that records workspace file artifacts on the session state.

Whenever a session's agent successfully runs a file-producing tool
(``Write``, ``Edit``, an MCP ``write_file``, ...), the paths of the files
it produced are appended to
:attr:`agentscope.state.AgentState.artifact_context`.  Because the agent
state is persisted with the session record, the artifact list survives
across turns and is returned by the session APIs without any extra
storage round-trip.

Each entry is stamped with where the agent ran:

- ``storage="local"`` — ``LocalBackend`` wrote to a host path the API
  can open directly.
- ``storage="sandbox"`` — the path only exists inside a container /
  remote sandbox, so consumers fetch it through the workspace API.

Pass a ``store`` (the app ``StorageBase``) to persist each new entry
immediately, and a ``blob_store`` to upload sandbox files so they stay
downloadable after the sandbox is gone.
"""
from __future__ import annotations

import io
import json
import os
import posixpath
import re
from collections.abc import (
    AsyncGenerator,
    Callable,
    Iterable,
    Mapping,
    Sequence,
)
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol
from urllib.parse import unquote

from .._logging import logger
from ..message import DataBlock, TextBlock
from ..tool import ToolResponse
from ._base import MiddlewareBase

if TYPE_CHECKING:
    from ..agent import Agent
    from ..message import ToolCallBlock
    from ..state import SessionArtifact
    from ..tool import BackendBase


def storage_mode_for(
    backend: "BackendBase | None",
) -> Literal["local", "sandbox"]:
    """Whether paths produced through *backend* are host-visible.

    ``LocalBackend`` reads the host filesystem directly, so a path it
    writes can be opened by the API process. Every other backend (Docker,
    E2B, Daytona, Kubernetes, AppleContainer, Bubblewrap) is only
    reachable through the backend, so the frontend must go through the
    session's workspace API instead.

    Args:
        backend (`BackendBase | None`):
            The session's backend, or ``None`` when unknown.

    Returns:
        `Literal["local", "sandbox"]`: The mode stamped on new artifacts.
    """
    if backend is None:
        return "local"
    # Imported lazily: ``tool`` imports ``state``, so a module-level
    # import here would close the cycle through ``middleware``.
    from ..tool._builtin._backend import LocalBackend

    return "local" if isinstance(backend, LocalBackend) else "sandbox"


class SessionArtifactStore(Protocol):
    """Persistence hook for :class:`SessionArtifactsMiddleware`.

    Implemented by storage backends; the shape matches
    :meth:`agentscope.app.storage.StorageBase.append_session_artifacts`.
    """

    async def append_session_artifacts(
        self,
        user_id: str,
        agent_id: str,
        session_id: str,
        artifacts: Sequence["SessionArtifact"],
        *,
        max_entries: int = 200,
    ) -> list["SessionArtifact"]:
        """Append *artifacts* to the stored session and return the new ones."""
        raise NotImplementedError


class ArtifactBlobStore(Protocol):
    """The subset of ``BlobStoreBase`` the middleware uploads through."""

    async def write_stream(self, key: str, stream: Any) -> str:
        """Write a blob under *key* and return its URI."""
        raise NotImplementedError


#: Tools (matched by base name, case-insensitively) whose successful
#: output is treated as a workspace file artifact.
DEFAULT_ARTIFACT_TOOL_NAMES = frozenset(
    {
        "write",
        "edit",
        "write_file",
        "edit_file",
        "create_file",
        "apply_patch",
    },
)

#: Argument / metadata keys that carry a file path.
_PATH_KEYS = (
    "file_path",
    "path",
    "filepath",
    "dest",
    "target_path",
    "output_path",
)

_PATH_EXT_RE = re.compile(r"\.[A-Za-z][A-Za-z0-9._+-]{0,11}$")

#: Fallback path scanner for tool results that only mention the file in
#: their text (custom tools, or ``Write``/``Edit`` on an older backend
#: that does not attach metadata).
_TEXT_PATH_RE = re.compile(
    r"(?<![\w.-])"
    r"((?:/|\.{1,2}/)?[\w@%+~-]+(?:/[\w@%+~.-]+)*\.[A-Za-z][\w.+-]{0,11})"
    r"(?![\w.-])",
)

#: Workspace-internal directories that must never surface as artifacts.
_INTERNAL_DIRS = frozenset({"skills", "sessions"})

#: Ceiling on one upload. ``read_file`` returns the whole payload, so a
#: file above this is left in the sandbox and served from the backend.
DEFAULT_MAX_BLOB_BYTES = 50 * 1024 * 1024

_SAFE_KEY_CHARS = re.compile(r"[^\w.@%+~-]+")


def _tool_base_name(name: str) -> str:
    """Return the unprefixed, lower-cased tool name.

    Handles the naming schemes used by AgentScope: MCP tools are exposed
    as ``mcp__<server>__<tool>`` and grouped tools may carry a
    ``group/tool`` prefix.
    """
    base = (name or "").strip().rsplit("/", 1)[-1]
    if "__" in base:
        base = base.rsplit("__", 1)[-1]
    return base.lower()


def is_artifact_tool_name(
    name: str | None,
    tool_names: Iterable[str] | None = None,
) -> bool:
    """Whether *name* is a file-producing tool.

    Args:
        name (`str | None`):
            The tool name as it appears on the tool call.
        tool_names (`Iterable[str] | None`, optional):
            Override the default set of artifact tool names.

    Returns:
        `bool`: True when the tool's successful result should be
        recorded as an artifact.
    """
    names = (
        DEFAULT_ARTIFACT_TOOL_NAMES
        if tool_names is None
        else frozenset(str(n).strip().lower() for n in tool_names)
    )
    return _tool_base_name(name or "") in names


def _workspace_base(workspace_dir: str | Path | None) -> str:
    """Normalize the workspace root into a posix string."""
    if workspace_dir is None:
        return ""
    base = str(workspace_dir).replace("\\", "/").rstrip("/")
    return "" if base in {"", "/"} else base


def _is_absolute_path(path: str) -> bool:
    """Whether *path* is absolute on posix or Windows."""
    return posixpath.isabs(path) or bool(re.match(r"^[A-Za-z]:/", path))


def _is_artifact_path(path: str, base: str) -> bool:
    """Reject directories, extension-less names, and internal files."""
    parts = [part for part in path.split("/") if part]
    if not parts:
        return False
    name = parts[-1]
    if name in {".", ".."} or not _PATH_EXT_RE.search(name):
        return False
    if any(part in {".skills", ".mcp"} for part in parts):
        return False
    if base and path.startswith(base + "/"):
        relative = path[len(base) + 1 :]
    else:
        relative = path
    first = relative.split("/", 1)[0]
    return first not in _INTERNAL_DIRS


def normalize_artifact_path(
    path: str,
    workspace_dir: str | Path | None = None,
) -> str:
    """Normalize *path* into a stable, workspace-scoped artifact path.

    Absolute paths are kept as-is; relative paths are joined under
    *workspace_dir*.  No filesystem I/O is performed — the path may live
    inside a sandbox that the host cannot resolve.

    Args:
        path (`str`):
            The raw path taken from tool arguments, metadata, or output.
        workspace_dir (`str | Path | None`, optional):
            The workspace root used to resolve relative paths.  When
            omitted, relative paths are discarded.

    Returns:
        `str`: The normalized path, or ``""`` when *path* is not a
        user-facing file artifact.
    """
    raw = (path or "").strip().strip("\"'`")
    if not raw:
        return ""
    if raw.startswith("file://"):
        raw = raw[len("file://") :]
        if raw.startswith("localhost/"):
            raw = raw[len("localhost") :]
        raw = unquote(raw)
    raw = raw.replace("\\", "/")
    if raw.startswith(("http://", "https://", "<")):
        return ""

    base = _workspace_base(workspace_dir)
    if not _is_absolute_path(raw):
        if not base:
            return ""
        raw = posixpath.join(base, raw)

    normalized = posixpath.normpath(raw)
    if not _is_artifact_path(normalized, base):
        return ""
    return normalized


def _dedupe_key(path: str) -> str:
    """The key used to dedupe an artifact path within one session."""
    value = (path or "").strip()
    return value.casefold() if os.name == "nt" else value


def _dedupe_paths(paths: Sequence[str]) -> list[str]:
    """Drop empty values and duplicates, preserving the first occurrence."""
    out: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not path:
            continue
        key = path.casefold() if os.name == "nt" else path
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _paths_from_mapping(
    data: Mapping[str, Any] | None,
    workspace_dir: str | Path | None,
) -> list[str]:
    """Collect path-valued keys from tool arguments or result metadata."""
    if not isinstance(data, Mapping):
        return []
    out: list[str] = []
    for key in _PATH_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            out.append(normalize_artifact_path(value, workspace_dir))
    return out


def _content_text(
    content: Sequence[TextBlock | DataBlock] | str | None,
) -> str:
    """Flatten a tool result into text for the fallback path scan."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    chunks: list[str] = []
    for block in content:
        if isinstance(block, TextBlock):
            chunks.append(block.text)
    return "\n".join(chunks)


def _paths_from_text(
    text: str,
    workspace_dir: str | Path | None,
) -> list[str]:
    """Scan tool output text for path-like tokens."""
    if not text:
        return []
    out: list[str] = []
    for match in _TEXT_PATH_RE.finditer(text):
        out.append(normalize_artifact_path(match.group(1), workspace_dir))
    return out


def extract_artifact_paths(
    *,
    tool_name: str,
    args: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    content: Sequence[TextBlock | DataBlock] | str | None = None,
    workspace_dir: str | Path | None = None,
    tool_names: Iterable[str] | None = None,
) -> list[str]:
    """Extract artifact paths from one successful tool call.

    Structured paths (result metadata first, then tool arguments) win;
    the textual scan is only a fallback so ordinary command output does
    not flood the artifact list.

    Args:
        tool_name (`str`):
            The tool name from the tool call.
        args (`Mapping[str, Any] | None`, optional):
            Parsed tool-call arguments.
        metadata (`Mapping[str, Any] | None`, optional):
            The final :class:`~agentscope.tool.ToolResponse` metadata.
        content (`Sequence[TextBlock | DataBlock] | str | None`, optional):
            The final tool result content, used for the fallback scan.
        workspace_dir (`str | Path | None`, optional):
            The workspace root used to resolve relative paths.
        tool_names (`Iterable[str] | None`, optional):
            Override the default artifact tool set.

    Returns:
        `list[str]`: Normalized, de-duplicated artifact paths.
    """
    if not is_artifact_tool_name(tool_name, tool_names):
        return []

    structured = _dedupe_paths(
        _paths_from_mapping(metadata, workspace_dir),
    )
    if structured:
        return structured

    structured = _dedupe_paths(
        _paths_from_mapping(args, workspace_dir),
    )
    if structured:
        return structured

    return _dedupe_paths(
        _paths_from_text(_content_text(content), workspace_dir),
    )


def _parse_tool_input(raw: Any) -> Mapping[str, Any] | None:
    """Parse the raw JSON string of a :class:`ToolCallBlock`."""
    if isinstance(raw, Mapping):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


def blob_key(prefix: str, session_id: str, path: str) -> str:
    """Build a blob key from a sandbox path.

    Segments are sanitized and traversal is stripped so the key cannot
    escape the store's namespace.

    Args:
        prefix (`str`):
            Store-level prefix, e.g. ``"artifacts"``.
        session_id (`str`):
            The owning session, used as a namespace.
        path (`str`):
            The sandbox path of the file.

    Returns:
        `str`: A relative, storage-safe key.
    """
    segments: list[str] = []
    if prefix.strip("/"):
        segments.append(_SAFE_KEY_CHARS.sub("_", prefix.strip("/")))
    namespace = _SAFE_KEY_CHARS.sub("_", (session_id or "default").strip())
    segments.append(namespace)
    for part in (path or "").replace("\\", "/").split("/"):
        if part in {"", ".", ".."}:
            continue
        segments.append(_SAFE_KEY_CHARS.sub("_", part))
    return "/".join(segment for segment in segments if segment)


class SessionArtifactsMiddleware(MiddlewareBase):
    """Record workspace files produced by successful tool calls.

    Hooks ``on_acting``: the middleware yields every downstream item
    unchanged, then inspects the final :class:`~agentscope.tool.ToolResponse`
    and appends the paths of files written or edited by the tool to
    ``agent.state.artifact_context``.  Failed, denied, or interrupted
    calls are ignored.

    Example:
        ```python
        agent = Agent(
            ...,
            middlewares=[
                SessionArtifactsMiddleware(
                    workspace_dir=workspace.workdir,
                    store=storage,          # optional immediate persistence
                    user_id=user_id,
                    agent_id=agent_id,
                ),
            ],
        )
        # Later: agent.state.artifact_context.entries
        ```
    """

    def __init__(
        self,
        *,
        workspace_dir: str | Path | None = None,
        agent_id: str = "",
        user_id: str = "",
        store: SessionArtifactStore | None = None,
        backend: "BackendBase | None" = None,
        blob_store: ArtifactBlobStore | None = None,
        blob_key_prefix: str = "artifacts",
        max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES,
        tool_names: Iterable[str] | None = None,
        extra_tool_names: Iterable[str] = (),
        max_entries: int = 200,
    ) -> None:
        """Bind the workspace, the artifact policy, and the storage hooks.

        Args:
            workspace_dir (`str | Path | None`, optional):
                Root used to resolve relative artifact paths.  Pass the
                workspace's ``workdir`` when the agent runs in one.
            agent_id (`str`, optional):
                Producer id stamped on recorded entries.  Useful when
                several agents share a session (teams / sub-agents).
            user_id (`str`, optional):
                Owner id passed to *store*; required when *store* is set.
            store (`SessionArtifactStore | None`, optional):
                Persistence hook invoked after the in-memory state is
                updated.  Pass the app storage (it implements
                :meth:`StorageBase.append_session_artifacts`) to make
                artifacts survive a crashed run; omit it to rely on the
                end-of-run state persistence only.
            backend (`BackendBase | None`, optional):
                The session's backend. Used to stamp each entry's
                ``storage`` flag: ``"local"`` when the path is host
                visible (``LocalBackend``), ``"sandbox"`` otherwise so
                the frontend knows to fetch it through the workspace
                API. Defaults to ``"local"`` when omitted.
            blob_store (`ArtifactBlobStore | None`, optional):
                When set, each ``sandbox`` artifact is uploaded here and
                its URI recorded on the entry, so the file stays
                downloadable after the sandbox is gone. Upload failures
                only warn; the entry remains readable through the
                backend.
            blob_key_prefix (`str`, optional):
                Store-level prefix for upload keys.
            max_blob_bytes (`int`, optional):
                Files larger than this are not uploaded; they stay
                sandbox-only.
            tool_names (`Iterable[str] | None`, optional):
                Replace the default artifact tool set.
            extra_tool_names (`Iterable[str]`, optional):
                Additional tool names appended to the default set.
            max_entries (`int`, optional):
                Maximum number of entries retained in the state.  The
                oldest entries are dropped first.
        """
        self._workspace_dir = _workspace_base(workspace_dir) or None
        self._agent_id = (agent_id or "").strip()
        self._user_id = (user_id or "").strip()
        self._store = store
        self._storage_mode = storage_mode_for(backend)
        self._backend = backend
        self._blob_store = blob_store
        self._blob_key_prefix = blob_key_prefix
        self._max_blob_bytes = max(0, int(max_blob_bytes))
        if tool_names is None:
            names = set(DEFAULT_ARTIFACT_TOOL_NAMES)
            names.update(str(n).strip().lower() for n in extra_tool_names)
        else:
            names = {str(n).strip().lower() for n in tool_names}
        self._tool_names = frozenset(name for name in names if name)
        self._max_entries = max(1, int(max_entries))

    async def on_acting(  # type: ignore[override]
        self,
        agent: "Agent",
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator:
        """Observe a tool call and record the files it produced.

        Args:
            agent (`Agent`):
                The executing agent; its ``state`` receives the paths.
            input_kwargs (`dict`):
                Acting input kwargs (contains ``tool_call``).
            next_handler (`Callable[..., AsyncGenerator]`):
                The downstream middleware or the raw tool execution.

        Yields:
            `ToolChunk | ToolResponse`:
                Downstream items, unchanged.
        """
        tool_call = input_kwargs.get("tool_call")
        response: ToolResponse | None = None
        async for item in next_handler():
            if isinstance(item, ToolResponse):
                response = item
            yield item

        if response is not None and tool_call is not None:
            await self._record(agent, tool_call, response)

    async def _record(
        self,
        agent: "Agent",
        tool_call: "ToolCallBlock",
        response: ToolResponse,
    ) -> None:
        """Append the response's artifact paths to state and storage."""
        try:
            # Compared by value: ``ToolResultState`` is a ``StrEnum``.
            if response.state != "success":
                return
            if not is_artifact_tool_name(
                tool_call.name,
                self._tool_names,
            ):
                return
            paths = extract_artifact_paths(
                tool_name=tool_call.name,
                args=_parse_tool_input(tool_call.input),
                metadata=response.metadata,
                content=response.content,
                workspace_dir=self._workspace_dir,
                tool_names=self._tool_names,
            )
            if not paths:
                return
            state = getattr(agent, "state", None)
            if state is None:
                return
            context = state.artifact_context
            known = {entry.dedupe_key(): entry for entry in context.entries}
            added = context.add_paths(
                paths,
                agent_id=self._agent_id,
                storage=self._storage_mode,
                max_entries=self._max_entries,
            )
            # A path recorded earlier whose upload failed is retried on
            # the next write of the same file, so the blob copy can
            # appear without waiting for a new artifact.
            retry = [
                entry
                for path in paths
                if (entry := known.get(_dedupe_key(path))) is not None
                and entry.storage == "sandbox"
                and not entry.blob_uri
            ]
            if added or retry:
                await self._upload_to_blob(state, [*added, *retry])
            if added:
                await self._persist(state, added)
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "Failed to record session artifacts for tool %r",
                getattr(tool_call, "name", ""),
                exc_info=True,
            )

    async def _upload_to_blob(
        self,
        state: Any,
        added: list["SessionArtifact"],
    ) -> None:
        """Upload sandbox artifacts so they outlive the sandbox.

        Runs before :meth:`_persist`, so the recorded entries already
        carry their ``blob_uri``. A file that is missing, too large, or
        rejected by the store keeps an empty ``blob_uri`` and stays
        readable through the backend.
        """
        if self._blob_store is None or self._backend is None:
            return
        session_id = str(getattr(state, "session_id", "") or "")
        for entry in added:
            if entry.storage != "sandbox":
                continue
            size = await self._artifact_size(entry.path)
            if (
                size is not None
                and self._max_blob_bytes
                and size > self._max_blob_bytes
            ):
                logger.warning(
                    "Artifact %s is %d bytes, over the %d-byte upload "
                    "limit; leaving it in the sandbox.",
                    entry.path,
                    size,
                    self._max_blob_bytes,
                )
                continue
            try:
                data = await self._backend.read_file(entry.path)
                entry.blob_uri = await self._blob_store.write_stream(
                    blob_key(self._blob_key_prefix, session_id, entry.path),
                    io.BytesIO(data),
                )
            except Exception:  # pylint: disable=broad-except
                logger.warning(
                    "Failed to upload artifact %s to the blob store",
                    entry.path,
                    exc_info=True,
                )

    async def _artifact_size(self, path: str) -> int | None:
        """Best-effort size lookup before pulling the bytes."""
        try:
            entry = await self._backend.stat(path)  # type: ignore[union-attr]
        except Exception:  # pylint: disable=broad-except
            return None
        if entry is None or entry.is_dir:
            return None
        return entry.size_bytes

    async def _persist(
        self,
        state: Any,
        added: list["SessionArtifact"],
    ) -> None:
        """Write newly added artifacts through the configured store.

        Storage failures must never break a tool call: the in-memory
        state already carries the entries and the end-of-run
        ``update_session_state`` writes them again.
        """
        if self._store is None:
            return
        session_id = str(getattr(state, "session_id", "") or "")
        if not session_id or not self._user_id:
            return
        try:
            await self._store.append_session_artifacts(
                self._user_id,
                self._agent_id,
                session_id,
                added,
                max_entries=self._max_entries,
            )
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "Failed to persist %d session artifact(s) for %s",
                len(added),
                session_id,
                exc_info=True,
            )


__all__ = [
    "DEFAULT_ARTIFACT_TOOL_NAMES",
    "DEFAULT_MAX_BLOB_BYTES",
    "ArtifactBlobStore",
    "SessionArtifactStore",
    "SessionArtifactsMiddleware",
    "blob_key",
    "extract_artifact_paths",
    "is_artifact_tool_name",
    "normalize_artifact_path",
    "storage_mode_for",
]
