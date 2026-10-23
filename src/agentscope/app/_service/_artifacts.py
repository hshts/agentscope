# -*- coding: utf-8 -*-
"""Artifact blob lookups for downloads.

Sandbox artifacts are uploaded to the application blob store as they are
recorded (see :class:`~agentscope.middleware.SessionArtifactsMiddleware`).
This service is the read side: it finds the uploaded copy for a session
artifact path, so ``GET /workspace/files`` can serve the file after the
sandbox — or the workspace itself — is gone.

Kept apart from :class:`~agentscope.app._service.WorkspaceService` on
purpose: that service owns workspace lifecycle, uploads and git, while
artifact addressing only needs the session record and the blob store.
"""
from typing import AsyncIterator

from ..rag.blob_store import BlobStoreBase
from ..storage import StorageBase
from ..._logging import logger

#: Read size for a streamed download; matches the workspace endpoint.
_CHUNK_SIZE = 64 * 1024


class ArtifactBlobService:
    """Resolve uploaded artifact copies and stream their bytes.

    Args:
        storage (`StorageBase`):
            Application storage, holding the session records whose
            ``artifact_context`` names the uploaded copies.
        blob_store (`BlobStoreBase | None`, optional):
            Where uploads landed. ``None`` disables lookups entirely,
            which is the default when the app has no blob store.
    """

    def __init__(
        self,
        storage: StorageBase,
        blob_store: BlobStoreBase | None = None,
    ) -> None:
        """Bind the storage and the optional blob store."""
        self._storage = storage
        self._blob_store = blob_store

    @property
    def blob_store(self) -> BlobStoreBase | None:
        """The store uploads landed in, or ``None`` when disabled."""
        return self._blob_store

    async def blob_uri(
        self,
        user_id: str,
        agent_id: str,
        session_id: str,
        target: str = "",
        raw_path: str = "",
    ) -> str:
        """Return the blob URI recorded for a sandbox artifact path.

        Only ``storage == "sandbox"`` entries carry an uploaded copy;
        local entries are host-visible and are served from the workspace
        backend. Matching accepts either the backend-resolved path or the
        path the caller supplied, so a lookup can happen before the
        workspace is resolved (it may already have been torn down).

        Args:
            user_id (`str`):
                The authenticated owner.
            agent_id (`str`):
                The agent owning the session.
            session_id (`str`):
                The session whose artifacts to search.
            target (`str`, optional):
                The backend-resolved absolute path, when known.
            raw_path (`str`, optional):
                The path exactly as the request supplied it.

        Returns:
            `str`: The blob URI, or ``""`` when there is no uploaded copy.
        """
        if self._blob_store is None:
            return ""
        try:
            record = await self._storage.get_session(
                user_id,
                agent_id,
                session_id,
            )
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "Failed to load session %s while resolving an artifact blob",
                session_id,
                exc_info=True,
            )
            return ""
        if record is None:
            return ""
        wanted = {target, raw_path} - {""}
        for entry in record.state.artifact_context.entries:
            if entry.storage != "sandbox" or not entry.blob_uri:
                continue
            if entry.path in wanted:
                return entry.blob_uri
        return ""

    async def stream(self, uri: str) -> AsyncIterator[bytes]:
        """Yield an uploaded artifact's bytes for a streaming response.

        Args:
            uri (`str`):
                A blob URI previously returned by :meth:`blob_uri`.

        Yields:
            `bytes`: Successive chunks of the file.
        """
        if self._blob_store is None:
            raise RuntimeError("No blob store configured.")
        async with self._blob_store.open(uri) as stream:
            while chunk := await stream.read(_CHUNK_SIZE):
                yield chunk

    async def exists(self, uri: str) -> bool:
        """Whether *uri* is still present in the blob store."""
        if self._blob_store is None:
            return False
        return await self._blob_store.exists(uri)

    async def size(self, uri: str) -> int | None:
        """Return the stored byte length, when the backend can measure it."""
        if self._blob_store is None:
            return None
        return await self._blob_store.size(uri)


__all__ = ["ArtifactBlobService"]
