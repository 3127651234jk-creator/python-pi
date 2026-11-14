"""Error mapping for remote operations (port of ``packages/env/src/errors.ts``)."""

from __future__ import annotations

from typing import Optional

from ..durable.env import Context, FileError, Result, err
from .connection import RemoteError


def abort_result(context: Context, path: Optional[str] = None) -> Optional[Result]:
    if context.abort_signal is not None and context.abort_signal.is_set():
        return err(FileError("aborted", "aborted", path))
    return None


# ``NodeExecutionEnv``'s mapping of Node error codes to ``FileError`` codes.
def to_file_error(error: BaseException, fallback_path: Optional[str] = None) -> FileError:
    if isinstance(error, FileError):
        return error
    if not isinstance(error, RemoteError):
        cause = error if isinstance(error, Exception) else RuntimeError(str(error))
        return FileError("unknown", str(cause), fallback_path, cause)
    path = error.path if error.path is not None else fallback_path
    if error.code == "aborted":
        return FileError("aborted", str(error), path, error)
    if error.code == "ENOENT":
        return FileError("not_found", str(error), path, error)
    if error.code in ("EACCES", "EPERM"):
        return FileError("permission_denied", str(error), path, error)
    if error.code == "ENOTDIR":
        return FileError("not_directory", str(error), path, error)
    if error.code == "EISDIR":
        return FileError("is_directory", str(error), path, error)
    if error.code in ("EINVAL", "SYMLINK", "NOT_REGULAR"):
        return FileError("invalid", str(error), path, error)
    return FileError("unknown", str(error), path, error)
