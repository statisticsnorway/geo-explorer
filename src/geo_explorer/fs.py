import datetime
import glob
import io
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import tzlocal
from fsspec.spec import AbstractFileSystem


class LocalFileSystem(AbstractFileSystem):
    """Mimicks GCS's FileSystem but using standard library (os, glob, shutil)."""

    @staticmethod
    def glob(
        path: str,
        detail: bool = False,
        recursive: bool = True,
        include_hidden: bool = True,
        **kwargs,
    ) -> list[dict] | list[str]:
        relevant_paths = glob.iglob(
            path, recursive=recursive, include_hidden=include_hidden, **kwargs
        )

        if not detail:
            return list(relevant_paths)
        with ThreadPoolExecutor() as executor:
            return list(executor.map(get_file_info, relevant_paths))

    @classmethod
    def ls(cls, path: str, detail: bool = False, **kwargs):
        return cls().glob(
            str(Path(path) / "**"), detail=detail, recursive=False, **kwargs
        )

    @staticmethod
    def info(path) -> dict[str, Any]:
        return get_file_info(path)

    @staticmethod
    def open(path: str, *args, **kwargs) -> io.TextIOWrapper:
        return open(path, *args, **kwargs)

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}()"


def get_file_info(path) -> dict[str, str | float]:
    tz = tzlocal.get_localzone()
    return {
        "updated": datetime.datetime.fromtimestamp(os.path.getmtime(path), tz=tz),
        "size": os.path.getsize(path),
        "name": path,
        "type": "directory" if os.path.isdir(path) else "file",
    }
