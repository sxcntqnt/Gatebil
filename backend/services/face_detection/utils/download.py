# utils/download.py
"""
Download utilities.
===================

Thin wrapper around torch.hub.download_url_to_file, which supersedes the
custom urllib implementation previously maintained in this module.

torch.hub.download_url_to_file provides:
    - Atomic writes (temp file + move, no partial files on failure)
    - Progress bar via tqdm (if installed)
    - SHA256 integrity checking
    - Automatic ~/.cache/torch/hub/checkpoints/ cache directory
    - Correct TLS CA handling across platforms and Python versions

The legacy custom implementation of download_url_to_file is retired.
New code should call torch.hub.download_url_to_file directly or use
the convenience wrapper below.
"""

from __future__ import annotations

import os
from typing import Optional

import torch


def download_url_to_file(
    url: str,
    dst: str,
    hash_prefix: Optional[str] = None,
    progress: bool = True,
) -> None:
    """
    Download a file from `url` to `dst`.

    A direct pass-through to torch.hub.download_url_to_file.
    Provided for drop-in compatibility with legacy call sites that imported
    this function from utils.download.

    Uses an atomic temp-file-then-move strategy, so a failed download never
    leaves a corrupt partial file at `dst`.

    Parameters
    ----------
    url : str
        Remote URL to download.

    dst : str
        Full local path where the file should be saved.
        Parent directory must already exist.

    hash_prefix : str or None
        If provided, the downloaded file's SHA256 digest must start with
        this prefix. Raises RuntimeError on mismatch.

    progress : bool
        If True and tqdm is installed, display a download progress bar.
        Default True.

    Examples
    --------
    >>> download_url_to_file(
    ...     'https://example.com/weights.pt',
    ...     '/tmp/weights.pt',
    ... )
    """
    torch.hub.download_url_to_file(
        url,
        dst,
        hash_prefix=hash_prefix,
        progress=progress,
    )


def get_cache_dir(subdir: str = 'checkpoints') -> str:
    """
    Return the torch hub cache directory, creating it if needed.

    Respects the TORCH_HOME and XDG_CACHE_HOME environment variables,
    matching torch.hub's own resolution logic.

    Parameters
    ----------
    subdir : str
        Subdirectory within the torch home directory. Default 'checkpoints'.

    Returns
    -------
    str
        Absolute path to the cache directory.
    """
    torch_home = os.path.expanduser(
        os.getenv(
            'TORCH_HOME',
            os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch'),
        )
    )
    cache_dir = os.path.join(torch_home, subdir)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir
