#!/usr/bin/env python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
"""File-system agnostic IO APIs"""

import hashlib
import os
import shutil
import tempfile

try:
    from hdfs_io import copy, exists, makedirs  # for internal use only
except ImportError:
    from .hdfs_io import copy, exists, makedirs

__all__ = ["copy", "exists", "makedirs"]

_HDFS_PREFIX = "hdfs://"


def is_non_local(path):
    return path.startswith(_HDFS_PREFIX)


def md5_encode(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()


def get_local_temp_path(hdfs_path: str, cache_dir: str) -> str:
    """Generate a unique local cache path for an HDFS resource.
    Creates a MD5-hashed subdirectory in cache_dir to avoid name conflicts,
    then returns path combining this subdirectory with the HDFS basename.

    Args:
        hdfs_path (str): Source HDFS path to be cached
        cache_dir (str): Local directory for storing cached files

    Returns:
        str: Absolute local filesystem path in format:
            {cache_dir}/{md5(hdfs_path)}/{basename(hdfs_path)}
    """
    # make a base64 encoding of hdfs_path to avoid directory conflict
    encoded_hdfs_path = md5_encode(hdfs_path)
    temp_dir = os.path.join(cache_dir, encoded_hdfs_path)
    os.makedirs(temp_dir, exist_ok=True)
    dst = os.path.join(temp_dir, os.path.basename(hdfs_path))
    return dst


def _record_directory_structure(folder_path):
    record_file = os.path.join(folder_path, ".directory_record.txt")
    with open(record_file, "w") as f:
        for root, dirs, files in os.walk(folder_path):
            for dir_name in dirs:
                relative_dir = os.path.relpath(os.path.join(root, dir_name), folder_path)
                f.write(f"dir:{relative_dir}\n")
            for file_name in files:
                if file_name != ".directory_record.txt":
                    relative_file = os.path.relpath(os.path.join(root, file_name), folder_path)
                    f.write(f"file:{relative_file}\n")
    return record_file


def _check_directory_structure(folder_path, record_file):
    if not os.path.exists(record_file):
        return False
    existing_entries = set()
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            relative_dir = os.path.relpath(os.path.join(root, dir_name), folder_path)
            existing_entries.add(f"dir:{relative_dir}")
        for file_name in files:
            if file_name != ".directory_record.txt":
                relative_file = os.path.relpath(os.path.join(root, file_name), folder_path)
                existing_entries.add(f"file:{relative_file}")
    with open(record_file) as f:
        recorded_entries = set(f.read().splitlines())
    return existing_entries == recorded_entries


def _check_copy(src: str, dest: str) -> bool:
    """Return True when dest appears to be a complete copy of src.

    For files: compares file size.
    For directories: compares file count AND total size of all files.
    """
    if os.path.isfile(src):
        return os.path.isfile(dest) and os.path.getsize(src) == os.path.getsize(dest)
    if os.path.isdir(src):
        src_info = [(f, os.path.getsize(os.path.join(r, f))) for r, _, fs in os.walk(src) for f in fs]
        dst_info = [(f, os.path.getsize(os.path.join(r, f))) for r, _, fs in os.walk(dest) for f in fs]
        if len(src_info) != len(dst_info):
            return False
        return sum(s for _, s in src_info) == sum(s for _, s in dst_info)
    return False


def _scan_shm_for_existing(shm_root: str, basename: str, src_abs: str):
    """Scan shm cache for any existing copy with matching basename and content.

    Returns the path to the existing copy if found, else None.
    """
    if not os.path.exists(shm_root):
        return None
    for entry in os.scandir(shm_root):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        candidate = os.path.join(entry.path, basename)
        if os.path.exists(candidate) and _check_copy(src_abs, candidate):
            return candidate
    return None


def copy_to_shm(src: str) -> str:
    """Copy a file or directory into /dev/shm (shared memory) for fast multi-process access.

    All workers on the same node share this RAM-backed path, eliminating repeated
    disk reads when multiple processes load the same model.

    Uses a per-model file lock so that when multiple workers start simultaneously,
    only one performs the actual copy while the others wait and then reuse it.

    Args:
        src (str): Local filesystem path to copy into shared memory.

    Returns:
        str: Path inside /dev/shm where the resource was copied.
    """
    from filelock import FileLock

    shm_root = "/dev/shm/verl-cache/"
    src_abs = os.path.abspath(os.path.normpath(src))
    basename = os.path.basename(src_abs)
    dest_dir = os.path.join(shm_root, hashlib.md5(src_abs.encode("utf-8")).hexdigest())
    dest = os.path.join(dest_dir, basename)

    # Fast path (no lock): canonical cache hit for this exact source path
    if os.path.exists(dest) and _check_copy(src_abs, dest):
        print(f"[use_shm] Reusing existing shm copy: {dest}")
        return dest

    # Fast path (no lock): dedup scan for same model under a different path string
    existing = _scan_shm_for_existing(shm_root, basename, src_abs)
    if existing is not None:
        print(f"[use_shm] Reusing existing shm copy (dedup): {existing}")
        return existing

    # Acquire a per-model lock so only one worker performs the copy;
    # others wait here and then reuse the result via the re-checks below.
    os.makedirs(shm_root, exist_ok=True)
    lock_path = os.path.join(shm_root, f".{basename}.lock")
    with FileLock(lock_path):
        # Re-check under lock: another worker may have finished copying while we waited
        if os.path.exists(dest) and _check_copy(src_abs, dest):
            print(f"[use_shm] Reusing existing shm copy (post-lock): {dest}")
            return dest
        existing = _scan_shm_for_existing(shm_root, basename, src_abs)
        if existing is not None:
            print(f"[use_shm] Reusing existing shm copy (dedup post-lock): {existing}")
            return existing

        # Still nothing — do the actual copy
        os.makedirs(dest_dir, exist_ok=True)
        print(f"[use_shm] Copying {src} → {dest}")
        if os.path.isdir(src):
            shutil.copytree(src, dest, symlinks=False, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest)
        return dest


def copy_to_local(src: str, cache_dir=None, filelock=".file.lock", verbose=False, always_recopy=False, use_shm: bool = False) -> str:
    """Copy files/directories from HDFS to local cache with validation.

    Args:
        src (str): Source path - HDFS path (hdfs://...) or local filesystem path
        cache_dir (str, optional): Local directory for cached files. Uses system tempdir if None
        filelock (str): Base name for file lock. Defaults to ".file.lock"
        verbose (bool): Enable copy operation logging. Defaults to False
        always_recopy (bool): Force fresh copy ignoring cache. Defaults to False
        use_shm (bool): After local caching, copy into /dev/shm for fast RAM-based access.
            Useful when multiple worker processes on the same node load the same model.
            Defaults to False.

    Returns:
        str: Local filesystem path to copied resource
    """
    local_path = copy_local_path_from_hdfs(src, cache_dir, filelock, verbose, always_recopy)
    if use_shm:
        return copy_to_shm(local_path)
    return local_path


def copy_local_path_from_hdfs(src: str, cache_dir=None, filelock=".file.lock", verbose=False, always_recopy=False) -> str:
    """Deprecated. Please use copy_to_local instead."""
    from filelock import FileLock

    assert src[-1] != "/", f"Make sure the last char in src is not / because it will cause error. Got {src}"

    if is_non_local(src):
        # download from hdfs to local
        if cache_dir is None:
            # get a temp folder
            cache_dir = tempfile.gettempdir()
        os.makedirs(cache_dir, exist_ok=True)
        assert os.path.exists(cache_dir)
        local_path = get_local_temp_path(src, cache_dir)
        # get a specific lock
        filelock = md5_encode(src) + ".lock"
        lock_file = os.path.join(cache_dir, filelock)
        with FileLock(lock_file=lock_file):
            if always_recopy and os.path.exists(local_path):
                if os.path.isdir(local_path):
                    shutil.rmtree(local_path, ignore_errors=True)
                else:
                    os.remove(local_path)
            if not os.path.exists(local_path):
                if verbose:
                    print(f"Copy from {src} to {local_path}")
                copy(src, local_path)
                if os.path.isdir(local_path):
                    _record_directory_structure(local_path)
            elif os.path.isdir(local_path):
                # always_recopy=False, local path exists, and it is a folder: check whether there is anything missed
                record_file = os.path.join(local_path, ".directory_record.txt")
                if not _check_directory_structure(local_path, record_file):
                    if verbose:
                        print(f"Recopy from {src} to {local_path} due to missing files or directories.")
                    shutil.rmtree(local_path, ignore_errors=True)
                    copy(src, local_path)
                    _record_directory_structure(local_path)
        return local_path
    else:
        return src
