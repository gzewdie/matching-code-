# Databricks notebook Python 3.10+
# Copy up to N files between Unity Catalog Volumes with parallel I/O, retries, and basic validation.

from concurrent.futures import ThreadPoolExecutor, as_completed
from fnmatch import fnmatch
from time import sleep
from typing import Iterable, List, Tuple
import os
import posixpath
import random

# ---------- CONFIG ----------
SRC_ROOT = "/Volumes/<catalog>/<schema>/<src_volume>/"    # trailing slash ok
DST_ROOT = "/Volumes/<catalog>/<schema>/<dst_volume>/"    # trailing slash ok
INCLUDE_GLOB = "**/*"   # use e.g. "*.parquet" or "**/*.csv"
EXCLUDE_GLOBS = []      # e.g. ["**/_delta_log/**", "**/.*/**"]
N_LIMIT = 1000
OVERWRITE = False
MAX_WORKERS = 32        # driver-side parallelism
MAX_RETRIES = 5
# ----------------------------

def _norm(p: str) -> str:
    p = p.replace("\\", "/")
    return p if p.endswith("/") else p

def _is_dir(info) -> bool:
    # dbutils.fs.ls returns FileInfo-like with .path and .size (-1 for dirs)
    return info.path.endswith("/")

def _walk_dir(root: str) -> Iterable[Tuple[str, int]]:
    # Depth-first traversal using dbutils.fs.ls
    stack = [root if root.endswith("/") else root + "/"]
    seen = set()
    while stack:
        d = stack.pop()
        if d in seen: 
            continue
        seen.add(d)
        for info in dbutils.fs.ls(d):
            if _is_dir(info):
                stack.append(info.path)
            else:
                yield (info.path, info.size)

def _rel_path(full_path: str, root: str) -> str:
    # Produce relative path under root
    if not root.endswith("/"): root += "/"
    if full_path.startswith(root):
        return full_path[len(root):]
    # Fallback: best-effort
    return full_path.split("/")[-1]

def _match_any(globs: List[str], rel: str) -> bool:
    return any(fnmatch(rel, g) for g in globs)

def _include(rel: str) -> bool:
    inc = _match_any([INCLUDE_GLOB], rel) if INCLUDE_GLOB else True
    exc = _match_any(EXCLUDE_GLOBS, rel) if EXCLUDE_GLOBS else False
    return inc and not exc

def _dst_path(dst_root: str, rel: str) -> str:
    dst_root = dst_root if dst_root.endswith("/") else dst_root + "/"
    return posixpath.join(dst_root, rel)

def _parent_dir(p: str) -> str:
    return posixpath.dirname(p)

def _exists(path: str) -> bool:
    try:
        dbutils.fs.ls(path)
        return True
    except Exception:
        return False

def _size_of(path: str) -> int:
    try:
        info = dbutils.fs.ls(path)
        if len(info) == 1 and not _is_dir(info[0]):
            return info[0].size
        # If it listed a directory, return -1
        return -1
    except Exception:
        return -1

def _mkdirs(path: str):
    dbutils.fs.mkdirs(path)

def _cp_once(src: str, dst: str, overwrite: bool) -> None:
    # dbutils.fs.cp handles single file copy; not recursive here
    dbutils.fs.cp(src, dst, recurse=False)
    if not overwrite:
        # Extra guard: if source updated during copy, sizes must match
        s1 = _size_of(src)
        d1 = _size_of(dst)
        if s1 >= 0 and d1 >= 0 and s1 != d1:
            # Remove the bad copy to avoid partials
            try:
                dbutils.fs.rm(dst)
            except Exception:
                pass
            raise IOError(f"Size mismatch after copy: {src} ({s1}) -> {dst} ({d1})")

def _copy_with_retries(src: str, dst: str, overwrite: bool, retries: int) -> Tuple[str, bool, str]:
    # Returns (path, success, message)
    # Skip if exists and not overwriting and size matches
    if _exists(dst) and not overwrite:
        ssz = _size_of(src)
        dsz = _size_of(dst)
        if ssz >= 0 and ssz == dsz:
            return (dst, True, "skipped_exists")
    # Ensure parent
    _mkdirs(_parent_dir(dst))
    delay = 0.5
    for attempt in range(retries + 1):
        try:
            _cp_once(src, dst, overwrite=overwrite)
            return (dst, True, "copied")
        except Exception as e:
            if attempt == retries:
                return (dst, False, f"error: {type(e).__name__}: {e}")
            # Jittered backoff
            sleep(delay + random.random() * 0.25)
            delay = min(delay * 2, 10.0)
    return (dst, False, "unknown")

def list_candidate_files(src_root: str) -> List[Tuple[str, int, str]]:
    src_root = _norm(src_root)
    files = []
    for path, size in _walk_dir(src_root):
        rel = _rel_path(path, src_root)
        if _include(rel):
            files.append((path, size, rel))
            if len(files) >= N_LIMIT:
                break
    return files

def copy_files(src_root: str, dst_root: str) -> None:
    src_root = _norm(src_root)
    dst_root = _norm(dst_root)
    if not src_root.startswith("/Volumes/") or not dst_root.startswith("/Volumes/"):
        raise ValueError("Use Unity Catalog Volume paths under /Volumes/<catalog>/<schema>/<volume>/")
    candidates = list_candidate_files(src_root)
    if not candidates:
        print("No files matched.")
        return
    print(f"Planning to copy {len(candidates)} files (limit={N_LIMIT}).")

    results = {"copied": 0, "skipped_exists": 0, "failed": 0}
    failures = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = []
        for src, _, rel in candidates:
            dst = _dst_path(dst_root, rel)
            futs.append(ex.submit(_copy_with_retries, src, dst, OVERWRITE, MAX_RETRIES))
        for f in as_completed(futs):
            dst, ok, msg = f.result()
            if ok and msg == "copied":
                results["copied"] += 1
            elif ok and msg == "skipped_exists":
                results["skipped_exists"] += 1
            else:
                results["failed"] += 1
                failures.append((dst, msg))

    print(f"Done. Copied={results['copied']}, Skipped={results['skipped_exists']}, Failed={results['failed']}")
    if failures:
        print("Failures (sample up to 20):")
        for p, m in failures[:20]:
            print(f"- {p}: {m}")

# ---- RUN ----
# Set the config above, then execute:
copy_files(SRC_ROOT, DST_ROOT)

