from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
from threading import Lock
import posixpath, random

SRC_ROOT = "/Volumes/<catalog>/<schema>/<src_volume>/"
DST_ROOT = "/Volumes/<catalog>/<schema>/<dst_volume>/"
TARGET_COUNT = 1000
MAX_MOVE_WORKERS = 64
MAX_RETRIES = 3
OVERWRITE = True
VALIDATE_SIZE = True
RANDOM_SEED = 42

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

def _norm_dir(p):
    p = p.replace("\\", "/")
    return p if p.endswith("/") else p + "/"

def _parent_dir(p):
    return posixpath.dirname(p.rstrip("/"))

def _rel(p, root):
    root = _norm_dir(root)
    return p[len(root):] if p.startswith(root) else posixpath.basename(p)

def _mkdirs(p):
    dbutils.fs.mkdirs(p)

def _exists(path):
    try:
        dbutils.fs.ls(path)
        return True
    except Exception:
        return False

def _size_of(path):
    try:
        info = dbutils.fs.ls(path)
        if len(info) == 1 and not info[0].path.endswith("/"):
            return info[0].size
    except Exception:
        pass
    return -1

def _mv_native(src, dst):
    dbutils.fs.mv(src, dst, recurse=False)

def _cp(src, dst):
    dbutils.fs.cp(src, dst, recurse=False)

def _rm(path):
    dbutils.fs.rm(path, recurse=False)

def _move_with_retries(src, dst, mkdir_cache, cache_lock):
    dparent = _parent_dir(dst)
    with cache_lock:
        if dparent not in mkdir_cache:
            _mkdirs(dparent)
            mkdir_cache.add(dparent)
        if _exists(dst):
            if OVERWRITE:
                _rm(dst)
            else:
                return ("skipped_exists", dst)
    delay = 0.25
    for attempt in range(MAX_RETRIES + 1):
        try:
            try:
                _mv_native(src, dst)
                return ("moved", dst)
            except Exception:
                _cp(src, dst)
                if VALIDATE_SIZE:
                    ssz, dsz = _size_of(src), _size_of(dst)
                    if ssz >= 0 and dsz >= 0 and ssz != dsz:
                        try:
                            _rm(dst)
                        except Exception:
                            pass
                        raise IOError("size_mismatch")
                _rm(src)
                return ("moved", dst)
        except Exception:
            if attempt == MAX_RETRIES:
                return ("failed", dst)
            sleep(delay + random.random() * 0.2)
            delay = min(delay * 2, 6.0)

def move_random_flat():
    src = _norm_dir(SRC_ROOT)
    dst = _norm_dir(DST_ROOT)
    if not (src.startswith("/Volumes/") and dst.startswith("/Volumes/")):
        raise ValueError("Use /Volumes/<catalog>/<schema>/<volume>/ paths")
    entries = [e for e in dbutils.fs.ls(src) if not e.path.endswith("/")]
    if not entries:
        print("No files to move"); return
    if len(entries) > TARGET_COUNT:
        entries = random.sample(entries, TARGET_COUNT)
    files = [(e.path, posixpath.join(dst, _rel(e.path, src))) for e in entries]
    _mkdirs(dst)
    results = {"moved": 0, "skipped_exists": 0, "failed": 0}
    failures = []
    mkdir_cache, cache_lock = set([dst.rstrip("/")]), Lock()
    with ThreadPoolExecutor(max_workers=MAX_MOVE_WORKERS) as ex:
        futs = [ex.submit(_move_with_retries, s, d, mkdir_cache, cache_lock) for (s, d) in files]
        for f in as_completed(futs):
            st, p = f.result()
            if st == "moved":
                results["moved"] += 1
            elif st == "skipped_exists":
                results["skipped_exists"] += 1
            else:
                results["failed"] += 1
                failures.append(p)
    print(f"Moved={results['moved']} Skipped={results['skipped_exists']} Failed={results['failed']}")
    if failures:
        print("Failures (up to 20):")
        for p in failures[:20]:
            print(p)

move_random_flat()

