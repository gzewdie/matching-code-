from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, as_completed
from threading import Lock
from time import sleep
import posixpath, random

SRC_ROOT = "/Volumes/<catalog>/<schema>/<src_volume>/"
DST_ROOT = "/Volumes/<catalog>/<schema>/<dst_volume>/"
TARGET_COUNT = 1000
MAX_DIR_LISTS = 8000
MAX_LIST_WORKERS = 16
MAX_COPY_WORKERS = 64
MAX_RETRIES = 3
OVERWRITE = True
VALIDATE_SIZE = False
RANDOM_SEED = 42

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

def _norm_dir(p):
    p = p.replace("\\", "/")
    return p if p.endswith("/") else p + "/"

def _rel(src_full, root):
    root = _norm_dir(root)
    return src_full[len(root):] if src_full.startswith(root) else src_full.split("/")[-1]

def _parent_dir(p):
    return posixpath.dirname(p.rstrip("/"))

def _dst_for(rel):
    return posixpath.join(_norm_dir(DST_ROOT), rel)

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
        return -1
    except Exception:
        return -1

def _copy_once(src, dst):
    dbutils.fs.cp(src, dst, recurse=False)

def _copy_with_retries(src, dst, mkdir_cache, cache_lock):
    dparent = _parent_dir(dst)
    with cache_lock:
        if dparent not in mkdir_cache:
            _mkdirs(dparent)
            mkdir_cache.add(dparent)
    if not OVERWRITE and _exists(dst):
        if VALIDATE_SIZE:
            ssz, dsz = _size_of(src), _size_of(dst)
            if ssz >= 0 and dsz >= 0 and ssz == dsz:
                return ("skipped_exists", dst)
        else:
            return ("skipped_exists", dst)
    delay = 0.25
    for attempt in range(MAX_RETRIES + 1):
        try:
            _copy_once(src, dst)
            if VALIDATE_SIZE:
                ssz, dsz = _size_of(src), _size_of(dst)
                if ssz >= 0 and dsz >= 0 and ssz != dsz:
                    try:
                        dbutils.fs.rm(dst)
                    except Exception:
                        pass
                    raise IOError("size_mismatch")
            return ("copied", dst)
        except Exception:
            if attempt == MAX_RETRIES:
                return ("failed", dst)
            sleep(delay + random.random() * 0.2)
            delay = min(delay * 2, 6.0)

def gather_random_files():
    root = _norm_dir(SRC_ROOT)
    if not (root.startswith("/Volumes/") and _norm_dir(DST_ROOT).startswith("/Volumes/")):
        raise ValueError("Use /Volumes/<catalog>/<schema>/<volume>/ paths")
    dirs = [root]
    files = []
    dir_lists = 0
    inflight = {}

    def submit_one(ex):
        nonlocal dir_lists
        if not dirs or dir_lists >= MAX_DIR_LISTS:
            return
        i = random.randrange(len(dirs))
        d = dirs.pop(i)
        fut = ex.submit(dbutils.fs.ls, d)
        inflight[fut] = d
        dir_lists += 1

    with ThreadPoolExecutor(max_workers=MAX_LIST_WORKERS) as ex:
        for _ in range(min(MAX_LIST_WORKERS, len(dirs))):
            submit_one(ex)
        while inflight and len(files) < TARGET_COUNT:
            done, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                inflight.pop(fut, None)
                try:
                    entries = fut.result()
                except Exception:
                    entries = []
                random.shuffle(entries)
                for info in entries:
                    if len(files) >= TARGET_COUNT:
                        break
                    if info.path.endswith("/"):
                        dirs.append(info.path)
                    else:
                        rel = _rel(info.path, root)
                        files.append((info.path, rel))
                if len(files) >= TARGET_COUNT:
                    break
                if dirs and dir_lists < MAX_DIR_LISTS:
                    submit_one(ex)
            if not inflight and dirs and len(files) < TARGET_COUNT and dir_lists < MAX_DIR_LISTS:
                for _ in range(min(MAX_LIST_WORKERS, len(dirs))):
                    submit_one(ex)
            if not inflight and not dirs:
                break
    return files[:TARGET_COUNT]

def copy_random_files():
    candidates = gather_random_files()
    if not candidates:
        print("No files selected")
        return
    print(f"Selected {len(candidates)} files")
    mkdir_cache = set()
    cache_lock = Lock()
    results = {"copied": 0, "skipped_exists": 0, "failed": 0}
    failures = []

    with ThreadPoolExecutor(max_workers=MAX_COPY_WORKERS) as ex:
        futs = [ex.submit(_copy_with_retries, src, _dst_for(rel), mkdir_cache, cache_lock) for (src, rel) in candidates]
        for f in as_completed(futs):
            status, path = f.result()
            if status == "copied":
                results["copied"] += 1
            elif status == "skipped_exists":
                results["skipped_exists"] += 1
            else:
                results["failed"] += 1
                failures.append(path)

    print(f"Copied={results['copied']} Skipped={results['skipped_exists']} Failed={results['failed']}")
    if failures:
        print("Failures (up to 20):")
        for p in failures[:20]:
            print(p)

copy_random_files()

