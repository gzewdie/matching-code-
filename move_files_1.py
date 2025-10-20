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

def _cp(src, dst):
    dbutils.fs.cp(src, dst, recurse=False)

def _rm(path):
    dbutils.fs.rm(path, recurse=False)

def _move_cp_rm(src, dst, mkdir_cache, cache_lock):
    dparent = _parent_dir(dst)
    with cache_lock:
        if dparent not in mkdir_cache:
            _mkdirs(dparent)
            mkdir_cache.add(dparent)
        if _exists(dst):
            if OVERWRITE:
                try: _rm(dst)
                except Exception: pass
            else:
                return ("skipped_exists", dst)
    delay = 0.25
    for attempt in range(MAX_RETRIES + 1):
        try:
            _cp(src, dst)
            if VALIDATE_SIZE:
                ssz, dsz = _size_of(src), _size_of(dst)
                if ssz >= 0 and dsz >= 0 and ssz != dsz:
                    try: _rm(dst)
                    except Exception: pass
                    raise IOError("size_mismatch")
            _rm(src)
            return ("moved", dst)
        except Exception:
            if attempt == MAX_RETRIES:
                return ("failed", dst)
            sleep(delay + random.random() * 0.2)
            delay = min(delay * 2, 6.0)

def _sample_paths_fast(src_root, target, seed):
    src = _norm_dir(src_root)
    df = (spark.read.format("binaryFile")
          .option("recursiveFileLookup", "false")
          .load(src)
          .select("path", "length"))
    fraction = 0.002
    got = []
    for _ in range(6):
        sample = df.sample(False, fraction, seed).limit(max(1, target - len(got)))
        rows = sample.collect()
        if not rows and fraction < 0.5:
            fraction *= 2
            continue
        got.extend(rows)
        if len(got) >= target:
            break
        fraction = min(fraction * 2, 0.5)
    if len(got) < target:
        more = df.limit(max(0, target - len(got))).collect()
        got.extend(more)
    paths = [r["path"] for r in got[:target]]
    random.shuffle(paths)
    return paths

def move_random_fast():
    src = _norm_dir(SRC_ROOT); dst = _norm_dir(DST_ROOT)
    if not (src.startswith("/Volumes/") and dst.startswith("/Volumes/")):
        raise ValueError("Use /Volumes/<catalog>/<schema>/<volume>/ paths")
    paths = _sample_paths_fast(src, TARGET_COUNT, RANDOM_SEED or random.randint(1, 10_000_000))
    if not paths:
        print("No files found"); return
    _mkdirs(dst)
    mkdir_cache, cache_lock = set([dst.rstrip("/")]), Lock()
    moved = skipped = failed = 0
    failures = []
    with ThreadPoolExecutor(max_workers=MAX_MOVE_WORKERS) as ex:
        futs = [ex.submit(_move_cp_rm, p, posixpath.join(dst, posixpath.basename(p)), mkdir_cache, cache_lock) for p in paths]
        for f in as_completed(futs):
            st, path = f.result()
            if st == "moved":
                moved += 1
            elif st == "skipped_exists":
                skipped += 1
            else:
                failed += 1
                failures.append(path)
    print(f"Moved={moved} Skipped={skipped} Failed={failed}")
    if failures:
        print("Failures (up to 20):")
        for p in failures[:20]:
            print(p)

move_random_fast()

