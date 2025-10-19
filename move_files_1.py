import time, posixpath, random

SRC_ROOT = "/Volumes/<catalog>/<schema>/<src_volume>/"
DST_ROOT = "/Volumes/<catalog>/<schema>/<dst_volume>/"
random.seed(1)

def _norm(p): return p if p.endswith("/") else p + "/"

t0=time.time()
src_entries = dbutils.fs.ls(_norm(SRC_ROOT))
t1=time.time()
print(f"ls(SRC_ROOT) -> {len(src_entries)} entries in {t1-t0:.2f}s")

files = [e for e in src_entries if not e.path.endswith("/")]
if not files:
    raise RuntimeError("No files in SRC_ROOT")

# write probe
probe_dir = posixpath.join(_norm(DST_ROOT), "__probe__")
probe_file = posixpath.join(probe_dir, "ok.txt")
try:
    dbutils.fs.mkdirs(probe_dir)
    dbutils.fs.put(probe_file, "ok", True)
    print("DST write probe: OK")
    dbutils.fs.rm(probe_dir, True)
except Exception as e:
    raise RuntimeError(f"DST write probe failed: {e}")

# single file move probe (copy+delete)
src_one = random.choice(files).path
dst_one = posixpath.join(_norm(DST_ROOT), posixpath.basename(src_one))
t0=time.time()
dbutils.fs.cp(src_one, dst_one)
dbutils.fs.rm(src_one)
t1=time.time()
print(f"Single move (cp+rm) success in {t1-t0:.2f}s: {dst_one}")

from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from time import sleep, time
from threading import Lock
import posixpath, random

SRC_ROOT = "/Volumes/<catalog>/<schema>/<src_volume>/"
DST_ROOT = "/Volumes/<catalog>/<schema>/<dst_volume>/"
TARGET_COUNT = 1000
MOVE_WORKERS = 64
MAX_RETRIES = 3
PER_FILE_TIMEOUT_SEC = 120
OVERWRITE = True
VALIDATE_SIZE = True
RANDOM_SEED = 42

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

def _norm_dir(p): return p if p.endswith("/") else p + "/"
def _parent(p): return posixpath.dirname(p.rstrip("/"))

def _exists(path):
    try: dbutils.fs.ls(path); return True
    except Exception: return False

def _mkdirs(path): dbutils.fs.mkdirs(path)

def _size_of(path):
    try:
        info = dbutils.fs.ls(path)
        if len(info)==1 and not info[0].path.endswith("/"):
            return info[0].size
    except Exception:
        pass
    return -1

def _cp(src, dst): dbutils.fs.cp(src, dst, recurse=False)
def _rm(path): dbutils.fs.rm(path, recurse=False)

def _move_cp_rm(src, dst, mkdir_cache, lock):
    dparent = _parent(dst)
    with lock:
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
        start = time()
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
            if time() - start > PER_FILE_TIMEOUT_SEC or attempt == MAX_RETRIES:
                return ("failed", dst)
            sleep(delay + random.random()*0.2)
            delay = min(delay*2, 6.0)

def move_random_flat():
    src = _norm_dir(SRC_ROOT); dst = _norm_dir(DST_ROOT)
    entries = [e for e in dbutils.fs.ls(src) if not e.path.endswith("/")]
    if not entries:
        print("No files to move"); return
    if len(entries) > TARGET_COUNT:
        entries = random.sample(entries, TARGET_COUNT)
    pairs = [(e.path, posixpath.join(dst, posixpath.basename(e.path))) for e in entries]

    mkdir_cache, lock = set([dst.rstrip("/")]), Lock()
    moved = skipped = failed = 0
    printed = 0

    with ThreadPoolExecutor(max_workers=MOVE_WORKERS) as ex:
        futs = [ex.submit(_move_cp_rm, s, d, mkdir_cache, lock) for (s, d) in pairs]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                st, p = f.result(timeout=PER_FILE_TIMEOUT_SEC + 5)
            except TimeoutError:
                st, p = ("failed", "<timeout>")
            if st == "moved": moved += 1
            elif st == "skipped_exists": skipped += 1
            else: failed += 1
            if i - printed >= 25 or i == len(futs):
                printed = i
                print(f"Progress: {i}/{len(futs)} | moved={moved} skipped={skipped} failed={failed}")

    print(f"Done: moved={moved} skipped={skipped} failed={failed}")

move_random_flat()

