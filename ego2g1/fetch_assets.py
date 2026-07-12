"""Fetch openpi's public GCS assets over plain HTTPS — no gcsfs/google-auth.

For machines whose Python env lacks the GCS client stack but has HTTPS access
to storage.googleapis.com. Files land in the exact cache layout openpi's
maybe_download uses (~/.cache/openpi/<bucket>/<path>), so tokenizer loading
and CheckpointWeightLoader find them and never attempt a download.

Stdlib only. Usage from the openpi root:
    python ego2g1/fetch_assets.py --dry-run     # list what would be fetched
    python ego2g1/fetch_assets.py               # tokenizer + pi05_base params
    python ego2g1/fetch_assets.py gs://openpi-assets/checkpoints/pi05_base/params
"""

import http.client
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

RETRYABLE = (urllib.error.URLError, ConnectionError, TimeoutError, http.client.HTTPException)


def _urlopen_retry(req, *, timeout: int, retries: int = 12):
    """urlopen with exponential backoff on transient network/TLS errors.
    A non-transient HTTPError (404/401) is re-raised immediately."""
    attempt = 0
    while True:
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError:
            raise  # a real HTTP status, not a transport hiccup — don't retry
        except RETRYABLE as e:
            attempt += 1
            if attempt > retries:
                raise
            wait = min(2**attempt, 60)
            print(f"  retry {attempt}/{retries} in {wait}s ({type(e).__name__})", flush=True)
            time.sleep(wait)

DEFAULT_ASSETS = [
    "gs://big_vision/paligemma_tokenizer.model",
    "gs://openpi-assets/checkpoints/pi05_base/params",
]
CACHE = pathlib.Path(os.getenv("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser()


def head_object(bucket: str, name: str) -> int | None:
    """Size of an exact object, or None if it doesn't exist / isn't readable.
    Needed because some buckets (big_vision) forbid anonymous LISTING while
    allowing object reads."""
    url = f"https://storage.googleapis.com/{bucket}/{urllib.parse.quote(name)}"
    req = urllib.request.Request(url, method="HEAD")
    try:
        with _urlopen_retry(req, timeout=60) as r:
            return int(r.headers["Content-Length"])
    except urllib.error.HTTPError:
        return None


def list_objects(bucket: str, prefix: str):
    """Yield (name, size) under a prefix via the public GCS JSON API."""
    base = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
    token = None
    while True:
        query = {"prefix": prefix, "fields": "items(name,size),nextPageToken", "maxResults": "1000"}
        if token:
            query["pageToken"] = token
        with _urlopen_retry(f"{base}?{urllib.parse.urlencode(query)}", timeout=60) as r:
            page = json.load(r)
        for item in page.get("items", []):
            yield item["name"], int(item["size"])
        token = page.get("nextPageToken")
        if not token:
            return


def fetch(bucket: str, name: str, size: int, retries: int = 12) -> None:
    """Download with resume (HTTP Range on the .part file) and exponential
    backoff — the route to storage.googleapis.com may reset connections
    intermittently; progress is never lost across retries or reruns."""
    dest = CACHE / bucket / name
    if dest.exists() and dest.stat().st_size == size:
        print(f"  cached  {name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://storage.googleapis.com/{bucket}/{urllib.parse.quote(name)}"
    tmp = dest.with_name(dest.name + ".part")
    if size == 0:  # zero-byte marker object (e.g. commit_success.txt)
        dest.write_bytes(b"")
        print(f"  {0.0:9.1f} / 0.0 MB  {name[-60:]}")
        return
    attempt = 0
    while True:
        done = tmp.stat().st_size if tmp.exists() else 0
        if done > size:
            tmp.unlink()
            done = 0
        if done == size:
            break
        try:
            req = urllib.request.Request(url)
            if done:
                req.add_header("Range", f"bytes={done}-")
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "ab" if done else "wb") as f:
                while chunk := r.read(1 << 22):  # 4 MiB
                    f.write(chunk)
                    done += len(chunk)
                    print(f"\r  {done / 1e6:9.1f} / {size / 1e6:.1f} MB  {name[-60:]}", end="", flush=True)
            if done == size:
                break
            raise ConnectionError(f"short read at {done}/{size} bytes")
        except urllib.error.HTTPError:
            print()
            raise
        except RETRYABLE as e:
            attempt += 1
            if attempt > retries:
                print()
                raise
            wait = min(2**attempt, 60)
            print(f"\r  retry {attempt}/{retries} in {wait}s at {done / 1e6:.1f} MB ({type(e).__name__})  ", flush=True)
            time.sleep(wait)
    print()
    if tmp.stat().st_size != size:
        raise RuntimeError(f"{name}: got {tmp.stat().st_size} bytes, expected {size}")
    os.replace(tmp, dest)


def main(argv: list[str]) -> None:
    dry_run = "--dry-run" in argv
    assets = [a for a in argv if a.startswith("gs://")] or DEFAULT_ASSETS
    for asset in assets:
        bucket, _, prefix = asset.removeprefix("gs://").partition("/")
        size = head_object(bucket, prefix)
        objects = [(prefix, size)] if size is not None else list(list_objects(bucket, prefix))
        if not objects:
            raise SystemExit(f"nothing found at {asset} — check the path")
        total = sum(s for _, s in objects)
        print(f"{asset}: {len(objects)} files, {total / 1e9:.2f} GB -> {CACHE / bucket / prefix}")
        if dry_run:
            for name, s in objects[:8]:
                print(f"    {s / 1e6:9.1f} MB  {name}")
            if len(objects) > 8:
                print(f"    ... and {len(objects) - 8} more")
            continue
        for name, s in objects:
            fetch(bucket, name, s)
    print("done" if not dry_run else "dry run only — nothing downloaded")


if __name__ == "__main__":
    main(sys.argv[1:])
