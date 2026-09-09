#!/usr/bin/env python3
"""Download the Telecom Italia "SMS, Call, Internet - MI" files from Harvard Dataverse.

Dataset: doi:10.7910/DVN/EGZHFV (Telecom Italia Big Data Challenge 2014, Milan
grid), 62 daily tab-separated files, about 20.8 GB, released under ODbL 1.0.
The dataset carries a guestbook ("Privacy risk assessment") that requires an
email address per download; this script posts that response for each file,
follows the signed URL it returns, and verifies the Dataverse MD5. Nothing is
committed: ``data/telecom_italia_mi/`` is gitignored. The dataset record with
sizes and hashes is written later by ``scripts/run_telecom_italia_benchmark.py``.

    python scripts/fetch_telecom_italia_mi.py --email you@example.com \
        --out data/telecom_italia_mi [--days 2013-11-01 2013-11-30] [--jobs 3]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SERVER = "https://dataverse.harvard.edu"
DOI = "doi:10.7910/DVN/EGZHFV"
HEADERS = {"User-Agent": "ai-ran-kpi-forecasting/fetch_telecom_italia_mi (python urllib)"}


def dataset_files() -> list[dict]:
    url = f"{SERVER}/api/datasets/:persistentId/?persistentId={DOI}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as resp:
        data = json.load(resp)["data"]["latestVersion"]
    files = []
    for f in data["files"]:
        df = f["dataFile"]
        files.append({"id": df["id"], "filename": df["filename"], "size": df["filesize"], "md5": df["md5"]})
    return sorted(files, key=lambda f: f["filename"])


def signed_url(file_id: int, email: str) -> str:
    body = json.dumps({"guestbookResponse": {"email": email}}).encode()
    req = urllib.request.Request(
        f"{SERVER}/api/access/datafile/{file_id}", data=body, method="POST",
        headers={"Content-Type": "application/json", **HEADERS},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)["data"]["signedUrl"]


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_one(entry: dict, out: Path, email: str) -> tuple[str, str]:
    dest = out / entry["filename"]
    if dest.exists() and dest.stat().st_size == entry["size"] and md5_of(dest) == entry["md5"]:
        return entry["filename"], "present"
    url = signed_url(entry["id"], email)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=600) as resp, tmp.open("wb") as fh:
        for block in iter(lambda: resp.read(1 << 20), b""):
            fh.write(block)
    if md5_of(tmp) != entry["md5"]:
        tmp.unlink(missing_ok=True)
        return entry["filename"], "md5 mismatch"
    tmp.rename(dest)
    return entry["filename"], "downloaded"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--email", required=True, help="email for the dataset guestbook (required by the publisher)")
    ap.add_argument("--out", type=Path, default=Path("data/telecom_italia_mi"))
    ap.add_argument("--days", nargs=2, metavar=("FIRST", "LAST"), help="inclusive YYYY-MM-DD range; default all")
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    files = dataset_files()
    if args.days:
        first, last = args.days
        files = [f for f in files if first <= f["filename"][-14:-4] <= last]
    print(f"[fetch] {len(files)} files, {sum(f['size'] for f in files) / 1e9:.1f} GB -> {args.out}", flush=True)
    failures = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(fetch_one, f, args.out, args.email) for f in files]
        for fut in as_completed(futures):
            try:
                name, status = fut.result()
            except Exception as exc:  # noqa: BLE001 - report and continue; the summary line carries the count
                name, status = "?", f"error: {exc}"
            if status not in {"present", "downloaded"}:
                failures += 1
            print(f"[fetch] {name}: {status}", flush=True)
    print(f"[fetch] done, {failures} failures", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
