"""Download only the approved public SQaLe sources, at immutable revisions.

No authentication, model loading, uploads, jobs or APIs that charge are used.
Run explicitly: .venv/Scripts/python.exe scripts/download_sqale.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sqale" / "sources"
SOURCES = {
    "queries": {
        "repo": "trl-lab/SQaLe-2-text-to-SQL-Queries",
        "revision": "85fc905c3484f9d197ee8736d672a7d0e4903fc6",
        "files": {
            "README.md": "535572ce48f74c7465f96fe63fe4744a1383ad14809e226dc5804b91d5de7e67",
            "data/train-00000-of-00001.parquet": "6486484255b7d40ec400996a12047002b8371cfa9eb93b947c724132978f5d95",
            "data/test-00000-of-00001.parquet": "f083b28f62dd414f1c6ac55c0561af28a5acb6fcd48738a5459e524467420d59",
        },
    },
    "schemas": {
        "repo": "trl-lab/SQaLe-2-text-to-SQL-Schemas",
        "revision": "091b1a9e946dec386470c7dc381af166d2b41c38",
        "files": {
            "README.md": "fa2380e7d5dabf6c720bcb42e9718bb3f027e29eae6f58ed98dba1168b1fba4a",
            # One of sixteen train shards first; more only if preparation needs them.
            "data/train-00000-of-00016.parquet": "97ed9242eef62efa5a87f63af86f574c3003d788f9408ffc7504a222ae7a2881",
            "data/test-00000-of-00001.parquet": "dc5b8fff62b0d37e6f502e91908619846b264b1e66b962a70b698466ee4a6ac2",
        },
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name, source in SOURCES.items():
        for filename, expected in source["files"].items():
            target = OUT / name / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and (expected is None or sha256(target) == expected):
                print(f"cached {name}/{filename}", flush=True)
            else:
                partial = target.with_suffix(target.suffix + ".partial")
                offset = partial.stat().st_size if partial.exists() else 0
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{filename}"
                with requests.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
                    response.raise_for_status()
                    mode = "ab" if offset and response.status_code == 206 else "wb"
                    print(f"downloading {name}/{filename}", flush=True)
                    with partial.open(mode) as f:
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            f.write(chunk)
                if expected is not None and sha256(partial) != expected:
                    raise ValueError(f"hash mismatch for {name}/{filename}; partial retained")
                partial.replace(target)
            records.append({"source": name, "repo": source["repo"],
                            "revision": source["revision"], "file": filename,
                            "bytes": target.stat().st_size, "sha256": sha256(target)})
            (OUT / "downloads.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(records), "bytes": sum(r["bytes"] for r in records)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (requests.RequestException, OSError, ValueError) as error:
        # Do not print redirect URLs: CDN query strings can contain credentials.
        print(f"Download stopped: {type(error).__name__}; completed files and partials retained.", flush=True)
        raise SystemExit(1) from None
