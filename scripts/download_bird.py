"""Download the approved, content-pinned Mini-Dev SQLite sources; no model or upload."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def get(url, **kwargs):
    try:
        response = requests.get(url, timeout=(30, 120), **kwargs)
        if response.status_code != 200:
            raise RuntimeError("download HTTP status " + str(response.status_code))
        return response
    except requests.RequestException as error:
        # Redirect URLs may carry short-lived registry credentials; never log them.
        raise RuntimeError("download failed: " + type(error).__name__) from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/bird")
    args = parser.parse_args()
    out = Path(args.out)
    pins = json.loads(Path("experiments/bird-final/SOURCES.json").read_text())
    source = out / "source"
    source.mkdir(parents=True, exist_ok=True)
    for filename in ("README.md", "data/manifest.json", pins["annotations_file"],
                     "baseline_inference/README.md", "baseline_inference/prompt_template.txt",
                     "baseline_inference/validate_baseline_inference.py"):
        hf_hub_download(pins["annotations_repo"], filename, revision=pins["annotations_revision"],
                        repo_type="dataset", token=False, local_dir=source)
    if sha(source / pins["annotations_file"]) != pins["annotations_sha256"]:
        raise ValueError("BIRD annotations checksum mismatch")
    licence = source / "CC-BY-SA-4.0.txt"
    if not licence.exists():
        response = get("https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt")
        licence.write_bytes(response.content)
    registry = "https://registry-1.docker.io/v2/" + pins["database_image_repository"]
    response = get("https://auth.docker.io/token", params={"service": "registry.docker.io",
                   "scope": "repository:" + pins["database_image_repository"] + ":pull"})
    headers = {"Authorization": "Bearer " + response.json()["token"],
               "Accept": "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"}
    layers, databases = source / "layers", out / "databases"
    layers.mkdir(exist_ok=True)
    databases.mkdir(exist_ok=True)
    records = []
    for item in pins["databases"]:
        manifest = get(registry + "/manifests/" + item["manifest_digest"], headers=headers)
        if "sha256:" + hashlib.sha256(manifest.content).hexdigest() != item["manifest_digest"]:
            raise ValueError("image manifest digest mismatch")
        (source / (item["db_id"] + ".image.json")).write_bytes(manifest.content)
        layer = item["layer_digest"].split(":")[1]
        archive = layers / (layer + ".tar.gz")
        if not archive.exists():
            partial = archive.with_suffix(".partial")
            with get(registry + "/blobs/" + item["layer_digest"], headers=headers, stream=True) as response:
                count, last = 0, 0
                with partial.open("wb") as target:
                    for block in response.iter_content(4 * 1024 * 1024):
                        target.write(block)
                        count += len(block)
                        if count - last >= 32 * 1024 * 1024:
                            print(json.dumps({"downloading_db": item["db_id"], "bytes": count}), flush=True)
                            last = count
            if partial.stat().st_size != item["layer_bytes"] or sha(partial) != layer:
                raise ValueError("database layer digest/size mismatch")
            partial.rename(archive)
        if archive.stat().st_size != item["layer_bytes"] or sha(archive) != layer:
            raise ValueError("cached database layer mismatch")
        with tarfile.open(archive, "r:gz") as bundle:
            members = [m for m in bundle.getmembers() if m.isfile() and m.name.endswith(".sqlite")]
            if len(members) != 1 or members[0].size > 20 * 1024 ** 3:
                raise ValueError("expected one bounded regular SQLite database in the pinned data layer")
            member = members[0]
            db = databases / (item["db_id"] + ".sqlite")
            with bundle.extractfile(member) as incoming, db.open("wb") as target:
                shutil.copyfileobj(incoming, target, length=4 * 1024 * 1024)
        with db.open("rb") as stream:
            if stream.read(16) != b"SQLite format 3\0":
                raise ValueError("invalid SQLite header")
        record = {**item, "db_path": db.as_posix(), "db_sha256": sha(db),
                  "db_bytes": db.stat().st_size, "archive_member": member.name}
        records.append(record)
        print(json.dumps({"downloaded_db": item["db_id"], "bytes": record["db_bytes"],
                          "sha256": record["db_sha256"]}), flush=True)
    manifest = {"pins": pins, "databases": records, "licence_sha256": sha(licence),
                "annotations_sha256": sha(source / pins["annotations_file"]),
                "paid_runs": 0, "uploads": 0, "training_or_model_runs": 0}
    (out / "downloads.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"databases": len(records), "total_db_bytes": sum(r["db_bytes"] for r in records),
                      "license": pins["license"], "annotations_revision": pins["annotations_revision"]}), flush=True)


if __name__ == "__main__":
    main()
