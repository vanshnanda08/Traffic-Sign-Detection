"""Explicit authorized asset retrieval; never implicit downloads from training."""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests

from p57.dataset import ROOT, save_json


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_asset(name):
    spec = json.loads((ROOT / "configs/assets_w2.json").read_text())[name]
    path = ROOT / spec["path"]
    receipt = path.with_suffix(path.suffix + ".source.json")
    if path.exists():
        if not receipt.exists() or json.loads(receipt.read_text())["sha256"] != file_sha256(path):
            raise ValueError(f"Existing asset lacks valid provenance: {path}")
        return json.loads(receipt.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for attempt in range(3):
        partial = path.with_suffix(path.suffix + ".part")
        try:
            start = time.monotonic()
            with requests.get(spec["url"], stream=True, timeout=(30, 120), headers={"User-Agent": "P57-academic-reproducibility/1"}) as response:
                response.raise_for_status()
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(1024*1024):
                        stream.write(chunk)
                expected = response.headers.get("Content-Length")
                if expected and partial.stat().st_size != int(expected):
                    raise ValueError("Download length mismatch")
                final = urlsplit(response.url)
                record = {**spec, "name": name, "sha256": file_sha256(partial), "bytes": partial.stat().st_size,
                          "retrieved_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic()-start,
                          "resolved_url_without_query": urlunsplit((final.scheme,final.netloc,final.path,"","")),
                          "http_status": response.status_code, "attempts": attempt+1}
            partial.rename(path)
            save_json(receipt, record)
            return record
        except (requests.RequestException, OSError, ValueError) as error:
            errors.append({"attempt": attempt+1, "error_type": type(error).__name__})
    save_json(path.with_suffix(path.suffix + ".failed.json"), {**spec, "status": "blocked", "attempts": errors})
    raise RuntimeError(f"Official asset retrieval failed: {name}; see {path}.failed.json")
