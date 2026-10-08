"""Merge sweep shards into one results file.

Reads shard outputs from bench/fusion.py: either JSON files (--out of each
shard) or log files containing its TENSIXFUSE_RESULTS line (pod logs from the
Kubernetes run). Fails if a configuration is missing or didn't match the model.

    python bench/merge.py shards/*.json --out results/fusion.json
    python bench/merge.py logs/*.log --expect 48
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TAG = "TENSIXFUSE_RESULTS "


def load(path: Path) -> tuple[list[dict], float | None]:
    text = path.read_text()
    for line in text.splitlines():
        if line.startswith(TAG):
            payload = json.loads(line[len(TAG) :])
            return payload["records"], payload["seconds"]
    return json.loads(text), None


def key(rec: dict) -> tuple:
    return (rec["M"], tuple(rec["block"]), tuple(rec["grid"]), rec["mcast"], rec["variant"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", default="results/fusion.json")
    p.add_argument("--expect", type=int, default=None, help="number of configurations expected")
    args = p.parse_args()

    records, shard_seconds = [], []
    for name in args.inputs:
        recs, secs = load(Path(name))
        records.extend(recs)
        if secs is not None:
            shard_seconds.append(secs)
    records.sort(key=key)

    seen = [key(r) for r in records]
    dupes = len(seen) - len(set(seen))
    bad = [r for r in records if not r["matches_model"]]
    for r in records:
        r.pop("sim_warnings", None)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(records, indent=1))

    summary = {
        "shards": len(args.inputs),
        "runs": len(records),
        "duplicates": dupes,
        "model_mismatches": len(bad),
    }
    if shard_seconds:
        summary["slowest_shard_s"] = max(shard_seconds)
        summary["all_shards_s"] = round(sum(shard_seconds), 1)
    print(json.dumps(summary))
    ok = not bad and not dupes and (args.expect is None or len(records) == args.expect)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
