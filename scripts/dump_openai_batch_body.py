"""One-shot: download an OpenAI batch result and dump the first N response bodies.

Used to reverse-engineer the actual ``output[]`` shape for the GA ``web_search``
tool — specifically to find where retrieved URLs live (annotations, web_search_call
sources, etc.) so the adapter parser can be fixed.

Usage:
    python scripts/dump_openai_batch_body.py <batch_id> [<n>] [<out_path>]

Reads OPENAI_API_KEY from the env. Writes a pretty-printed JSON file with the
first N succeeded rows' ``custom_id`` + ``body`` to <out_path>.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from truthbot.verify.batch import _openai_results


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    batch_id = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("scratch_openai_batch_body.json")

    rows = _openai_results(batch_id)
    succeeded = [r for r in rows if r.get("status") == "succeeded"]
    print(f"Total rows: {len(rows)}; succeeded: {len(succeeded)}", file=sys.stderr)

    sample = []
    for r in succeeded[:n]:
        body = r.get("body") or {}
        sample.append({"custom_id": r.get("custom_id"), "body": body})

    out_path.write_text(json.dumps(sample, indent=2, default=str))
    print(f"Wrote {len(sample)} rows to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
