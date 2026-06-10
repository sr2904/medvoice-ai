from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 4:
        print(
            "Usage: python scripts/prepare_common_voice_manifest.py "
            "<csv_or_tsv_path> <audio_root> <output_jsonl> [--limit N]"
        )
        return 1

    annotations_path = Path(sys.argv[1]).expanduser().resolve()
    audio_root = Path(sys.argv[2]).expanduser().resolve()
    output_path = Path(sys.argv[3]).expanduser().resolve()

    limit = None
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        if idx + 1 < len(sys.argv):
            limit = int(sys.argv[idx + 1])

    if not annotations_path.exists():
        print(f"Annotation file not found: {annotations_path}")
        return 1

    if not audio_root.exists():
        print(f"Audio root not found: {audio_root}")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    delimiter = "\t" if annotations_path.suffix.lower() == ".tsv" else ","

    rows_written = 0

    with annotations_path.open("r", encoding="utf-8", newline="") as handle, output_path.open(
        "w", encoding="utf-8"
    ) as out:
        reader = csv.DictReader(handle, delimiter=delimiter)

        for row in reader:
            audio_rel = (
                row.get("filename")
                or row.get("path")
                or row.get("file")
                or row.get("audio")
            )
            transcript = (
                row.get("text")
                or row.get("sentence")
                or row.get("transcript")
            )

            if not audio_rel or not transcript:
                continue

            audio_path = (audio_root / audio_rel).resolve()

            if not audio_path.exists():
                # fallback: use only basename if needed
                audio_path = (audio_root / Path(audio_rel).name).resolve()

            if not audio_path.exists():
                continue

            payload = {
                "audio_path": str(audio_path),
                "reference_text": transcript.strip(),
            }
            out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            rows_written += 1

            if limit is not None and rows_written >= limit:
                break

    print(f"Wrote {rows_written} benchmark rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())