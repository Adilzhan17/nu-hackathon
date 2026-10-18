"""Import the organizer's XLSX into a human review CSV and a video inventory.

Run: .venv/bin/python ml/import_open_labels.py --source data/raw/open --output data/review
Requires PyAV. Does not infer event types or convert point labels into interval truth.
Exact duplicate videos receive the same group_id; related takes need human grouping.
"""

import argparse
import csv
import hashlib
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path):
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = [
                "".join(t.text or "" for t in item.findall(".//s:t", ns))
                for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("s:si", ns)
            ]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        for row in sheet.findall(".//s:sheetData/s:row", ns):
            if row.get("r") == "1":
                continue
            values = {}
            for cell in row.findall("s:c", ns):
                value = cell.find("s:v", ns)
                text = value.text if value is not None else ""
                if cell.get("t") == "s":
                    text = strings[int(text)]
                elif cell.get("t") == "inlineStr":
                    text = "".join(t.text or "" for t in cell.findall(".//s:t", ns))
                values[re.sub(r"\d", "", cell.get("r"))] = text
            yield values


def parse_anchor(text):
    if not text.strip():
        return None
    minutes = re.search(r"(\d+)\s*мин", text)
    seconds = re.search(r"(\d+)\s*сек", text)
    if not minutes and not seconds:
        raise ValueError(f"Unrecognized timestamp: {text!r}")
    return (int(minutes[1]) * 60 if minutes else 0) + (int(seconds[1]) if seconds else 0)


def main():
    import av

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "event_review.csv"
    inventory_path = args.output / "inventory.json"
    if csv_path.exists() or inventory_path.exists():
        parser.error("Output exists; choose a new directory to preserve human annotations")
    videos, groups, labels, review, warnings = {}, defaultdict(list), Counter(), [], []
    for path in sorted((args.source / "videos").glob("*.mov")):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            duration = (float(stream.duration * stream.time_base) if stream.duration is not None
                        else float(container.duration / av.time_base))
            # Decode the first frame to catch unsupported/broken files early.
            next(container.decode(stream))
            videos[path.stem] = {
                "video": "videos/" + path.name, "sha256": digest,
                "group_id": "video_" + digest[:16], "duration_sec": duration,
                "width": stream.width, "height": stream.height,
                "fps": float(stream.average_rate) if stream.average_rate else None,
                "codec": stream.codec_context.name,
            }
        groups[digest].append(path.name)
    for row in read_rows(args.source / "open_labels.xlsx"):
        video = videos[row["A"].strip()]
        verdict = row["C"].strip()
        if verdict not in {"нарушение", "без нарушения"}:
            raise ValueError(f"Unexpected verdict: {verdict!r}")
        anchor = parse_anchor(row.get("B", ""))
        if verdict == "нарушение" and anchor is None:
            raise ValueError(f"Missing event timestamp: {video['video']}")
        if anchor is not None and not 0 <= anchor <= video["duration_sec"]:
            warnings.append(f"Timestamp outside video: {video['video']} at {anchor}s")
        labels[verdict] += 1
        review.append({
            "video": video["video"], "group_id": video["group_id"],
            "split": "", "start_sec": "", "end_sec": "",
            "label": "normal" if verdict == "без нарушения" else "",
            "anchor_sec": anchor if anchor is not None else "",
            "source_verdict": verdict, "source_timestamp": row.get("B", ""),
            "duration_sec": round(video["duration_sec"], 3),
            "review_status": "pending", "observation": "", "rule_id": "",
        })
    if len(review) != len(videos):
        warnings.append("Source row count differs from video count; verify annotation coverage")
    unique = {video["sha256"]: video for video in videos.values()}
    report = {
        "source_video_count": len(videos), "unique_video_count": len(unique),
        "source_label_counts": dict(labels),
        "duplicate_groups": [names for names in groups.values() if len(names) > 1],
        "unique_duration_minutes": round(sum(v["duration_sec"] for v in unique.values()) / 60, 2),
        "warnings": warnings,
        "limitations": [
            "Only first frame decoded; full video decoding not checked",
            "group_id only groups byte-identical files; human review must group related takes",
            "Point annotations are source labels, not verified event interval boundaries",
            "event_review.csv is a work queue, not a training-ready manifest",
        ],
        "videos": list(videos.values()),
    }
    with csv_path.open("x", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)
    with inventory_path.open("x", encoding="utf-8") as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "videos"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
