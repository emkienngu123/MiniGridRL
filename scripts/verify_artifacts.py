"""Check report/video contracts and render all report pages for visual QA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import pypdfium2 as pdfium
from pypdf import PdfReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="submission")
    parser.add_argument("--report", help="Optional private report to validate separately; never included in public outputs")
    args = parser.parse_args()
    folder = Path(args.submission)
    report_pages = None
    if args.report:
        report = Path(args.report)
        reader = PdfReader(report)
        if not 3 <= len(reader.pages) <= 5:
            raise ValueError(f"Report requires 3-5 pages, got {len(reader.pages)}")
        if any(len(page.extract_text()) < 100 for page in reader.pages):
            raise ValueError("Report has an empty or nearly empty page")
        report_pages = len(reader.pages)
        renders = Path("tmp/private-report-preview")
        renders.mkdir(parents=True, exist_ok=True)
        for index, page in enumerate(pdfium.PdfDocument(report)):
            page.render(scale=1.4).to_pil().save(renders / f"page-{index + 1}.png")
    movie = imageio.get_reader(folder / "demo.mp4")
    try:
        frames = movie.count_frames()
        metadata = movie.get_meta_data()
        seconds = frames / metadata["fps"]
        if not 180 <= seconds <= 300:
            raise ValueError(f"Demo requires 180-300 seconds, got {seconds}")
        for index in (0, frames // 2, frames - 1):
            if movie.get_data(index).shape != (720, 1280, 3):
                raise ValueError("Demo frame dimensions are inconsistent")
    finally:
        movie.close()
    result = {"separate_report_pages": report_pages, "report_in_public_package": False,
              "demo_seconds": seconds, "demo_frames": frames, "status": "passed"}
    (folder / "artifact-checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
