import json
import sys
import time
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi

VIDEOS = {
    "K5y7hhNyoU0": "Part 1 (Q1-20)",
    "NvTcIeDhmtE": "Part 2 (Q21-40)",
    "njftvsbPQ2Q": "Part 3 (Q41-60)",
    "zq8dsIhFEKk": "Part 4 (Q61-80)",
    "jY-ewNEAgfY": "Part 5 (Q81-100)",
    "c4ldYeJsAp0": "Part 6 (Q101-123)",
}

OUT_DIR = Path("transcripts")
OUT_DIR.mkdir(exist_ok=True)

api = YouTubeTranscriptApi()

results = {}
for vid, label in VIDEOS.items():
    try:
        fetched = api.fetch(vid)
        segments = [{"start": s.start, "duration": s.duration, "text": s.text} for s in fetched]
        raw_text = " ".join(s["text"].replace("\n", " ") for s in segments)
        (OUT_DIR / f"{vid}.txt").write_text(raw_text, encoding="utf-8")
        (OUT_DIR / f"{vid}.segments.json").write_text(
            json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        results[vid] = {"status": "ok", "label": label, "segments": len(segments), "chars": len(raw_text)}
        print(f"[OK] {vid} ({label}) - {len(segments)} segments, {len(raw_text)} chars")
    except Exception as e:
        results[vid] = {"status": "failed", "label": label, "error": str(e)}
        print(f"[FAIL] {vid} ({label}) - {e}")
    time.sleep(1)

Path("transcripts/_fetch_summary.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
)
print("\nDone.")
