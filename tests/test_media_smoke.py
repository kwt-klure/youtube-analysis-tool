from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from youtube_analysis_tool import pipeline as p
from youtube_analysis_tool.bundle_check import check_bundle


class LocalMediaSmokeTests(unittest.TestCase):
    def test_visual_off_reuse_and_keep_with_real_local_media(self):
        ffmpeg, env = p.resolve_command("ffmpeg")
        ffprobe, _ = p.resolve_command("ffprobe")
        if not ffmpeg or not ffprobe or not importlib.util.find_spec("cv2"):
            self.skipTest("requires ffmpeg, ffprobe and OpenCV")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "synthetic.mp4"
            subprocess.run(
                [ffmpeg, "-v", "error", "-f", "lavfi", "-i",
                 "testsrc=size=320x180:rate=2:duration=3", "-c:v", "mpeg4", str(source)],
                env=env, check=True, capture_output=True, timeout=30,
            )
            reuse = root / "transcript.json"
            reuse.write_text(json.dumps({"source": "external", "text": "Synthetic visual evidence",
                                         "segments": [{"start": 0, "end": 3, "text": "Synthetic visual evidence"}]}))
            with mock.patch.object(p, "transcribe_with_whisper", side_effect=AssertionError("ASR invoked")), \
                    mock.patch.object(p, "transcribe_with_openai_api", side_effect=AssertionError("API invoked")):
                for name, kwargs in [("off", {"transcript_mode": "off"}), ("reuse", {"reuse_transcript": reuse})]:
                    output = root / name
                    p.analyze_source(str(source), out_dir=output, visuals_mode="on",
                                     keyframe_mode="interval", interval_seconds=1,
                                     max_frames=2, ocr_mode="off", artifacts_mode="debug", **kwargs)
                    bundle = json.loads((output / "output.json").read_text())
                    self.assertTrue(check_bundle(output / "output.json")["valid"])
                    self.assertTrue((output / "visuals" / "manifest.json").is_file())
                    self.assertTrue((output / "visuals" / "selection.json").is_file())
                    self.assertGreater(bundle["visual_sampling"]["selected_frame_count"], 0)
                    self.assertLessEqual(bundle["visual_sampling"]["selected_frame_count"], 2)
                    for directory in ("video", "audio", "keyframes"):
                        self.assertFalse((output / directory).exists())
                kept = root / "kept"
                p.analyze_source(str(source), out_dir=kept, transcript_mode="off",
                                 cleanup_intermediates=False)
                self.assertTrue((kept / "video").is_dir())
                self.assertFalse(json.loads((kept / "output.json").read_text())["processing"]["cleanup_applied"])
            self.assertTrue(source.exists())


if __name__ == "__main__":
    unittest.main()
