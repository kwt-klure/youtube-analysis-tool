from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from youtube_analysis_tool import contact_sheet


class ContactSheetGridTests(unittest.TestCase):
    def test_auto_grid_keeps_short_vertical_videos_readable(self) -> None:
        grid = contact_sheet.auto_contact_sheet_grid(frame_count=59, columns=None)

        self.assertEqual(6, grid.columns)
        self.assertEqual(10, grid.rows)

    def test_explicit_columns_control_rows(self) -> None:
        grid = contact_sheet.auto_contact_sheet_grid(frame_count=23, columns=5)

        self.assertEqual(5, grid.columns)
        self.assertEqual(5, grid.rows)


class ContactSheetCommandTests(unittest.TestCase):
    def test_build_ffmpeg_command_uses_literal_tile_filter(self) -> None:
        command = contact_sheet.build_contact_sheet_command(
            video_path=Path("/tmp/source file.mp4"),
            output_path=Path("/tmp/contact sheet.jpg"),
            fps=1.0,
            thumb_width=240,
            grid=contact_sheet.ContactSheetGrid(columns=6, rows=10),
            padding=6,
            margin=8,
        )

        self.assertEqual("ffmpeg", command[0])
        self.assertIn("/tmp/source file.mp4", command)
        self.assertIn("/tmp/contact sheet.jpg", command)
        vf = command[command.index("-vf") + 1]
        self.assertIn("fps=1", vf)
        self.assertIn("scale=240:-1", vf)
        self.assertIn("tile=6x10:padding=6:margin=8", vf)


class ContactSheetCreationTests(unittest.TestCase):
    def test_url_run_replaces_stale_contact_sheet_media_without_deleting_output_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "out"
            video_dir = output_root / "video"
            video_dir.mkdir(parents=True)
            stale_video = video_dir / "source.avi"
            stale_video.write_bytes(b"stale-video")
            (output_root / "contact-sheet.jpg").write_bytes(b"stale-sheet")
            (output_root / "contact-sheet.json").write_text("{}", encoding="utf-8")
            output_json = output_root / "output.json"
            output_json.write_text('{"keep": true}', encoding="utf-8")
            downloaded_video = video_dir / "source.mp4"

            def fake_download(_source, paths, **_kwargs):
                self.assertFalse(stale_video.exists())
                self.assertFalse((output_root / "contact-sheet.jpg").exists())
                self.assertFalse((output_root / "contact-sheet.json").exists())
                self.assertTrue(output_json.exists())
                downloaded_video.write_bytes(b"new-video")
                return {
                    "format": {"duration": "1.0"},
                    "streams": [{"codec_type": "video"}],
                }, downloaded_video

            def fake_run_command(command):
                self.assertEqual(str(downloaded_video), command[command.index("-i") + 1])
                Path(command[-1]).write_bytes(b"new-sheet")

            with mock.patch.object(
                contact_sheet.pipeline,
                "fetch_youtube_metadata",
                return_value={"id": "demo", "title": "Demo"},
            ), mock.patch.object(
                contact_sheet.pipeline,
                "download_youtube_media",
                side_effect=fake_download,
            ), mock.patch.object(
                contact_sheet.pipeline,
                "run_command",
                side_effect=fake_run_command,
            ):
                result = contact_sheet.create_contact_sheet(
                    "https://youtu.be/demo",
                    out_dir=output_root,
                )

            self.assertEqual(downloaded_video, result.video_path)
            self.assertEqual(b"new-sheet", result.sheet_path.read_bytes())
            self.assertEqual('{"keep": true}', output_json.read_text(encoding="utf-8"))

    def test_create_contact_sheet_for_local_source_writes_manifest_and_keeps_media(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "cute short.mp4"
            source_path.write_bytes(b"video")
            output_root = root / "out"

            def fake_materialize(local_source: Path, paths):
                linked = paths.video_dir / "source.mp4"
                linked.parent.mkdir(parents=True, exist_ok=True)
                linked.write_bytes(local_source.read_bytes())
                return {"format": {"duration": "59.4"}, "streams": [{"codec_type": "video"}]}, linked

            def fake_run_command(command):
                Path(command[-1]).write_bytes(b"sheet")

            with mock.patch.object(contact_sheet.pipeline, "materialize_local_input", side_effect=fake_materialize), mock.patch.object(
                contact_sheet.pipeline,
                "run_command",
                side_effect=fake_run_command,
            ):
                result = contact_sheet.create_contact_sheet(
                    str(source_path),
                    out_dir=output_root,
                    fps=1.0,
                    columns=6,
                    thumb_width=240,
                    max_video_height=720,
                )

            manifest = json.loads((output_root / "contact-sheet.json").read_text(encoding="utf-8"))
            sheet_exists = result.sheet_path.exists()
            video_exists = (output_root / "video" / "source.mp4").exists()

        self.assertEqual(output_root / "contact-sheet.jpg", result.sheet_path)
        self.assertTrue(sheet_exists)
        self.assertTrue(video_exists)
        self.assertEqual(str(result.sheet_path), manifest["contact_sheet"]["path"])
        self.assertFalse(manifest["media"]["is_symlink"])
        self.assertEqual(6, manifest["contact_sheet"]["columns"])
        self.assertEqual(10, manifest["contact_sheet"]["rows"])
        self.assertEqual(60, manifest["contact_sheet"]["estimated_frame_count"])
        self.assertEqual(720, manifest["media"]["requested_max_video_height"])
        self.assertFalse(manifest["media"]["height_limit_applied"])

    def test_contact_sheet_max_video_height_flag_parses(self) -> None:
        args = contact_sheet.parse_args(
            ["--source", "https://youtu.be/demo", "--max-video-height", "720"]
        )

        self.assertEqual(720, args.max_video_height)


if __name__ == "__main__":
    unittest.main()
