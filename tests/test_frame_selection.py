from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from youtube_analysis_tool import triage
from youtube_analysis_tool.pipeline import (
    analysis_kwargs_from_args,
    analysis_paths,
    analyze_source,
    evenly_spaced_indices,
    parse_args,
    select_keyframes_for_processing,
)


def frame_row(filename: str, timestamp: float) -> dict[str, object]:
    return {
        "kind": "scene",
        "filename": filename,
        "timestamp_seconds": timestamp,
        "timestamp_hms": f"00:00:{int(timestamp):02d}",
    }


class FrameSelectionTests(unittest.TestCase):
    def test_evenly_spaced_indices_preserve_first_and_last(self) -> None:
        self.assertEqual([0, 2, 4], evenly_spaced_indices(5, 3))
        self.assertEqual([0, 1, 2], evenly_spaced_indices(3, 4))
        self.assertEqual([0], evenly_spaced_indices(5, 1))

    def test_uncapped_selection_preserves_all_rows_without_prepass(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = analysis_paths(Path(tmpdir))
            rows = [frame_row("b.jpg", 10.0), frame_row("a.jpg", 0.0)]
            with mock.patch(
                "youtube_analysis_tool.pipeline.triage.read_grayscale_image"
            ) as image_reader, mock.patch(
                "youtube_analysis_tool.pipeline.triage.frame_content_hash"
            ) as content_hasher, mock.patch(
                "youtube_analysis_tool.pipeline.triage.group_by_content_hash"
            ) as grouping:
                selection = select_keyframes_for_processing(rows, paths, max_frames=None)

            manifest = json.loads(paths.visual_selection_path.read_text(encoding="utf-8"))

        image_reader.assert_not_called()
        content_hasher.assert_not_called()
        grouping.assert_not_called()
        self.assertEqual(list(reversed(rows)), selection.rows)
        self.assertEqual("uncapped", selection.method)
        self.assertEqual(2, selection.candidate_count)
        self.assertEqual(2, selection.deduplicated_count)
        self.assertEqual(2, selection.selected_count)
        self.assertEqual(
            {
                "candidate_frame_count": 2,
                "deduplicated_frame_count": 2,
                "selected_frame_count": 2,
                "max_frames": None,
                "selection_method": "uncapped",
                "frames": [
                    {
                        **row,
                        "selected": True,
                        "selection_reason": "uncapped",
                        "duplicate_group": None,
                        "representative_filename": row["filename"],
                    }
                    for row in reversed(rows)
                ],
            },
            manifest,
        )

    def test_capped_selection_preserves_exact_groups_and_stable_ranking(self) -> None:
        rows = [
            frame_row("late.jpg", 20.0),
            frame_row("z-first.jpg", 10.0),
            frame_row("different.jpg", 5.0),
            frame_row("blurry.jpg", 0.0),
            frame_row("a-second.jpg", 10.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = analysis_paths(Path(tmpdir))
            paths.keyframes_dir.mkdir(parents=True)
            for row in rows:
                content = b"different-content" if row["filename"] == "different.jpg" else b"same-content"
                (paths.keyframes_dir / str(row["filename"])).write_bytes(content)
            with mock.patch.object(
                triage, "read_grayscale_image",
                side_effect=lambda path: [[100.0 if path.name == "blurry.jpg" else 200.0]],
            ), mock.patch.object(
                triage, "compute_phash", return_value="0000000000000000",
            ), mock.patch.object(
                triage, "compute_blur_score", side_effect=lambda matrix: matrix[0][0],
            ):
                selection = select_keyframes_for_processing(rows, paths, max_frames=2)

            manifest = json.loads(paths.visual_selection_path.read_text(encoding="utf-8"))

        self.assertEqual(["different.jpg", "z-first.jpg"], [row["filename"] for row in selection.rows])
        self.assertEqual(5, selection.candidate_count)
        self.assertEqual(2, selection.deduplicated_count)
        self.assertEqual(2, selection.selected_count)
        self.assertEqual("exact_content_then_time_targets", selection.method)
        self.assertEqual(
            ["blurry.jpg", "different.jpg", "z-first.jpg", "a-second.jpg", "late.jpg"],
            [entry["filename"] for entry in selection.entries],
        )
        self.assertEqual(
            ["pre-dup-0001", "pre-dup-0002", "pre-dup-0001", "pre-dup-0001", "pre-dup-0001"],
            [entry["duplicate_group"] for entry in selection.entries],
        )
        self.assertEqual(
            ["z-first.jpg", "different.jpg", "z-first.jpg", "z-first.jpg", "z-first.jpg"],
            [entry["representative_filename"] for entry in selection.entries],
        )
        self.assertEqual(
            ["duplicate", "selected", "selected", "duplicate", "duplicate"],
            [entry["selection_reason"] for entry in selection.entries],
        )
        self.assertEqual(
            {
                "candidate_frame_count": 5,
                "deduplicated_frame_count": 2,
                "selected_frame_count": 2,
                "max_frames": 2,
                "selection_method": "exact_content_then_time_targets",
                "frames": selection.entries,
            },
            manifest,
        )
        self.assertEqual(
            set(rows[0]) | {"phash", "content_sha256", "blur_score", "duplicate_group",
                           "representative_filename", "selected", "selection_reason"},
            set(selection.entries[0]),
        )

    def test_capped_selection_leaves_unknown_content_hashes_distinct(self) -> None:
        rows = [frame_row(f"frame-{index}.jpg", float(index)) for index in range(4)]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = analysis_paths(Path(tmpdir))
            with mock.patch.object(
                triage, "read_grayscale_image", return_value=None,
            ), mock.patch.object(
                triage, "compute_phash", return_value="0000000000000000",
            ), mock.patch.object(
                triage, "frame_content_hash", side_effect=[None, "", None, ""],
            ):
                selection = select_keyframes_for_processing(rows, paths, max_frames=4)

        self.assertEqual(rows, selection.rows)
        self.assertEqual(4, selection.deduplicated_count)
        self.assertEqual(4, selection.selected_count)
        self.assertEqual(
            [f"pre-dup-{index:04d}" for index in range(1, 5)],
            [entry["duplicate_group"] for entry in selection.entries],
        )

    def test_uncapped_selection_still_deduplicates_during_triage(self) -> None:
        rows = [frame_row("early.jpg", 0.0), frame_row("text.jpg", 1.0), frame_row("different.jpg", 2.0)]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = analysis_paths(Path(tmpdir))
            paths.keyframes_dir.mkdir(parents=True)
            for row in rows:
                content = b"different-content" if row["filename"] == "different.jpg" else b"same-content"
                (paths.keyframes_dir / str(row["filename"])).write_bytes(content)
            selection = select_keyframes_for_processing(rows, paths, max_frames=None)
            with mock.patch.object(
                triage, "read_grayscale_image", return_value=None,
            ), mock.patch.object(
                triage, "compute_phash", return_value="0000000000000000",
            ), mock.patch.object(
                triage, "compute_blur_score", return_value=200.0,
            ):
                frames = triage.build_frame_records(
                    paths.root,
                    selection.rows,
                    [{"filename": "text.jpg", "text": "more OCR text"}],
                )

        self.assertEqual(rows, selection.rows)
        self.assertEqual(3, selection.deduplicated_count)
        self.assertEqual(["dup-0001", "dup-0001", "dup-0002"], [frame["duplicate_group"] for frame in frames])
        self.assertEqual(
            ["text.jpg", "different.jpg"],
            [frame["filename"] for frame in frames if frame["is_duplicate_representative"]],
        )

    def test_capped_selection_deduplicates_before_even_timeline_cap(self) -> None:
        hashes = {
            "a.jpg": "0000000000000000",
            "b.jpg": "0000000000000000",
            "c.jpg": "ffffffffffffffff",
            "d.jpg": "aaaaaaaaaaaaaaaa",
            "e.jpg": "5555555555555555",
            "f.jpg": "0f0f0f0f0f0f0f0f",
        }
        blur = {
            "a.jpg": 10.0,
            "b.jpg": 20.0,
            "c.jpg": 30.0,
            "d.jpg": 40.0,
            "e.jpg": 50.0,
            "f.jpg": 60.0,
        }
        rows = [
            frame_row("a.jpg", 0.0),
            frame_row("b.jpg", 10.0),
            frame_row("c.jpg", 20.0),
            frame_row("d.jpg", 30.0),
            frame_row("e.jpg", 40.0),
            frame_row("f.jpg", 50.0),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = analysis_paths(Path(tmpdir))
            paths.keyframes_dir.mkdir(parents=True)
            for row in rows:
                (paths.keyframes_dir / str(row["filename"])).write_bytes(hashes[str(row["filename"])].encode())

            with mock.patch(
                "youtube_analysis_tool.pipeline.triage.read_grayscale_image",
                side_effect=lambda path: [[blur[path.name]]],
            ), mock.patch(
                "youtube_analysis_tool.pipeline.triage.compute_phash",
                side_effect=lambda matrix, path: hashes[path.name],
            ), mock.patch(
                "youtube_analysis_tool.pipeline.triage.compute_blur_score",
                side_effect=lambda matrix: matrix[0][0],
            ):
                selection = select_keyframes_for_processing(rows, paths, max_frames=3)

            manifest = json.loads(paths.visual_selection_path.read_text(encoding="utf-8"))

        self.assertEqual(6, selection.candidate_count)
        self.assertEqual(5, selection.deduplicated_count)
        self.assertEqual(3, selection.selected_count)
        self.assertEqual(
            ["b.jpg", "d.jpg", "f.jpg"],
            [str(row["filename"]) for row in selection.rows],
        )
        reasons = {entry["filename"]: entry["selection_reason"] for entry in manifest["frames"]}
        self.assertEqual("duplicate", reasons["a.jpg"])
        self.assertEqual("selected", reasons["b.jpg"])
        self.assertEqual("over_budget", reasons["c.jpg"])
        self.assertEqual("selected", reasons["d.jpg"])
        self.assertEqual("over_budget", reasons["e.jpg"])
        self.assertEqual("selected", reasons["f.jpg"])

    def test_analyze_source_sends_only_selected_rows_to_ocr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "demo.mp4"
            source_path.write_bytes(b"video")
            output_root = root / "out"
            captured_filenames: list[str] = []

            def fake_create_keyframes(video_path, metadata, paths, **kwargs):
                del video_path, metadata, kwargs
                rows = [frame_row(f"frame-{index}.jpg", float(index * 10)) for index in range(4)]
                for row in rows:
                    (paths.keyframes_dir / str(row["filename"])).write_bytes(str(row["filename"]).encode())
                return rows

            def fake_ocr(paths, rows, *, ocr_mode):
                del paths, ocr_mode
                captured_filenames.extend(str(row["filename"]) for row in rows)
                return [], {
                    "mode": "on",
                    "status": "completed",
                    "attempted": True,
                    "frame_count": len(rows),
                    "error": None,
                }

            hashes = [
                "0000000000000000",
                "ffffffffffffffff",
                "aaaaaaaaaaaaaaaa",
                "5555555555555555",
            ]
            with mock.patch(
                "youtube_analysis_tool.pipeline.materialize_local_input",
                return_value=(
                    {"format": {"duration": "40"}, "streams": [{"codec_type": "video"}]},
                    source_path,
                ),
            ), mock.patch(
                "youtube_analysis_tool.pipeline.create_keyframes",
                side_effect=fake_create_keyframes,
            ), mock.patch(
                "youtube_analysis_tool.pipeline.triage.read_grayscale_image",
                return_value=[[200.0]],
            ), mock.patch(
                "youtube_analysis_tool.pipeline.triage.compute_phash",
                side_effect=hashes,
            ), mock.patch(
                "youtube_analysis_tool.pipeline.triage.compute_blur_score",
                return_value=200.0,
            ), mock.patch(
                "youtube_analysis_tool.pipeline.run_ocr_stage",
                side_effect=fake_ocr,
            ):
                analyze_source(
                    str(source_path),
                    out_dir=output_root,
                    transcript_mode="off",
                    visuals_mode="on",
                    max_frames=2,
                    ocr_mode="on",
                    triage_mode="off",
                    artifacts_mode="debug",
                )

            output = json.loads((output_root / "output.json").read_text(encoding="utf-8"))
            selection = json.loads(
                (output_root / "visuals" / "selection.json").read_text(encoding="utf-8")
            )
            candidate_images = sorted(
                path.name for path in (output_root / "visuals" / "candidates").glob("*.jpg")
            )

        self.assertEqual(["frame-0.jpg", "frame-3.jpg"], captured_filenames)
        self.assertEqual(4, output["visual_sampling"]["candidate_frame_count"])
        self.assertEqual(4, output["visual_sampling"]["deduplicated_frame_count"])
        self.assertEqual(2, output["visual_sampling"]["selected_frame_count"])
        self.assertEqual(2, output["visual_sampling"]["max_frames"])
        self.assertEqual(2, selection["selected_frame_count"])
        self.assertEqual(["frame-0.jpg", "frame-3.jpg"], candidate_images)
        selected_entries = [entry for entry in selection["frames"] if entry["selected"]]
        self.assertTrue(all(entry["debug_image_path"] for entry in selected_entries))

    def test_max_frames_cli_is_optional_and_positive(self) -> None:
        args = parse_args(["--source", "demo.mp4", "--max-frames", "36"])
        self.assertEqual(36, args.max_frames)
        self.assertEqual(36, analysis_kwargs_from_args(args)["max_frames"])

        default_args = parse_args(["--source", "demo.mp4"])
        self.assertIsNone(default_args.max_frames)


if __name__ == "__main__":
    unittest.main()
