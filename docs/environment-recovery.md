# Environment Recovery

Keep a working environment intact while testing repairs or dependency updates.
The optional [runtime snapshot](../constraints/macos-arm64-py311.txt) records
package versions from macOS 27 Apple Silicon with CPython 3.11. It does not change
normal package installation or require another environment manager.

## Build Separately

Run from the intended repository revision on macOS arm64 with Python 3.11.
This snapshot includes MLX wheels requiring macOS 26 or newer; the recovery
check was performed on macOS 27. Older macOS versions need their own baseline.

```bash
RECOVERY_DIR="$(mktemp -d -t youtube-recovery)"
python3.11 -m venv "$RECOVERY_DIR/venv"
"$RECOVERY_DIR/venv/bin/python" -m pip install \
  -c constraints/macos-arm64-py311.txt \
  '.[youtube,mlx-whisper]' openai-whisper
"$RECOVERY_DIR/venv/bin/python" -m pip check
"$RECOVERY_DIR/venv/bin/python" -m unittest discover -v
```

The install copies this checkout into the new environment; it does not replace
the active `.venv` or deploy skills. The explicit `openai-whisper` requirement
installs the default local ASR backend. MLX remains an opt-in extra. A constraint
limits a package's version but does not request its installation.

The tests do not request model weights, run live YouTube downloads, or send media
to a model provider. The local-media smoke requires `ffmpeg`, `ffprobe`, and
OpenCV; check for skipped tests. Before using ASR after recovery, separately
validate the chosen backend with a short approved clip and its required weights.

Do not move a virtual environment into `.venv`: installed entry points contain
absolute interpreter paths. After verification, retain the old environment and
recreate a replacement at its final location using the same recipe. Switching
the active checkout or deploying skills is a separate, deliberate action. Delete
the disposable recovery directory when it is no longer needed.

## What Is Not Captured

- The Python interpreter, macOS version and system tools such as ffmpeg/tesseract.
- ASR model weights, credentials, configuration, or local output.
- Hash-locked wheels, build-isolation dependencies, or availability of packages
  on the index. This is a recovery baseline, not an offline or bit-for-bit build.
- Other platforms, older macOS releases or Python versions. Use normal
  installation and verify them independently rather than applying these
  platform-specific pins blindly.

## Update Selectively

YouTube changes can require a newer yt-dlp before other dependencies need an
update. In the disposable environment, try an explicit version separately:

```bash
"$RECOVERY_DIR/venv/bin/python" -m pip install --upgrade 'yt-dlp[default]==VERSION'
"$RECOVERY_DIR/venv/bin/python" -m pip check
"$RECOVERY_DIR/venv/bin/python" -m unittest discover -v
```

Replace `VERSION` with the release being evaluated. This intentionally omits
the old constraints for that upgrade. Inspect changes to its dependencies,
verify the affected source with an approved small canary, then update the
snapshot. Do not upgrade the whole live environment merely to repair a download.

When refreshing the snapshot, review `pip freeze --exclude-editable` rather than
publishing its output blindly. Keep only package/version pins; exclude local
paths, editable installs, private indexes, direct URLs and credentials. Record
the tested repo revision and verification limits with the maintenance change.
