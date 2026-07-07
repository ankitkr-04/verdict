"""Download the local GGUF model from Hugging Face if it isn't on disk yet.

Idempotent: run it before starting llama-server; it does nothing when the file
is already present. The active model is exposed at a stable path
(models/model.gguf, a symlink) so swapping models never touches server flags.

    python scripts/fetch_model.py                    # defaults from settings.py
    VERDICT_MODEL_FILE=...Q8_0.gguf python scripts/fetch_model.py
    python scripts/fetch_model.py --repo unsloth/Qwen3-14B-GGUF --file Qwen3-14B-Q4_K_M.gguf

Uses huggingface_hub when installed (resume + progress bars); otherwise falls
back to a plain stdlib streaming download. Gated repos: export HF_TOKEN.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import settings

CHUNK = 1 << 20  # 1 MiB


def download_via_hub(repo: str, file: str, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo, filename=file, local_dir=dest_dir)
    return Path(path)


def download_via_stdlib(repo: str, file: str, dest_dir: Path) -> Path:
    import os

    url = f"https://huggingface.co/{repo}/resolve/main/{file}"
    dest = dest_dir / file
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url)
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    print(f"downloading {url}", file=sys.stderr)
    with urllib.request.urlopen(req) as resp, open(part, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        next_mark = 0
        while chunk := resp.read(CHUNK):
            out.write(chunk)
            done += len(chunk)
            if total and done * 100 // total >= next_mark:
                print(f"  {done * 100 // total}%  ({done >> 20} MiB)", file=sys.stderr)
                next_mark = done * 100 // total + 5
    part.replace(dest)
    return dest


def point_link(link: Path, target: Path) -> None:
    if link.resolve() == target.resolve():
        return
    if link.exists() and not link.is_symlink():
        print(
            f"NOT touching {link}: it is a regular file, not a symlink.\n"
            f"Move it aside, then re-run to link the active model.",
            file=sys.stderr,
        )
        return
    link.unlink(missing_ok=True)
    link.symlink_to(target.resolve())
    print(f"{link} -> {target.name}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=settings.MODEL_REPO, help="HF repo id")
    ap.add_argument("--file", default=settings.MODEL_FILE, help="GGUF filename in the repo")
    ap.add_argument("--dir", type=Path, default=settings.MODEL_DIR, help="local models dir")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    args.dir.mkdir(parents=True, exist_ok=True)
    dest = args.dir / args.file

    if dest.exists() and not args.force:
        print(f"already present: {dest} ({dest.stat().st_size >> 20} MiB)", file=sys.stderr)
    else:
        try:
            dest = download_via_hub(args.repo, args.file, args.dir)
        except ImportError:
            print("huggingface_hub not installed; using stdlib fallback "
                  "(pip install huggingface_hub for resume support)", file=sys.stderr)
            try:
                dest = download_via_stdlib(args.repo, args.file, args.dir)
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                print(f"download failed: {e}", file=sys.stderr)
                return 1
        except Exception as e:  # hub raises its own error types; keep the CLI graceful
            print(f"download failed: {e}", file=sys.stderr)
            return 1
        print(f"downloaded: {dest} ({dest.stat().st_size >> 20} MiB)", file=sys.stderr)

    point_link(args.dir / settings.MODEL_LINK.name, dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
