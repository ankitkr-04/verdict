"""GGUF weights acquisition: download from Hugging Face when absent.

Called automatically by the local-server manager at startup, so runs never fail
on a missing model. Also runnable standalone to pre-fetch (e.g. before timing
runs, so the download doesn't eat wall-clock budget):

    python -m src.llm.model_fetch
    python -m src.llm.model_fetch --repo unsloth/Qwen3-14B-GGUF --file Qwen3-14B-Q4_K_M.gguf

The active model is exposed at a stable symlink (models/model.gguf) so swapping
models never touches server flags. Uses huggingface_hub when installed (resume
support); otherwise a plain stdlib streaming download. Gated repos: set HF_TOKEN.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

from src.core import settings

log = logging.getLogger("verdict.model_fetch")

_CHUNK = 1 << 20  # 1 MiB


class ModelFetchError(Exception):
    pass


def _download_via_hub(repo: str, file: str, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo, filename=file, local_dir=dest_dir))


def _download_via_stdlib(repo: str, file: str, dest_dir: Path) -> Path:
    url = f"https://huggingface.co/{repo}/resolve/main/{file}"
    dest = dest_dir / file
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url)
    if token := os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", f"Bearer {token}")
    log.info("downloading %s", url)
    with urllib.request.urlopen(req) as resp, open(part, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        next_mark = 5
        while chunk := resp.read(_CHUNK):
            out.write(chunk)
            done += len(chunk)
            if total and done * 100 // total >= next_mark:
                log.info("  %d%%  (%d MiB)", done * 100 // total, done >> 20)
                next_mark = done * 100 // total + 5
    part.replace(dest)
    return dest


def _point_link(link: Path, target: Path) -> None:
    if link == target or (link.exists() and link.resolve() == target.resolve()):
        return
    if link.exists() and not link.is_symlink():
        log.warning("not touching %s: regular file, not a symlink — move it aside", link)
        return
    link.unlink(missing_ok=True)
    link.symlink_to(target.resolve())
    log.info("%s -> %s", link, target.name)


def ensure_model(
    repo: str = settings.MODEL_REPO,
    file: str = settings.MODEL_FILE,
    dest_dir: Path = settings.MODEL_DIR,
    force: bool = False,
) -> Path:
    """Return a path to ready-to-use weights, downloading them first if absent.

    Idempotent and cheap when the file already exists; re-points the stable
    symlink, so switching VERDICT_MODEL_FILE/REPO alone activates a new model.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file

    if dest.exists() and not force:
        log.info("weights present: %s (%d MiB)", dest, dest.stat().st_size >> 20)
    else:
        try:
            dest = _download_via_hub(repo, file, dest_dir)
        except ImportError:
            log.info("huggingface_hub not installed; stdlib download (no resume)")
            try:
                dest = _download_via_stdlib(repo, file, dest_dir)
            except OSError as e:
                raise ModelFetchError(f"download of {repo}/{file} failed: {e}") from e
        except Exception as e:
            raise ModelFetchError(f"download of {repo}/{file} failed: {e}") from e
        log.info("downloaded: %s (%d MiB)", dest, dest.stat().st_size >> 20)

    link = dest_dir / settings.MODEL_LINK.name
    _point_link(link, dest)
    return link if link.exists() else dest


def main() -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Pre-fetch the local GGUF model.")
    ap.add_argument("--repo", default=settings.MODEL_REPO, help="HF repo id")
    ap.add_argument("--file", default=settings.MODEL_FILE, help="GGUF filename in the repo")
    ap.add_argument("--dir", type=Path, default=settings.MODEL_DIR, help="local models dir")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()
    try:
        print(ensure_model(args.repo, args.file, args.dir, force=args.force))
    except ModelFetchError as e:
        print(e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
