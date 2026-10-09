from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

# Official Apache-2.0 weights. Do not substitute the unofficial ldov/TeleOCR copy.
REPOSITORY = "StarDoc-AI/TeleOCR"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("model-cache/teleocr"))
    parser.add_argument(
        "--revision",
        default=None,
        help="Pin a commit SHA from the model page for reproducible runs.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=REPOSITORY,
        revision=args.revision,
        local_dir=args.output,
        allow_patterns=["*.json", "*.py", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.md"],
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
