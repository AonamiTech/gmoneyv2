from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPOSITORY = "PaddlePaddle/PaddleOCR-VL-1.6-GGUF"
FILES = (
    "PaddleOCR-VL-1.6-GGUF.gguf",
    "PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("model-cache/paddleocr-vl-1.6"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        path = hf_hub_download(
            repo_id=REPOSITORY,
            filename=filename,
            local_dir=args.output,
        )
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
