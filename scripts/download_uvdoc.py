from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from gmoney.inference.uvdoc import UVDOC_MODEL_REPOSITORY, UVDOC_MODEL_REVISION

FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("model-cache/uvdoc"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        path = hf_hub_download(
            repo_id=UVDOC_MODEL_REPOSITORY,
            revision=UVDOC_MODEL_REVISION,
            filename=filename,
            local_dir=args.output,
        )
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
