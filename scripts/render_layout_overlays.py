from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    selected: dict[str, dict] = {}
    for result in report["results"]:
        if result["found"] and result["hospital_id"] not in selected:
            selected[result["hospital_id"]] = result

    tiles: list[np.ndarray] = []
    for hospital, result in sorted(selected.items()):
        image = cv2.imread(result["image"], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot read {result['image']}")
        for box in result["table_boxes"]:
            left, top, right, bottom = (round(value) for value in box["coordinate"])
            cv2.rectangle(image, (left, top), (right, bottom), (20, 180, 20), 12)
        scale = 420 / image.shape[1]
        tile = cv2.resize(image, (420, round(image.shape[0] * scale)))
        tile = tile[:560]
        canvas = np.full((610, 440, 3), 255, dtype=np.uint8)
        canvas[40 : 40 + tile.shape[0], 10 : 10 + tile.shape[1]] = tile
        cv2.putText(
            canvas,
            f"{hospital} p{result['page_number']}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        tiles.append(canvas)

    columns = 3
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.full((rows * 610, columns * 440, 3), 245, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet[row * 610 : (row + 1) * 610, column * 440 : (column + 1) * 440] = tile
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise RuntimeError(f"cannot write {args.output}")
    print(f"wrote {len(tiles)} overlays to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
