from gmoney.inference.layout_bakeoff import table_boxes


def test_table_boxes_filters_labels_and_scores() -> None:
    output = {
        "pages": [
            {
                "res": {
                    "boxes": [
                        {"label": "table", "score": 0.9, "coordinate": [0, 0, 10, 10]},
                        {"label": "table", "score": 0.2, "coordinate": [0, 0, 10, 10]},
                        {"label": "text", "score": 0.99, "coordinate": [0, 0, 10, 10]},
                    ]
                }
            }
        ]
    }
    assert len(table_boxes(output, minimum_score=0.3)) == 1


def test_table_boxes_treats_empty_output_as_no_detection() -> None:
    assert table_boxes({}) == []
