
import pytest
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
import os

# Mock for YOLO since we can't run real segmentation
from unittest.mock import MagicMock, patch

# ========== Tests for segmenter.py ==========

@patch("segmenter.YOLO")
def test_segment_image(mock_yolo):
    from yolo8.segmenter import segment_image

    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    temp_path = Path(tempfile.gettempdir()) / "test_img.jpg"
    Image.fromarray(dummy_image).save(temp_path)

    # Setup mock
    mock_result = MagicMock()
    mock_result.boxes.xyxy.tolist.return_value = [[10, 10, 50, 50]]
    mock_result.boxes.cls = [0]
    mock_result.boxes.conf = [0.95]
    mock_result.boxes = MagicMock()
    mock_result.boxes.xyxy = MagicMock()
    mock_result.boxes.cls = [0]
    mock_result.boxes.conf = [0.95]
    mock_result.masks = MagicMock()
    mock_result.masks.data.cpu().numpy.return_value = np.ones((1, 40, 40))
    mock_result.names = ["test"]
    mock_result.probs = MagicMock()
    mock_result.probs.tolist.return_value = [0.95]

    mock_yolo_instance = MagicMock(return_value=[mock_result])
    mock_yolo.return_value = mock_yolo_instance

    result = segment_image(str(temp_path))
    assert "boxes" in result
    assert result["boxes"][0]["confidence"] == 0.95


# ========== Tests for fileloade.py core functions ==========

def test_classify_text_math():
    from fileloade import classify_text
    assert classify_text("sin(x) + cos(x) = 1") == "math_equation"

def test_classify_text_chemical():
    from fileloade import classify_text
    assert classify_text("H2 + O2 -> H2O") == "chemical_equation"

def test_classify_text_handwriting():
    from fileloade import classify_text
    assert classify_text("This is handwritten with ink") == "handwriting"

def test_classify_text_diagram():
    from fileloade import classify_text
    assert classify_text("This flowchart shows...") == "diagram"

def test_save_and_load_processed(tmp_path):
    from fileloade import save_processed, PROCESSED_TRACKER
    test_path = str(tmp_path / "file1.txt")
    save_processed(test_path)
    with open(PROCESSED_TRACKER, "r") as f:
        data = f.read()
    assert test_path in data


# ========== Tests for functions.py ==========

def test_run_cypher_query_mocked():
    with patch("functions.GraphDatabase.driver") as mock_driver:
        from functions import run_cypher_query
        mock_session = MagicMock()
        mock_session.run.return_value = [{"mocked": True}]
        mock_driver.return_value.session.return_value.__enter__.return_value = mock_session
        result = run_cypher_query("MATCH (n) RETURN n")
        assert result == [{"mocked": True}]
