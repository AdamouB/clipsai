import pytest
from unittest.mock import MagicMock, patch, mock_open

from clipsai.google_video.viral_clip_finder import find_viral_clips


class FakeDuration:
    def __init__(self, seconds: float) -> None:
        self._seconds = seconds

    def total_seconds(self) -> float:
        return self._seconds


class FakeShot:
    def __init__(self, start: float, end: float) -> None:
        self.start_time_offset = FakeDuration(start)
        self.end_time_offset = FakeDuration(end)


def build_operation(shots):
    result = MagicMock()
    result.annotation_results = [MagicMock(shot_annotations=shots)]
    operation = MagicMock()
    operation.result.return_value = result
    return operation


@patch("clipsai.google_video.viral_clip_finder.vi.VideoIntelligenceServiceClient")
def test_find_viral_clips_auto(mock_client):
    shots = [FakeShot(0, 5), FakeShot(5, 15), FakeShot(15, 20)]
    mock_client.return_value.annotate_video.return_value = build_operation(shots)
    with patch("builtins.open", mock_open(read_data=b"data")):
        clips = find_viral_clips("video.mp4", clip_length="auto", max_results=2)
    assert clips == [
        {"start_time": 5.0, "end_time": 15.0},
        {"start_time": 0.0, "end_time": 5.0},
    ]


@patch("clipsai.google_video.viral_clip_finder.vi.VideoIntelligenceServiceClient")
def test_find_viral_clips_fixed_length(mock_client):
    shots = [FakeShot(0, 10)]
    mock_client.return_value.annotate_video.return_value = build_operation(shots)
    with patch("builtins.open", mock_open(read_data=b"data")):
        clips = find_viral_clips("video.mp4", clip_length=4, max_results=3)
    assert clips == [
        {"start_time": 0.0, "end_time": 4.0},
        {"start_time": 4.0, "end_time": 8.0},
        {"start_time": 8.0, "end_time": 10.0},
    ]


@patch("clipsai.google_video.viral_clip_finder.vi.VideoIntelligenceServiceClient")
def test_find_viral_clips_invalid_length(mock_client):
    shots = [FakeShot(0, 10)]
    mock_client.return_value.annotate_video.return_value = build_operation(shots)
    with patch("builtins.open", mock_open(read_data=b"data")):
        with pytest.raises(ValueError):
            find_viral_clips("video.mp4", clip_length="bad")
