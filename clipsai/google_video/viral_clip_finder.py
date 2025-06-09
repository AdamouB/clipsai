"""Utilities for finding viral clips using Google Video Intelligence."""

from __future__ import annotations

from google.cloud import videointelligence_v1 as vi


def _read_video_bytes(path: str) -> bytes:
    """Read a video file and return its bytes."""
    with open(path, "rb") as f:
        return f.read()


def find_viral_clips(
    video_path: str,
    clip_length: str | float = "auto",
    max_results: int = 10,
) -> list[dict[str, float]]:
    """Find potential viral clips using Google Cloud Video Intelligence.

    Parameters
    ----------
    video_path:
        Path to the video file to analyze.
    clip_length:
        Desired clip length in seconds. Use ``"auto"`` to return detected shots
        ordered by duration.
    max_results:
        Maximum number of clips to return.

    Returns
    -------
    list of dict
        Each dictionary contains ``start_time`` and ``end_time`` in seconds.
    """
    client = vi.VideoIntelligenceServiceClient()
    features = [vi.Feature.SHOT_CHANGE_DETECTION]
    input_content = _read_video_bytes(video_path)

    operation = client.annotate_video(
        request={"features": features, "input_content": input_content}
    )
    result = operation.result(timeout=180)
    shots = result.annotation_results[0].shot_annotations

    segments = []
    for shot in shots:
        start = shot.start_time_offset.total_seconds()
        end = shot.end_time_offset.total_seconds()
        segments.append((start, end))

    if not segments:
        return []

    if clip_length == "auto":
        segments.sort(key=lambda s: s[1] - s[0], reverse=True)
        clips = [{"start_time": s, "end_time": e} for s, e in segments[:max_results]]
        return clips

    try:
        length = float(clip_length)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError("clip_length must be 'auto' or a number") from exc

    clips = []
    for start, end in segments:
        clip_start = start
        while clip_start < end and len(clips) < max_results:
            clip_end = min(clip_start + length, end)
            clips.append({"start_time": clip_start, "end_time": clip_end})
            clip_start += length
        if len(clips) >= max_results:
            break
    return clips
