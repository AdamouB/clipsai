"""
Resizing/cropping a media file to a different aspect ratio

Notes
-----
- ROI is "region of interest"
"""
# standard library imports
import logging

# current package imports
from .crops import Crops
from .exceptions import ResizerError
from .img_proc import calc_img_bytes
from .rect import Rect
from .segment import Segment
from .vid_proc import extract_frames

# local package imports
from clipsai.media.editor import MediaEditor
from clipsai.media.video_file import VideoFile
from clipsai.utils import pytorch
from clipsai.utils.conversions import bytes_to_gibibytes

# 3rd party imports
import cv2
from facenet_pytorch import MTCNN # Should be mocked in tests if not available
import mediapipe as mp # Should be mocked in tests if not available
import numpy as np
from sklearn.cluster import KMeans
import torch


class Resizer:
    """
    A class for calculating the initial coordinates for resizing by using
    segmentation and face detection.
    """

    def __init__(
        self,
        face_detect_margin: int = 20,
        face_detect_post_process: bool = False,
        device: str = None,
    ) -> None:
        """
        Initializes the Resizer with specific configurations for face
        detection. This class uses FaceNet for detecting faces and MediaPipe for
        analyzing mouth to aspect ratio to determine whose speaking within video frames.

        Parameters
        ----------
        face_detect_margin: int, optional
            The margin around detected faces, specified in pixels. Increasing this
            value results in a larger area around each detected face being included.
            Default is 20 pixels.
        face_detect_post_process: bool, optional
            Determines whether to apply post-processing on the detected faces. Setting
            this to False prevents normalization of output images, making them appear
            more natural to the human eye. Default is False (no post-processing).
        device: str, optional
            PyTorch device to perform computations on. Ex: 'cpu', 'cuda'. Default is
            None (auto detects the correct device)
        """
        if device is None:
            device = pytorch.get_compute_device()
        pytorch.assert_compute_device_available(device)
        logging.debug("FaceNet using device: {}".format(device))

        # These will raise ImportError if not installed, which is acceptable
        # if these parts of the library are considered optional.
        # For unit tests, these should be mocked.
        self._face_detector = MTCNN(
            margin=face_detect_margin,
            post_process=face_detect_post_process,
            device=device,
        )
        self._face_mesher = mp.solutions.face_mesh.FaceMesh()
        self._media_editor = MediaEditor()

    def resize(
        self,
        video_file: VideoFile,
        speaker_segments: list[dict],
        scene_changes: list[float],
        aspect_ratio: tuple = (9, 16),
        samples_per_segment: int = 13,
        face_detect_width: int = 960,
        n_face_detect_batches: int = 8,
        scene_merge_threshold: float = 0.25,
    ) -> Crops:
        """
        Calculates the coordinates to resize the video to for different
        segments given the diarized speaker segments and the desired aspect
        ratio.

        Parameters
        ----------
        video_file: VideoFile
            The video file to resize
        speaker_segments: list[dict]
            speakers: list[int]
                list of speakers (represented by int) talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
        scene_changes: list[float]
            List of scene change times in seconds
        aspect_ratio: tuple[int, int]
            The (width,height) aspect ratio to resize the video to
        samples_per_segment: int
            Number of frames to sample per segment for face detection.
        face_detect_width: int
            The width to use for face detection
        n_face_detect_batches: int
            Number of batches for GPU face detection in a video file
        scene_merge_threshold: float
            The threshold in seconds for merging scene changes with speaker segments.
            Scene changes within this threshold of a segment's start or end time will
            cause the segment to be adjusted.

        Returns
        -------
        Crops
            the resized speaker segments
        """
        logging.debug(
            "Video Resolution: {}x{}".format(
                video_file.get_width_pixels(), video_file.get_height_pixels()
            )
        )
        # calculate resize dimensions
        resize_width, resize_height = self._calc_resize_width_and_height_pixels(
            original_width_pixels=video_file.get_width_pixels(),
            original_height_pixels=video_file.get_height_pixels(),
            resize_aspect_ratio=aspect_ratio,
        )

        if not speaker_segments:
            logging.info(
                "No speaker segments provided or diarization skipped. "
                "Applying default center crop for the entire video."
            )
            video_width = video_file.get_width_pixels()
            video_height = video_file.get_height_pixels()

            default_x = max(0, (video_width - resize_width) // 2)
            default_y = max(0, (video_height - resize_height) // 2)

            video_duration = video_file.get_duration()
            if video_duration < 0:
                logging.warning("Could not read video duration for fallback crop. Defaulting to 0 duration.")
                video_duration = 0.0

            fallback_segment = Segment(
                speakers=[],
                start_time=0.0,
                end_time=video_duration,
                x=default_x,
                y=default_y
            )
            crop_segments = [fallback_segment]

            return Crops(
                original_width=video_width,
                original_height=video_height,
                crop_width=resize_width,
                crop_height=resize_height,
                segments=crop_segments
            )

        logging.debug(
            "Merging {} speaker segments with {} scene changes.".format(
                len(speaker_segments), len(scene_changes)
            )
        )
        segments = self._merge_scene_change_and_speaker_segments(
            speaker_segments, scene_changes, scene_merge_threshold
        )
        logging.debug("Video has {} distinct segments.".format(len(segments)))

        logging.debug("Determining the first second with a face for each segment.")
        segments = self._find_first_sec_with_face_for_each_segment(
            segments, video_file, face_detect_width, n_face_detect_batches
        )

        logging.debug(
            "Determining the region of interest for {} segments.".format(len(segments))
        )
        segments = self._add_x_y_coords_to_each_segment(
            segments,
            video_file,
            resize_width,
            resize_height,
            samples_per_segment,
            face_detect_width,
            n_face_detect_batches,
        )

        logging.debug("Merging identical segments together.")
        unmerge_segments_length = len(segments)
        segments = self._merge_identical_segments(segments, video_file)
        logging.debug(
            "Merged {} identical segments.".format(
                unmerge_segments_length - len(segments)
            )
        )

        crop_segments = []
        for segment in segments:
            crop_segments.append(
                Segment(
                    speakers=segment["speakers"],
                    start_time=segment["start_time"],
                    end_time=segment["end_time"],
                    x=segment["x"],
                    y=segment["y"],
                )
            )

        crops = Crops(
            original_width=video_file.get_width_pixels(),
            original_height=video_file.get_height_pixels(),
            crop_width=resize_width,
            crop_height=resize_height,
            segments=crop_segments,
        )

        return crops

    def _calc_resize_width_and_height_pixels(
        self,
        original_width_pixels: int,
        original_height_pixels: int,
        resize_aspect_ratio: tuple[int, int],
    ) -> tuple[int, int]:
        """
        Calculate the number of pixels along the width and height to resize the video
        to based on the desired aspect ratio.

        Parameters
        ----------
        original_pixels_width: int
            Number of pixels along the width of the original video.
        original_pixels_height: int
            Number of pixels along the height of the original video
        resize_aspect_ratio: tuple[int, int]
            The width:height aspect ratio to resize the video to

        Returns
        -------
        tuple[int, int]
            The number of pixels along the width and height to resize the video to
        """
        resize_ar_width, resize_ar_height = resize_aspect_ratio
        desired_aspect_ratio = resize_ar_width / resize_ar_height
        original_aspect_ratio = original_width_pixels / original_height_pixels

        # original aspect ratio is wider than desired aspect ratio
        if original_aspect_ratio > desired_aspect_ratio:
            resize_height_pixels = original_height_pixels
            resize_width_pixels = int(
                resize_height_pixels * resize_ar_width / resize_ar_height
            )
        # original aspect ratio is taller than desired aspect ratio
        else:
            resize_width_pixels = original_width_pixels
            resize_height_pixels = int(
                resize_width_pixels * resize_ar_height / resize_ar_width
            )

        return resize_width_pixels, resize_height_pixels

    def _merge_scene_change_and_speaker_segments(
        self,
        speaker_segments: list[dict],
        scene_changes: list[float],
        scene_merge_threshold: float,
    ) -> list[dict]:
        """
        Merge scene change segments with speaker segments based on a specified
        threshold.

        Parameters
        ----------
        speaker_segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
        scene_changes: list[float]
            List of scene change times in seconds.
        scene_merge_threshold: float
            The threshold in seconds for merging scene changes with speaker segments.
            Scene changes within this threshold of a segment's start or end time will
            cause the segment to be adjusted.

        Returns
        -------
        updated_speaker_segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
        """
        segments_idx = 0
        for scene_change_sec in scene_changes:
            segment = speaker_segments[segments_idx]
            while scene_change_sec > (segment["end_time"]):
                segments_idx += 1
                if segments_idx >= len(speaker_segments): # Boundary condition
                    return speaker_segments
                segment = speaker_segments[segments_idx]
            # scene change is close to speaker segment end -> merge the two
            if 0 < (segment["end_time"] - scene_change_sec) < scene_merge_threshold:
                segment["end_time"] = scene_change_sec
                if segments_idx == len(speaker_segments) - 1:
                    continue
                next_segment = speaker_segments[segments_idx + 1]
                next_segment["start_time"] = scene_change_sec
                continue
            # scene change is close to speaker segment start -> merge the two
            if 0 < (scene_change_sec - segment["start_time"]) < scene_merge_threshold:
                segment["start_time"] = scene_change_sec
                if segments_idx == 0:
                    continue
                prev_segment = speaker_segments[segments_idx - 1]
                prev_segment["end_time"] = scene_change_sec
                continue
            # scene change already exists
            if scene_change_sec == segment["end_time"]:
                continue
            # add scene change to segments
            new_segment = {
                "start_time": scene_change_sec,
                "speakers": segment["speakers"],
                "end_time": segment["end_time"],
            }
            segment["end_time"] = scene_change_sec
            speaker_segments = (
                speaker_segments[: segments_idx + 1]
                + [new_segment]
                + speaker_segments[segments_idx + 1 :]
            )

        return speaker_segments

    def _find_first_sec_with_face_for_each_segment(
        self,
        segments: list[dict],
        video_file: VideoFile,
        face_detect_width: int,
        n_face_detect_batches: int,
    ) -> list[dict]:
        """
        Find the first frame in a segment with a face.

        Parameters
        ----------
        segments: list[dict]
            List of speaker segments (dictionaries), each with the following keys
                speakers: list[int]
                    list of speaker numbers for the speakers talking in the segment
                start_time: float
                    start time of the segment in seconds
                end_time: float
                    end time of the segment in seconds
        video_file: VideoFile
            The video file to analyze.
        n_face_detect_batches: int
            The number of batches to use for identifyinng faces from a video file

        Returns
        -------
        list[dict]
            List of speaker segments (dictionaries), each with the following keys
                speakers: list[int]
                    list of speaker numbers for the speakers talking in the segment
                start_time: float
                    start time of the segment in seconds
                end_time: float
                    end time of the segment in seconds
                first_face_sec: float
                    the first second in the segment with a face
                found_face: bool
                    whether or not a face was found in the segment
        """
        for segment in segments:
            start_time = segment["start_time"]
            end_time = segment["end_time"]
            # start looking for faces an eighth of the way through the segment
            segment["first_face_sec"] = start_time + (end_time - start_time) / 8
            segment["found_face"] = False
            segment["is_analyzed"] = False

        batch_period = 1  # interval length to sample each segment at each iteration
        sample_period = 1  # interval between consecutive samples
        analyzed_segments = 0
        while analyzed_segments < len(segments):
            # select times to detect faces from
            detect_secs = []
            for segment in segments:
                if segment["is_analyzed"] is True:
                    continue
                segment_secs_left = segment["end_time"] - segment["first_face_sec"]
                num_samples = min(batch_period, segment_secs_left) // sample_period
                num_samples = max(1, int(num_samples))
                segment["num_samples"] = num_samples
                for i in range(num_samples):
                    detect_secs.append(segment["first_face_sec"] + i * sample_period)

            # detect faces
            if not detect_secs: # No times to detect, break to avoid infinite loop
                break

            n_batches = self._calc_n_batches(
                video_file=video_file,
                num_frames=len(detect_secs),
                face_detect_width=face_detect_width,
                n_face_detect_batches=n_face_detect_batches,
            )
            frames_per_batch = int(len(detect_secs) // n_batches + 1)
            face_detections = []
            for i in range(n_batches):
                current_detect_secs = detect_secs[
                        i
                        * frames_per_batch : min(
                            (i + 1) * frames_per_batch, len(detect_secs)
                        )
                    ]
                if not current_detect_secs: # Skip if no secs for this batch
                    continue
                frames = extract_frames(
                    video_file,
                    current_detect_secs,
                )
                face_detections += self._detect_faces(frames, face_detect_width)

            # check if any faces were found for each segment
            idx = 0
            for segment in segments:
                # segment already analyzed
                if segment["is_analyzed"] is True:
                    continue
                segment_idx = idx
                # check if any faces were found
                for _ in range(segment["num_samples"]):
                    if idx >= len(face_detections): # Boundary check for face_detections
                        break
                    faces = face_detections[idx]
                    if faces is not None:
                        segment["found_face"] = True
                        break
                    segment["first_face_sec"] += sample_period
                    idx += 1
                # update segment analyzation status
                is_analyzed = (
                    segment["found_face"] is True
                    or segment["first_face_sec"] >= segment["end_time"] - 0.25
                )
                if is_analyzed:
                    segment["is_analyzed"] = True
                    analyzed_segments += 1

                if segment_idx + segment["num_samples"] > idx : # if we broke early from inner loop
                    idx = segment_idx + segment["num_samples"]


            # increase period for next iteration
            batch_period = (batch_period + 3) * 2

        for segment in segments:
            if "num_samples" in segment: del segment["num_samples"] # Clean up temporary keys
            if "is_analyzed" in segment: del segment["is_analyzed"]

        return segments

    def _calc_n_batches(
        self,
        video_file: VideoFile,
        num_frames: int,
        face_detect_width: int,
        n_face_detect_batches: int,
    ) -> int:
        """
        Calculate the number of batches to use for extracting frames from a video file
        and detecting the face in each frame.

        Parameters
        ----------
        video_file: VideoFile
            The video file to analyze.
        num_frames: int
            The number of frames to analyze.
        face_detect_width: int
            The width to use for face detection.
        n_face_detect_batches: int
            Number of batches for GPU face detection in a video file.

        Returns
        -------
        int
            The number of batches to use.
        """
        if num_frames == 0:
            return 1 # Avoid division by zero if no frames

        # calculate memory needed to extract frames to CPU
        vid_height = video_file.get_height_pixels()
        vid_width = video_file.get_width_pixels()
        num_color_channels = 3
        bytes_per_frame = calc_img_bytes(vid_height, vid_width, num_color_channels)
        total_extract_bytes = num_frames * bytes_per_frame
        logging.debug(
            "Need {:.3f} GiB to extract (at most) {} frames".format(
                bytes_to_gibibytes(total_extract_bytes), num_frames
            )
        )

        # calculate memory needed to detect faces -> could be CPU or GPU
        downsample_factor = max(vid_width / face_detect_width, 1)
        face_detect_height = int(vid_height // downsample_factor)
        logging.debug(
            "Face detection dimensions: {}x{}".format(
                face_detect_height, face_detect_width
            )
        )
        bytes_per_frame_face = calc_img_bytes( # Different variable for face detect bytes
            face_detect_height, face_detect_width, num_color_channels
        )
        total_face_detect_bytes = num_frames * bytes_per_frame_face
        logging.debug(
            "Need {:.3f} GiB to detect faces from (at most) {} frames".format(
                bytes_to_gibibytes(total_face_detect_bytes), num_frames
            )
        )

        # calculate number of batches to use
        free_cpu_memory = pytorch.get_free_cpu_memory()
        if torch.cuda.is_available():
            n_extract_batches = int((total_extract_bytes // free_cpu_memory) + 1) if free_cpu_memory > 0 else num_frames
        else:
            # If no CUDA, CPU handles both extraction and face detection memory
            total_cpu_bytes = total_extract_bytes + total_face_detect_bytes
            n_extract_batches = int((total_cpu_bytes // free_cpu_memory) + 1) if free_cpu_memory > 0 else num_frames
            n_face_detect_batches = 0 # Ensure this is set if no CUDA

        # Ensure n_extract_batches is at least 1
        n_extract_batches = max(1, n_extract_batches)

        n_batches = int(max(n_extract_batches, n_face_detect_batches))
        # Ensure n_batches is at least 1 to prevent division by zero
        n_batches = max(1, n_batches)

        cpu_mem_per_batch = bytes_to_gibibytes(total_extract_bytes / n_batches)
        if n_face_detect_batches == 0 or not torch.cuda.is_available(): # Corrected this condition
            gpu_mem_per_batch = 0
        else:
            gpu_mem_per_batch = bytes_to_gibibytes(total_face_detect_bytes / n_batches)

        logging.debug(
            "Using {} batches to extract and detect frames. Need {:.3f} GiB of CPU "
            "memory per batch and {:.3f} GiB of GPU memory per batch".format(
                n_batches,
                cpu_mem_per_batch,
                gpu_mem_per_batch,
            )
        )
        return n_batches

    def _detect_faces(
        self,
        frames: list[np.ndarray],
        face_detect_width: int,
    ) -> list[np.ndarray]:
        """
        Detect faces in a list of frames.

        Parameters
        ----------
        frames: list[np.ndarray]
            The frames to detect faces in.
        face_detect_width: int
            The width to use for face detection.

        Returns
        -------
        list[np.ndarray]
            The face detections for each frame.
        """
        if len(frames) == 0:
            logging.debug("No frames to detect faces in.")
            return []

        # resize the frames
        logging.debug("Detecting faces in {} frames.".format(len(frames)))
        downsample_factor = max(frames[0].shape[1] / face_detect_width, 1)
        detect_height = int(frames[0].shape[0] / downsample_factor)
        resized_frames = []
        for frame in frames:
            resized_frame = cv2.resize(frame, (face_detect_width, detect_height))
            if torch.cuda.is_available():
                resized_frame = torch.from_numpy(resized_frame).to(
                    device="cuda", dtype=torch.uint8
                )
            resized_frames.append(resized_frame)

        # detect faces in batches
        if torch.cuda.is_available() and resized_frames: # Check if list is not empty
            try:
                resized_frames_tensor = torch.stack(resized_frames)
                detections, _ = self._face_detector.detect(resized_frames_tensor)
            except RuntimeError as e: # Catch potential CUDA errors
                logging.error(f"Face detection with CUDA failed: {e}. Falling back to CPU for this batch.")
                # Fallback to CPU for this batch
                cpu_resized_frames = [frame.cpu().numpy() if isinstance(frame, torch.Tensor) else frame for frame in resized_frames]
                detections, _ = self._face_detector.detect(cpu_resized_frames)

        elif resized_frames: # CPU path or if CUDA stacking failed
             detections, _ = self._face_detector.detect(resized_frames)
        else: # No frames to detect
            detections = []


        # detections are returned as numpy arrays regardless
        face_detections = []
        if detections is not None: # Detections can be None if no faces found in any frame
            for detection in detections:
                if detection is not None:
                    detection[detection < 0] = 0
                    detection = (detection * downsample_factor).astype(np.int16)
                face_detections.append(detection)

        logging.debug("Detected faces in {} frames processed.".format(len(frames)))
        return face_detections

    def _add_x_y_coords_to_each_segment(
        self,
        segments: list[dict],
        video_file: VideoFile,
        resize_width: int,
        resize_height: int,
        samples_per_segment: int,
        face_detect_width: int,
        n_face_detect_batches: int,
    ) -> list[dict]:
        """
        Add the x and y coordinates to resize each segment to.

        Parameters
        ----------
        segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
            first_face_sec: float
                the first second in the segment with a face
            found_face: bool
                whether or not a face was found in the segment
        video_file: VideoFile
            The video file to analyze.
        resize_width: int
            The width to resize the video to.
        resize_height: int
            The height to resize the video to.
        samples_per_segment: int
            Number of samples to take per segment for face detection.
        face_detect_width: int
            Width to resize the frames to for face detection.
        n_face_detect_batches: int
            Number of batches to process for face detection.


        Returns
        -------
        updated_segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
            x: int
                x-coordinate of the top left corner of the resized segment
            y: int
                y-coordinate of the top left corner of the resized segment
        """
        num_segments = len(segments)
        if num_segments == 0:
            return []

        num_frames = num_segments * samples_per_segment
        n_batches = self._calc_n_batches(
            video_file, num_frames, face_detect_width, n_face_detect_batches
        )
        segments_per_batch = int(num_segments // n_batches + 1)
        segments_with_xy_coords = []
        for i in range(n_batches):
            logging.debug("Analyzing batch {} of {}.".format(i, n_batches))
            cur_segments = segments[
                i
                * segments_per_batch : min((i + 1) * segments_per_batch, len(segments))
            ]
            if len(cur_segments) == 0:
                logging.debug("No segments left to analyze. (Batch {})".format(i))
                break
            segments_with_xy_coords += self._add_x_y_coords_to_each_segment_batch(
                segments=cur_segments,
                video_file=video_file,
                resize_width=resize_width,
                resize_height=resize_height,
                samples_per_segment=samples_per_segment,
                face_detect_width=face_detect_width,
            )
        return segments_with_xy_coords

    def _add_x_y_coords_to_each_segment_batch(
        self,
        segments: list[dict],
        video_file: VideoFile,
        resize_width: int,
        resize_height: int,
        samples_per_segment: int,
        face_detect_width: int,
    ) -> list[dict]:
        """
        Add the x and y coordinates to resize each segment to for a given batch.

        Parameters
        ----------
        segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
            first_face_sec: float
                the first second in the segment with a face
            found_face: bool
                whether or not a face was found in the segment
        video_file: VideoFile
            The video file to analyze.
        resize_width: int
            The width to resize the video to.
        resize_height: int
            The height to resize the video to.
        samples_per_segment: int
            Number of samples to take per segment for analyzing face locations.
        face_detect_width: int
            Width to which the video frames are resized for face detection.

        Returns
        -------
        updated_segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
            x: int
                x-coordinate of the top left corner of the resized segment
            y: int
                y-coordinate of the top left corner of the resized segment
        """
        fps = video_file.get_frame_rate()
        if fps <= 0: # Avoid division by zero if fps is invalid
            logging.error("Invalid FPS detected. Cannot process segments.")
            for segment in segments: # Ensure x,y are added to prevent key errors later
                segment["x"] = 0
                segment["y"] = 0
            return segments


        # define frames to analyze from each segment
        detect_secs = []
        for segment in segments:
            if segment.get("found_face", False) is False: # Use .get for safety
                segment["num_samples"] = 0 # Ensure key exists
                continue
            # define interval over which to analyze faces
            end_time = segment["end_time"]
            first_face_sec = segment["first_face_sec"]
            analyze_end_time = end_time - (end_time - first_face_sec) / 8

            # Ensure first_face_sec is not past analyze_end_time
            if first_face_sec >= analyze_end_time:
                segment["num_samples"] = 0
                detect_secs.append(first_face_sec) # Sample at least one frame if possible
                continue

            frames_left = int((analyze_end_time - first_face_sec) * fps + 1)
            num_samples = min(frames_left, samples_per_segment)
            segment["num_samples"] = num_samples

            if num_samples <=0: # if no samples can be taken
                continue

            detect_secs.append(first_face_sec)
            if num_samples > 1: # only sample if more than one sample is needed
                # Corrected range for np.random.choice to be non-negative
                # frames_left_for_choice is frames_left -1 because we already added first_face_sec
                frames_left_for_choice = max(0, frames_left -1)
                if frames_left_for_choice > 0 :
                    sample_frames = np.sort(
                        np.random.choice(range(1, frames_left_for_choice + 1), min(num_samples - 1, frames_left_for_choice), replace=False)
                    )
                    for sample_frame in sample_frames:
                        detect_secs.append(first_face_sec + sample_frame / fps)

        face_detections = []
        if detect_secs:
            logging.debug("Extracting {} frames".format(len(detect_secs)))
            frames = extract_frames(video_file, detect_secs)
            logging.debug("Extracted {} frames".format(len(frames))) # Log actual number extracted
            if frames: # Only detect if frames were successfully extracted
                face_detections = self._detect_faces(frames, face_detect_width)
            else: # If extract_frames returned empty (e.g. error or no valid times)
                logging.warning("No frames extracted for face detection in this batch.")
        else:
            logging.debug("No detection times identified for segments in this batch.")


        logging.debug("Calculating ROI for {} segments.".format(len(segments)))
        # find roi for each segment
        idx = 0
        for segment in segments:
            # find segment roi
            if segment.get("found_face", False) is True and segment.get("num_samples", 0) > 0 :
                # Check if enough face_detections are available
                if idx + segment["num_samples"] <= len(face_detections):
                    roi = self._calc_segment_roi(
                        frames=frames[idx : idx + segment["num_samples"]], # Ensure frames list is also sliced correctly
                        face_detections=face_detections[idx : idx + segment["num_samples"]],
                    )
                else: # Not enough detections, use default ROI
                    logging.warning(f"Not enough face detections for segment {segment}, using default ROI.")
                    roi = Rect(
                        x=(video_file.get_width_pixels()) // 4,
                        y=(video_file.get_height_pixels()) // 4,
                        width=(video_file.get_width_pixels()) // 2,
                        height=(video_file.get_height_pixels()) // 2,
                    )
                idx += segment.get("num_samples",0) # Use .get for safety
                if "num_samples" in segment: del segment["num_samples"]
            else:
                logging.debug("Using default ROI for segment {}".format(segment))
                roi = Rect(
                    x=(video_file.get_width_pixels()) // 4,
                    y=(video_file.get_height_pixels()) // 4,
                    width=(video_file.get_width_pixels()) // 2,
                    height=(video_file.get_height_pixels()) // 2,
                )
            if "found_face" in segment: del segment["found_face"] # Clean up temporary keys
            if "first_face_sec" in segment: del segment["first_face_sec"]

            # add crop coordinates to segment
            crop = self._calc_crop(roi, resize_width, resize_height)
            segment["x"] = int(crop.x)
            segment["y"] = int(crop.y)
        logging.debug("Calculated ROI for {} segments.".format(len(segments)))

        return segments

    def _calc_segment_roi(
        self,
        frames: list[np.ndarray],
        face_detections: list[np.ndarray],
    ) -> Rect:
        """
        Find the region of interest (ROI) for a given segment.

        Parameters
        ----------
        frames: np.ndarray
            The frames to analyze.
        face_detections: np.ndarray
            The face detection outputs for each frame

        Returns
        -------
        Rect
            The region of interest (ROI) for the segment.
        """
        segment_roi = None

        # preprocessing for kmeans
        bounding_boxes: list[np.ndarray] = []
        k = 0
        for face_detection in face_detections:
            if face_detection is None:
                continue
            k = max(k, len(face_detection))
            for bounding_box in face_detection:
                bounding_boxes.append(bounding_box)

        # no faces detected
        if k == 0 or not bounding_boxes: # Added check for empty bounding_boxes
            # Default ROI if no faces: center of the frame (or a reasonable default)
            # This case should ideally be handled by the found_face check before calling this
            logging.warning("No faces detected in segment for ROI calculation. Returning default ROI.")
            if frames: # if we have frame info to get dimensions
                 return Rect(frames[0].shape[1]//4, frames[0].shape[0]//4, frames[0].shape[1]//2, frames[0].shape[0]//2)
            else: # Absolute fallback, should ideally not be reached if found_face logic is robust
                 return Rect(0,0,100,100) # Arbitrary small rect

        bounding_boxes = np.stack(bounding_boxes)

        # single face detected
        if k == 1:
            box = np.mean(bounding_boxes, axis=0).astype(np.int16)
            x1, y1, x2, y2 = box
            segment_roi = Rect(x1, y1, x2 - x1, y2 - y1)
            return segment_roi

        # use kmeans to group the same bounding boxes together
        # n_init='auto' is preferred for scikit-learn >= 1.4
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init='auto', random_state=0).fit(
            bounding_boxes
        )
        bounding_box_labels = kmeans.labels_
        bounding_box_groups: list[list[dict]] = [[] for _ in range(k)]
        kmeans_idx = 0
        for i, face_detection in enumerate(face_detections):
            if face_detection is None:
                continue
            for bounding_box in face_detection:
                if kmeans_idx >= len(bounding_box_labels): # Safety break
                    break
                assert np.sum(bounding_box < 0) == 0
                bounding_box_label = bounding_box_labels[kmeans_idx]
                bounding_box_groups[bounding_box_label].append(
                    {"bounding_box": bounding_box, "frame": i}
                )
                kmeans_idx += 1
            if kmeans_idx >= len(bounding_box_labels): break


        # find the face who's mouth moves the most
        max_mouth_movement = 0
        for i, bounding_box_group in enumerate(bounding_box_groups):
            if not bounding_box_group: # Skip empty groups
                continue
            mouth_movement, roi = self._calc_mouth_movement(bounding_box_group, frames)
            if mouth_movement > max_mouth_movement:
                max_mouth_movement = mouth_movement
                segment_roi = roi

        # no mouth movement detected -> choose face with the most frames
        if segment_roi is None:
            logging.debug("No mouth movement detected for segment.")
            max_frames_in_group = 0 # Changed from max_frames
            for bounding_box_group in bounding_box_groups:
                if not bounding_box_group: continue # Skip empty
                if len(bounding_box_group) > max_frames_in_group:
                    max_frames_in_group = len(bounding_box_group)
                    avg_box = np.array([0.0,0.0,0.0,0.0]) # Use float for accumulation
                    for bounding_box_data in bounding_box_group:
                        avg_box += bounding_box_data["bounding_box"]
                    avg_box = avg_box / len(bounding_box_group)
                    avg_box = avg_box.astype(np.int16)
                    segment_roi = Rect(
                        avg_box[0],
                        avg_box[1],
                        max(1, avg_box[2] - avg_box[0]), # Ensure width/height are at least 1
                        max(1, avg_box[3] - avg_box[1]),
                    )
        # If still no ROI (e.g. all groups were empty, though checked above)
        if segment_roi is None:
            logging.warning("Could not determine segment ROI. Defaulting to center crop.")
            if frames:
                 return Rect(frames[0].shape[1]//4, frames[0].shape[0]//4, frames[0].shape[1]//2, frames[0].shape[0]//2)
            else:
                 return Rect(0,0,100,100)


        return segment_roi

    def _calc_mouth_aspect_ratio(self, face: np.ndarray) -> float:
        """
        Calculate the mouth aspect ratio using dlib shape predictor.

        Parameters
        ----------
        face: np.ndarray
            Pytorch array of a face

        Returns
        -------
        mar: float
            The mouth aspect ratio.
        """
        if face is None or face.size == 0: return None # Handle empty face array
        results = self._face_mesher.process(face)
        if results.multi_face_landmarks is None:
            return None

        landmarks = []
        for landmark in results.multi_face_landmarks[0].landmark:
            landmarks.append([landmark.x, landmark.y])
        landmarks = np.array(landmarks)
        landmarks[:, 0] *= face.shape[1]
        landmarks[:, 1] *= face.shape[0]

        # inner lip
        upper_lip = landmarks[[95, 88, 178, 87, 14, 317, 402, 318, 324], :]
        lower_lip = landmarks[[191, 80, 81, 82, 13, 312, 311, 310, 415], :]
        avg_mouth_height = np.mean(np.abs(upper_lip - lower_lip))
        mouth_width = np.sum(np.abs(landmarks[[308], :] - landmarks[[78], :]))
        if mouth_width == 0: return 0.0 # Avoid division by zero
        mar = avg_mouth_height / mouth_width

        return mar

    def _calc_crop(
        self,
        roi: Rect,
        resize_width: int,
        resize_height: int,
    ) -> Rect:
        """
        Calculate the crop given the ROI location.

        Parameters
        ----------
        roi: Rect
            The rectangle containing the region of interest (ROI).

        Returns
        -------
        Rect
            The crop rectangle.
        """
        roi_x_center = roi.x + roi.width // 2
        roi_y_center = roi.y + roi.height // 2
        crop = Rect(
            x=max(roi_x_center - (resize_width // 2), 0),
            y=max(roi_y_center - (resize_height // 2), 0),
            width=resize_width,
            height=resize_height,
        )
        return crop

    def _merge_identical_segments(
        self,
        segments: list[dict],
        video_file: VideoFile,
    ) -> list[dict]:
        """
        Merge identical segments that are next to each other.

        Parameters
        ----------
        segments: list[dict]
            speakers: list[int]
                the speaker labels of the speakers talking in the segment
            start_time: float
                the start time of the segment
            end_time: float
                the end time of the segment
            x: int
                x-coordinate of the top left corner of the resized segment
            y: int
                y-coordinate of the top left corner of the resized segment
        video_file: VideoFile
            The video file that the segments are from

        Returns
        -------
        list[dict]
            The merged segments.
        """
        if not segments: # Handle empty list
            return []

        idx = 0
        max_position_difference_ratio = 0.04
        video_width = video_file.get_width_pixels()
        video_height = video_file.get_height_pixels()

        # Ensure video_width and video_height are not zero to avoid DivisionByZeroError
        if video_width == 0: video_width = 1
        if video_height == 0: video_height = 1


        # Iterate up to len(segments) - 2 because we look at segments[idx+1]
        while idx < len(segments) -1:
            cur_x = segments[idx]["x"]
            next_x = segments[idx + 1]["x"]
            x_diff = abs(cur_x - next_x)

            # Check if x coordinates are similar
            if (x_diff / video_width) < max_position_difference_ratio:
                same_x = True
                # Don't average immediately, only if y is also same
            else:
                same_x = False

            curr_y = segments[idx]["y"]
            next_y = segments[idx + 1]["y"]
            y_diff = abs(curr_y - next_y)

            # Check if y coordinates are similar
            if (y_diff / video_height) < max_position_difference_ratio:
                same_y = True
                # Don't average immediately
            else:
                same_y = False

            # If both x and y are similar, and speakers are the same, merge segments
            if same_x and same_y and segments[idx]["speakers"] == segments[idx+1]["speakers"]:
                # Average the coordinates for the merged segment
                segments[idx]["x"] = int((cur_x + next_x) // 2)
                segments[idx]["y"] = int((curr_y + next_y) // 2)
                segments[idx]["end_time"] = segments[idx + 1]["end_time"]
                segments.pop(idx + 1) # Remove the merged segment
                # Do not increment idx, so the new segments[idx] can be compared with the next one
            else:
                idx += 1 # Move to next segment only if no merge occurred
        return segments

    def cleanup(self) -> None:
        """
        Remove the face detector from memory and explicity free up GPU memory.
        """
        if hasattr(self, '_face_detector') and self._face_detector is not None:
            del self._face_detector
            self._face_detector = None
        if hasattr(self, '_face_mesher') and self._face_mesher is not None:
            del self._face_mesher
            self._face_mesher = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.debug("Resizer resources cleaned up.")
