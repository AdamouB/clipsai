"""
Editing media files with ffmpeg.
"""
# standard library imports
import logging
import subprocess
import os
import uuid
import tempfile

# current package imports
from .exceptions import MediaEditorError
from .audio_file import AudioFile
from .audiovideo_file import AudioVideoFile
from .image_file import ImageFile
from .media_file import MediaFile
from .temporal_media_file import TemporalMediaFile
from .video_file import VideoFile

# local imports
from clipsai.filesys.manager import FileSystemManager
from clipsai.utils.conversions import seconds_to_hms_time_format
from clipsai.utils.type_checker import TypeChecker


# ffmpeg return code of 0 means success; any other (positive) integer means failure
SUCCESS = 0


class MediaEditor:
    """
    A class to edit media files using ffmpeg.
    """

    def __init__(self) -> None:
        """
        Initialize FfmpegEditor

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._file_system_manager = FileSystemManager()
        self._type_checker = TypeChecker()

    def _run_ffmpeg_command(
        self,
        ffmpeg_command_list: list[str],
        success_log_message: str,
        failure_log_message: str,
    ) -> bool:
        """
        Runs an ffmpeg command using subprocess.

        Parameters
        ----------
        ffmpeg_command_list: list[str]
            The ffmpeg command and its arguments as a list of strings.
        success_log_message: str
            Message to log on successful execution.
        failure_log_message: str
            Message to log on failed execution.

        Returns
        -------
        bool
            True if the command was successful, False otherwise.
        """
        logging.debug("ffmpeg_command: %s", ffmpeg_command_list)
        result = subprocess.run(
            ffmpeg_command_list,
            capture_output=True,
            text=True,
        )

        log_output_message = (
            "Terminal return code: '{}'\n"
            "Output: '{}'\n"
            "Err Output: '{}'\n"
            "".format(result.returncode, result.stdout, result.stderr)
        )

        if result.returncode == SUCCESS:
            logging.debug(success_log_message)
            logging.debug(log_output_message)
            return True
        else:
            logging.error(failure_log_message)
            logging.error(log_output_message)
            return False

    def trim(
        self,
        media_file: TemporalMediaFile,
        start_time: float,
        end_time: float,
        trimmed_media_file_path: str,
        overwrite: bool = True,
        video_codec: str = "copy",
        audio_codec: str = "copy",
        crf: str = "23",
        preset: str = "medium",
        num_threads: str = "0",
        crop_width: int = None,
        crop_height: int = None,
        crop_x: int = None,
    ) -> TemporalMediaFile or None:
        """
        Trims and potentially resizes a temporal media file (audio or video) into a
        new, trimmed media file

        - trimmed_media_file_path is overwritten if already exists

        Parameters
        ----------
        media_file: TemporalMediaFile
            the media file to trim
        start_time: float
            the time in seconds the trimmed media file begins
        end_time: float
            the time in seconds the trimmed media file ends
        trimmed_media_file_path: str
            absolute path to store the trimmed media file
        overwrite: bool
            Overwrites 'trimmed_media_file_path' if True; does not overwrite if False
        video_codec: str
            compression and decompression software for the video (libx264)
        audio_codec: str
            compression and decompression sfotware for the audio (aac)
        crf: str
            constant rate factor - an encoding mode that adjusts the file data rate up
            or down to achieve a selected quality level rather than a specific data
            rate. CRF values range from 0 to 51, with lower numbers delivering higher
            quality scores
        preset: str
            the encoding speed to compression ratio. A slower preset will provide
            better compression (compression is quality per filesize)
        num_threads: str
            the number of threads to use for encoding
        crop_x: int, optional
            x-coordinate of the top left corner of the crop area.
            none if no resizing
        crop_y: int, optional
            y-coordinate of the top left corner of the crop area.
            none if no resizing
        crop_width: int, optional
            Width of the crop area.
            none if no resizing
        crop_height: int, optional
            Height of the crop area.
            none if no resizing

        Returns
        -------
        MediaFile or None
            the trimmed media as a MediaFile object if successful; None if unsuccessful

        Raises
        ------
        MediaEditorError: start_time < 0
        MediaEditorError: end_time < 0
        MediaEditorError: start_time > end_time
        MediaEditorError: start_time > media_file's duration
        MediaEditorError: end_time > media_file's duration
        """
        self.assert_valid_media_file(media_file, TemporalMediaFile)
        if overwrite is True:
            self._file_system_manager.assert_parent_dir_exists(
                MediaFile(trimmed_media_file_path)
            )
        else:
            self._file_system_manager.assert_valid_path_for_new_fs_object(
                trimmed_media_file_path
            )
        self._file_system_manager.assert_paths_not_equal(
            media_file.path,
            trimmed_media_file_path,
            "media_file path",
            "trimmed_media_file_path",
        )
        self._assert_valid_trim_times(media_file, start_time, end_time)

        duration_secs = end_time - start_time
        start_time_hms_time_format = seconds_to_hms_time_format(start_time)
        duration_hms_time_format = seconds_to_hms_time_format(duration_secs)

        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-ss",
            start_time_hms_time_format,
            "-t",
            duration_hms_time_format,
            "-i",
            media_file.path,
            "-c:v",
            video_codec,
            "-preset",
            preset,
            "-c:a",
            audio_codec,
            "-map",
            "0",
            "-crf",
            crf,
            "-threads",
            num_threads,
        ]

        if crop_height is not None and crop_width is not None and crop_x is not None:
            logging.debug("Trim with resizing.")
            original_height = int(media_file.get_stream_info("v", "height"))
            crop_y = max(original_height // 2 - crop_height // 2, 0)
            crop_vf = "crop={width}:{height}:{x}:{y}".format(
                width=crop_width, height=crop_height, x=crop_x, y=crop_y
            )
            ffmpeg_command.extend(["-vf", crop_vf])

        ffmpeg_command.append(trimmed_media_file_path)

        success_msg = "Successfully trimmed media file '{}' to '{}'.".format(
            media_file.path, trimmed_media_file_path
        )
        failure_msg = "Failed to trim media file '{}' to '{}'.".format(
            media_file.path, trimmed_media_file_path
        )

        success = self._run_ffmpeg_command(ffmpeg_command, success_msg, failure_msg)

        if success:
            trimmed_media_file = self._create_media_file_of_same_type(
                trimmed_media_file_path, media_file
            )
            trimmed_media_file.assert_exists()
            return trimmed_media_file
        else:
            return None

    def copy_temporal_media_file(
        self,
        media_file: TemporalMediaFile,
        copied_media_file_path: str,
        overwrite: bool = True,
        video_codec: str = "copy",
        audio_codec: str = "copy",
        crf: str = "23",
        preset: str = "medium",
        num_threads: str = "0",
    ) -> TemporalMediaFile or None:
        """
        Creates a copy of a temporal media file (audio or video)

        - 'copied_media_file_path' is overwritten if already exists

        Parameters
        ----------
        media_file: TemporalMediaFile
            absolute path to the media file to copy
        copied_media_file_path: str
            absolute path to copy the media file to
        overwrite: bool
            Overwrites 'copied_media_file_path' if True; does not overwrite if False
        video_codec: str
            compression and decompression software for the video (libx264)
        audio_codec: str
            compression and decompression sfotware for the audio (aac)
        crf: str
            constant rate factor - an encoding mode that adjusts the file data rate up
            or down to achieve a selected quality level rather than a specific data
            rate. CRF values range from 0 to 51, with lower numbers delivering higher
            quality scores
        preset: str
            the encoding speed to compression ratio. A slower preset will provide
            better compression (compression is quality per filesize)
        num_threads: str
            the number of threads to use for encoding

        Returns
        -------
        MediaFile or None
            the copied media as a MediaFile object if successful; None if unsuccessful

        Raises
        ------
        MediaEditorError: media_file's duration could not be found
        """
        self.assert_valid_media_file(media_file, TemporalMediaFile)

        duration = media_file.get_duration()
        if duration == -1:
            msg = "Can't retrieve duration from media file '{}'".format(media_file.path)
            logging.error(msg)
            raise MediaEditorError(msg)

        copied_media_file = self.trim(
            media_file,
            0,
            duration,
            copied_media_file_path,
            overwrite,
            video_codec,
            audio_codec,
            crf,
            preset,
            num_threads,
        )
        if copied_media_file is None:
            logging.error("Copying media file '{}' to '{}' was unsuccessful." "".format(
                media_file.path, copied_media_file_path
            ))
            return None
        else:
            return copied_media_file

    def transcode(
        self,
        media_file: TemporalMediaFile,
        transcoded_media_file_path: str,
        video_codec: str,
        audio_codec: str,
        crf: str = "23",
        preset: str = "medium",
        overwrite: bool = True,
        num_threads: str = "0",
    ) -> TemporalMediaFile or None:
        """
        Transcodes media file (audio or video) to the specified codecs

        - 'transcoded_media_file_path' is overwritten if already exists

        Parameters
        ----------
        media_file: TemporalMediaFile
            absolute path to the media file to transcode
        transcoded_media_file_path: str
            absolute path to store the transcoded media file
        overwrite: bool
            Overwrites 'transcoded_media_file_path' if True; does not overwrite if False
        video_codec: str
            compression and decompression software for the video (libx264)
        audio_codec: str
            compression and decompression sfotware for the audio (aac)
        crf: str
            constant rate factor - an encoding mode that adjusts the file data rate up
            or down to achieve a selected quality level rather than a specific data
            rate. CRF values range from 0 to 51, with lower numbers delivering higher
            quality scores
        preset: str
            the encoding speed to compression ratio. A slower preset will provide
            better compression (compression is quality per filesize)
        num_threads: str
            the number of threads to use for encoding

        Returns
        -------
        MediaFile or None
            the transcoded media as a MediaFile object if successful; None if
            unsuccessful
        """
        return self.copy_temporal_media_file(
            media_file,
            transcoded_media_file_path,
            overwrite,
            video_codec,
            audio_codec,
            crf,
            preset,
            num_threads,
        )

    def watermark_and_crop_video(
        self,
        video_file: VideoFile,
        watermark_file: ImageFile,
        watermarked_video_file_path: str,
        size_dim: str,
        watermark_to_video_ratio_size_dim: float,
        x: str,
        y: str,
        opacity: float,
        overwrite: bool = True,
        start_time: float = None,
        end_time: float = None,
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        crf: str = "23",
        preset: str = "medium",
        num_threads: str = "0",
        crop_x: int = None,
        crop_y: int = None,
        crop_width: int = None,
        crop_height: int = None,
    ) -> VideoFile or None:
        """
        Watermark a video

        - 'watermarked_video_file_path' is overwritten if already exists
        - https://www.bannerbear.com/blog/how-to-add-watermark-to-videos-using-ffmpeg/
        #basic-command

        Parameters
        ----------
        video_file: VideoFile
            the video file to watermark
        watermark_file: ImageFile
            the image file to watermark the video with
        watermarked_video_file_path: str
            absolute path to store the watermarked video
        size_dim: str
            the dimension (height or width) to size the watermark with respect to the
            video. Needs to be 'h' (height) or 'w' (width)
        watermark_to_video_ratio_size_dim: float
            the size ratio of the watermark to the video along the chosen size
            dimension (width or height). Needs to be greater than zero
        x: str
            x position of watermark where the origin of both the video and watermark
            are the top left corner and x increases as you move right
                main_w: width of the video
                overlay_w: width of the watermark
        y: str
            y position of watermark where the origin of both the video and watermark
            are the top left corner and y increases as you move down
                main_h: height of the video
                overlay_h: height of the watermark
        opacity: float
            opacity of the watermark on the video; must be between 0 and 1
        overwrite: bool
            Overwrites 'watermarked_video_file_path' if True; does not overwrite if
            False
        start_time: float
            the time in seconds the trimmed media file begins
        end_time: float
            the time in seconds the trimmed media file ends
        video_codec: str
            compression and decompression software for the video (libx264)
        audio_codec: str
            compression and decompression sfotware for the audio (aac)
        crf: str
            constant rate factor - an encoding mode that adjusts the file data rate up
            or down to achieve a selected quality level rather than a specific data
            rate. CRF values range from 0 to 51, with lower numbers delivering higher
            quality scores
        preset: str
            the encoding speed to compression ratio. A slower preset will provide
            better compression (compression is quality per filesize)
        num_threads: str
            the number of threads to use
        crop_x: int, optional
            x-coordinate of the top left corner of the crop area.
            none if no resizing
        crop_y: int, optional
            y-coordinate of the top left corner of the crop area.
            none if no resizing
        crop_width: int, optional
            Width of the crop area.
            none if no resizing
        crop_height: int, optional
            Height of the crop area.
            none if no resizing

        Positioning Examples
        --------------------
        - top left corner: x=0 y=0
        - top right corner: x=main_w-overlay_w y=0
        - bottom left corner: x=0 y=main_h-overlay_h
        - bottom right corner: x=main_w-overlay_w y=main_h-overlay_h
        - middle of video: x=(main_w-overlay_w)/2 y=(main_h-overlay_h)/2

        Returns
        -------
        VideoFile
            the watermarked and possibly resized video as a VideoFile
            object if successful; None if unsuccessful

        Raises
        ------
        MediaEditorError: size_dim is not 'h' or 'w'
        MediaEditorError: watermark_to_video_ratio_size_dim <= 0
        MediaEditorError: opacity < 0 or opacity > 1
        """
        self.assert_valid_media_file(video_file, VideoFile)
        self.assert_valid_media_file(watermark_file, ImageFile)
        if overwrite is True:
            self._file_system_manager.assert_parent_dir_exists(
                MediaFile(watermarked_video_file_path)
            )
        else:
            self._file_system_manager.assert_valid_path_for_new_fs_object(
                watermarked_video_file_path
            )
        # Assert paths are not equal
        paths_to_check = {
            ("video_file path", video_file.path): [
                ("watermark_file path", watermark_file.path),
                ("watermarked_video_file_path", watermarked_video_file_path),
            ],
            ("watermark_file path", watermark_file.path): [
                ("watermarked_video_file_path", watermarked_video_file_path),
            ],
        }
        for (name1, path1), checks in paths_to_check.items():
            for name2, path2 in checks:
                self._file_system_manager.assert_paths_not_equal(path1, path2, name1, name2)


        if size_dim not in ["h", "w"]:
            raise MediaEditorError(f"size_dim must be one of 'h' or 'w', not '{size_dim}'")
        if watermark_to_video_ratio_size_dim <= 0:
            raise MediaEditorError(
                "watermark_to_video_ratio_size_dim must be greater than zero, "
                f"not '{watermark_to_video_ratio_size_dim}'"
            )
        if not (0 <= opacity <= 1):
            raise MediaEditorError(f"Opacity must be between 0 and 1, not '{opacity}'")

        _start_time = start_time if start_time is not None else 0.0
        _end_time = end_time if end_time is not None else video_file.get_duration()
        self._assert_valid_trim_times(video_file, _start_time, _end_time)

        duration_secs = _end_time - _start_time
        start_time_hms_time_format = seconds_to_hms_time_format(_start_time)
        duration_hms_time_format = seconds_to_hms_time_format(duration_secs)

        filter_complex_parts = []
        input_video_label = "[0:v]" # Use stream specifier for clarity

        if crop_x is not None and crop_y is not None and crop_width is not None and crop_height is not None:
            filter_complex_parts.append(
                f"{input_video_label}crop={crop_width}:{crop_height}:{crop_x}:{crop_y}[cropped]"
            )
            input_video_label = "[cropped]" # Output of crop becomes input for next

        filter_complex_parts.append(
            f"[1:v]format=rgba,colorchannelmixer=aa={opacity}[logo_alpha]"
        )
        filter_complex_parts.append(
             f"[logo_alpha]{input_video_label}scale2ref=oh*mdar:ih*{watermark_to_video_ratio_size_dim}"
             if size_dim == 'h' else
             f"[logo_alpha]{input_video_label}scale2ref=ow*mdar:iw*{watermark_to_video_ratio_size_dim}"
             + "[logo_scaled][video_ref]"
        )
        filter_complex_parts.append(f"[video_ref][logo_scaled]overlay=x={x}:y={y}")

        filter_complex = ";".join(filter_complex_parts)

        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-ss", start_time_hms_time_format,
            "-t", duration_hms_time_format,
            "-i", video_file.path,
            "-i", watermark_file.path,
            "-filter_complex", filter_complex,
            "-c:v", video_codec,
            "-preset", preset,
            "-c:a", audio_codec, # Ensure audio from original video is mapped
            "-map", "[0:a]?", # Map audio from the first input (video_file) if it exists
            "-crf", crf,
            "-threads", num_threads,
            watermarked_video_file_path,
        ]

        success_msg = "Successfully watermarked video '{}' to '{}'.".format(
            video_file.path, watermarked_video_file_path
        )
        failure_msg = "Failed to watermark video '{}' to '{}'.".format(
            video_file.path, watermarked_video_file_path
        )

        success = self._run_ffmpeg_command(ffmpeg_command, success_msg, failure_msg)

        if success:
            watermarked_video = VideoFile(watermarked_video_file_path)
            watermarked_video.assert_exists()
            return watermarked_video
        else:
            return None


    def watermark_corner_of_video(
        self,
        video_file: VideoFile,
        watermark_file: ImageFile,
        watermarked_video_file_path: str,
        watermark_to_video_ratio_along_smaller_video_dimension: float,
        corner: str,
        opacity: float,
        overwrite: bool = True,
        start_time: float = None,
        end_time: float = None,
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        crf: str = "23",
        preset: str = "medium",
        num_threads: str = "0",
        crop_x: int = None,
        crop_width: int = None,
        crop_height: int = None,
    ) -> VideoFile or None:
        """
        Watermark 'video_file' with 'watermark_file' in the chosen corner such that the
        watermark:video ratio is 0.25 along the shortest video dimension (height or
        width)

        Parameters
        ----------
        video_file: VideoFile
            the video file to watermark
        watermark_file: ImageFile
            the image file to watermark the video with
        watermarked_video_file_path: str
            absolute path to store the watermarked video
        watermark_to_video_ratio_along_smaller_video_dimension: float
            the ratio of the watermark size relative to the video size along the
            smaller of video's two size dimensions (width or height)
        corner: str
            the corner you want the watermark to be in. One of: "bottom_left",
            "bottom_right", "top_left", "top_right"
        opacity: float
            opacity of the src_img_file_path watermark on the video; must be between
            zero and one
        overwrite: bool
            Overwrites 'watermarked_video_file_path' if True; does not overwrite if
            False
        start_time: float
            the time in seconds the trimmed media file begins
        end_time: float
            the time in seconds the trimmed media file ends
        video_codec: str
            compression and decompression software for the video (libx264)
        audio_codec: str
            compression and decompression sfotware for the audio (aac)
        crf: str
            constant rate factor - an encoding mode that adjusts the file data rate up
            or down to achieve a selected quality level rather than a specific data
            rate. CRF values range from 0 to 51, with lower numbers delivering higher
            quality scores
        preset: str
            the encoding speed to compression ratio. A slower preset will provide
            better compression (compression is quality per filesize)
        num_threads: str
            the number of threads to use
        crop_x: int, optional
            x-coordinate of the top left corner of the crop area,
            none if no resizing
        crop_width: int, optional
            width of the crop area, none if no resizing
        crop_height: int, optional
            height of the crop area, none if no resizing

        Returns
        -------
        watermarked_video: Video
            Returns the watermarked and possibly cropped Video object
        """
        self.assert_valid_media_file(video_file, VideoFile)

        original_height = int(video_file.get_stream_info("v", "height"))
        original_width = int(video_file.get_stream_info("v", "width"))
        crop_y = None # Define crop_y to ensure it's always available

        if crop_height is not None and crop_width is not None and crop_x is not None: # Added crop_x check
            logging.debug("Watermark with resizing.")
            # Calculate crop_y based on centering the crop_height within original_height
            # This logic assumes crop_x is also provided for a complete crop definition
            crop_y = max(original_height // 2 - crop_height // 2, 0)

        # Define x and y based on whether cropping is applied
        main_w_ref = "W" if crop_width else "main_w"
        main_h_ref = "H" if crop_height else "main_h"

        corner_commands = {
            "bottom_left": {"x": "0", "y": f"{main_h_ref}-overlay_h"},
            "bottom_right": {"x": f"{main_w_ref}-overlay_w", "y": f"{main_h_ref}-overlay_h"},
            "top_left": {"x": "0", "y": "0"},
            "top_right": {"x": f"{main_w_ref}-overlay_w", "y": "0"},
        }

        if corner not in corner_commands:
            raise MediaEditorError(f"Invalid corner: {corner}. Must be one of {list(corner_commands.keys())}")

        # Determine size_dim based on the smaller dimension of the video AFTER potential cropping
        effective_width = crop_width if crop_width else original_width
        effective_height = crop_height if crop_height else original_height

        size_dim = "w" if effective_width < effective_height else "h"

        return self.watermark_and_crop_video(
            video_file=video_file,
            watermark_file=watermark_file,
            watermarked_video_file_path=watermarked_video_file_path,
            size_dim=size_dim,
            watermark_to_video_ratio_size_dim=watermark_to_video_ratio_along_smaller_video_dimension,
            x=corner_commands[corner]["x"],
            y=corner_commands[corner]["y"],
            opacity=opacity,
            overwrite=overwrite,
            start_time=start_time,
            end_time=end_time,
            video_codec=video_codec,
            audio_codec=audio_codec,
            crf=crf,
            preset=preset,
            num_threads=num_threads,
            crop_x=crop_x,
            crop_y=crop_y, # Pass calculated crop_y
            crop_width=crop_width,
            crop_height=crop_height,
        )

    def merge_audio_and_video(
        self,
        video_file: VideoFile,
        audio_file: AudioFile,
        merged_video_file_path: str,
        overwrite: bool = True,
        video_codec: str = "copy",
        audio_codec: str = "copy",
    ) -> VideoFile or None:
        """
        Merges an audio-only file and video-only file into a single video file

        - 'dest_merged_video_file_path' is overwritten if already exists
        - MoviePy reference:
        https://github.com/Zulko/moviepy/blob/master/moviepy/video/io/ffmpeg_tools.py

        Parameters
        ----------
        video_file: VideoFile
            the video file to merge
        audio_file: AudioFile
            the audio file to merge
        merged_video_file_path: str
            absolute path to store the merged video file
        overwrite: bool
            Overwrites 'audio_file_path' if True; does not overwrite if False
        audio_codec: str
            compression and decompression software for the audio (aac)
        video_codec: str
            compression and decompression software for the video (libx264)

        Returns
        -------
        VideoFile or None
            the merged video as a VideoFile object if successful; None if unsuccessful
        """
        self.assert_valid_media_file(audio_file, AudioFile)
        self.assert_valid_media_file(video_file, VideoFile)
        if overwrite is True:
            self._file_system_manager.assert_parent_dir_exists(
                MediaFile(merged_video_file_path)
            )
        else:
            self._file_system_manager.assert_valid_path_for_new_fs_object(
                merged_video_file_path
            )
        self._file_system_manager.assert_paths_not_equal(
            video_file.path,
            merged_video_file_path,
            "video_file path",
            "merged_video_file_path",
        )
        self._file_system_manager.assert_paths_not_equal(
            audio_file.path,
            merged_video_file_path,
            "audio_file path",
            "merged_video_file_path",
        )

        max_duration_diff = 3
        duration_diff = abs(video_file.get_duration() - audio_file.get_duration())
        if duration_diff > max_duration_diff:
            msg = (
                "Audio and video files cannot be merge. Audio file '{}' and video file "
                "'{}' have a duration difference of more than {} seconds."
                "".format(audio_file.path, video_file.path, max_duration_diff)
            )
            logging.error(msg)
            raise MediaEditorError(msg)

        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-i", video_file.path,
            "-i", audio_file.path,
            "-c:v", video_codec,
            "-c:a", audio_codec,
            "-shortest", # Ensure output duration is same as shortest input
            merged_video_file_path,
        ]

        success_msg = "Successfully merged video '{}' and audio '{}' to '{}'.".format(
            video_file.path, audio_file.path, merged_video_file_path
        )
        failure_msg = "Failed to merge video '{}' and audio '{}'.".format(
            video_file.path, audio_file.path
        )

        success = self._run_ffmpeg_command(ffmpeg_command, success_msg, failure_msg)

        if success:
            merged_video = VideoFile(merged_video_file_path)
            merged_video.assert_exists()
            return merged_video
        else:
            return None


    def concatenate(
        self,
        media_files: list[TemporalMediaFile],
        concatenated_media_file_path: str,
        overwrite: bool = True,
    ) -> MediaFile or None:
        """
        Concatenate media_files into a single media file.

        Parameters
        ----------
        media_files: list[TemporalMediaFile]
            list of media files to concatenate
        concatenated_media_file_path: str
            absolute path to store the concatenated media file
        overwrite: bool
            Overwrites'concatenated_media_file_path if True; does not overwrite if False

        Returns
        -------
        MediaFile or None
            the concatenated media file if successful; None if unsuccessful
        """
        if not media_files:
            logging.error("No media files provided for concatenation.")
            return None

        if overwrite is True:
            self._file_system_manager.assert_parent_dir_exists(
                TemporalMediaFile(concatenated_media_file_path) # Use specific type for constructor
            )
        else:
            self._file_system_manager.assert_valid_path_for_new_fs_object(
                concatenated_media_file_path
            )

        for i, media in enumerate(media_files):
            self.assert_valid_media_file(media, TemporalMediaFile)
            self._file_system_manager.assert_paths_not_equal(
                media.path,
                concatenated_media_file_path,
                f"temporal_media{i} path",
                "concatenated_media_file_path",
            )

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp_list_file:
            for media_file in media_files:
                # Ensure paths are suitable for ffmpeg (e.g., properly quoted if they contain spaces)
                # For simplicity, assuming paths are clean. Real-world might need more robust handling.
                tmp_list_file.write(f"file '{os.path.abspath(media_file.path)}'\n")
            media_paths_file_name = tmp_list_file.name

        logging.debug("Temporary media paths file: %s", media_paths_file_name)

        ffmpeg_command = [
            "ffmpeg",
            "-y", # Overwrite output files without asking
            "-f", "concat",
            "-safe", "0", # Disable checks for unsafe file paths if paths are complex
            "-i", media_paths_file_name,
            "-c", "copy", # Copy codecs if possible to avoid re-encoding
            # "-vf", "setpts=PTS-STARTPTS", # This might cause issues if streams are not uniform
            concatenated_media_file_path,
        ]

        success_msg = "Successfully concatenated media files to '{}'.".format(concatenated_media_file_path)
        failure_msg = "Failed to concatenate media files."

        success = self._run_ffmpeg_command(ffmpeg_command, success_msg, failure_msg)

        os.remove(media_paths_file_name) # Clean up the temporary list file

        if success:
            # Determine the type of the output file based on the first input file
            # This assumes all input files are of a compatible type for concatenation
            concatenated_media_file = self._create_media_file_of_same_type(
                concatenated_media_file_path, media_files[0]
            )
            concatenated_media_file.assert_exists()
            return concatenated_media_file
        else:
            return None

    def crop_video(
        self,
        original_video_file: VideoFile,
        cropped_video_file_path: str,
        x: int,
        y: int,
        width: int,
        height: int,
        start_time: float = None,
        end_time: float = None,
        audio_codec: str = "aac",
        video_codec: str = "libx264",
        crf: str = "18",
        preset: str = "veryfast",
        num_threads: str = "0",
        overwrite: bool = True,
    ) -> VideoFile or None:
        """
        Crop a video.

        Parameters
        ----------
        original_video_file: VideoFile
            the video file to crop
        cropped_video_file_path: str
            absolute path to store the cropped video file
        x: int
            x-coordinate of the top left corner of the cropped video
        y: int
            y-coordinate of the top left corner of the cropped video
        width: int
            width of the cropped video
        height: int
            height of the cropped video
        start_time: float
            the time in seconds to begin the cropped video
        end_time: float
            the time in seconds to end the cropped video
        audio_codec: str
            compression and decompression sfotware for the audio (aac)
        video_codec: str
            compression and decompression software for the video (libx264)
        crf: str
            constant rate factor - an encoding mode that adjusts the file data rate up
            or down to achieve a selected quality level rather than a specific data
            rate. CRF values range from 0 to 51, with lower numbers delivering higher
            quality scores
        preset: str
            the encoding speed to compression ratio. A slower preset will provide
            better compression (compression is quality per filesize)
        num_threads: str
            the number of threads to use for encoding
        overwrite: bool
            Overwrites 'cropped_video_file_path' if True; does not overwrite if False

        Returns
        -------
        VideoFile or None
            the cropped video if successful; None if unsuccessful
        """
        self.assert_valid_media_file(original_video_file, VideoFile)
        if overwrite is True:
            self._file_system_manager.assert_parent_dir_exists(
                VideoFile(cropped_video_file_path)
            )
        else:
            self._file_system_manager.assert_valid_path_for_new_fs_object(
                cropped_video_file_path
            )
        self._file_system_manager.assert_paths_not_equal(
            original_video_file.path,
            cropped_video_file_path,
            "original_video_file path",
            "cropped_video_file_path",
        )

        _start_time = start_time if start_time is not None else 0.0
        _end_time = end_time if end_time is not None else original_video_file.get_duration()
        self._assert_valid_trim_times(original_video_file, _start_time, _end_time)

        start_time_str = seconds_to_hms_time_format(_start_time)
        # Calculate duration for -to if end_time is used, or let ffmpeg handle till end if not.
        # Ffmpeg's -to is an absolute timestamp, not duration like -t.
        # For simplicity, using -t with calculated duration might be more consistent with other methods.
        duration_str = seconds_to_hms_time_format(_end_time - _start_time)


        ffmpeg_command = [
            "ffmpeg",
            "-y", # Overwrite output
            "-ss", start_time_str,
            "-t", duration_str,
            "-i", original_video_file.path,
            "-vf", f"crop={width}:{height}:{x}:{y}",
            "-c:v", video_codec,
            "-preset", preset,
            "-c:a", audio_codec,
            "-map", "0", # include all streams by default, adjust if specific streams are needed
            "-crf", crf,
            "-threads", num_threads,
            cropped_video_file_path,
        ]

        success_msg = "Successfully cropped video '{}' to '{}'.".format(
            original_video_file.path, cropped_video_file_path
        )
        failure_msg = "Failed to crop video '{}' to '{}'.".format(
            original_video_file.path, cropped_video_file_path
        )

        success = self._run_ffmpeg_command(ffmpeg_command, success_msg, failure_msg)

        if success:
            cropped_file = VideoFile(cropped_video_file_path)
            cropped_file.assert_exists()
            return cropped_file
        else:
            return None

    def resize_video(
        self,
        original_video_file: VideoFile,
        resized_video_file_path: str,
        width: int,
        height: int,
        segments: list[dict],
        audio_codec: str = "aac",
        video_codec: str = "libx264",
        crf: str = "18",
        preset: str = "veryfast",
        num_threads: str = "0",
        overwrite: bool = True,
    ) -> VideoFile or None:
        """
        Crop a series of videos from a video file to resize the video file

        Parameters
        ----------
        original_video_file: VideoFile
            the video file to crop
        resized_video_file_path: str
            absolute path to store the resized video file
        segments: list[dict]
            list of dictionaries where each dictionary is a distinct segment to crop
            the video. Each dictionary has the following keys:
            x: int
                x-coordinate of the top left corner of the cropped video segment
            y: int
                y-coordinate of the top left corner of the cropped video segment
            start_time: float
                the time in seconds to begin the cropped segment
            end_time: float
                the time in seconds to end the cropped segment
        audio_codec: str
            compression and decompression sfotware for the audio (aac)
        video_codec: str
            compression and decompression software for the video (libx264)
        crf: str
            constant rate factor - an encoding mode that adjusts the file data rate up
            or down to achieve a selected quality level rather than a specific data
            rate. CRF values range from 0 to 51, with lower numbers delivering higher
            quality scores
        preset: str
            the encoding speed to compression ratio. A slower preset will provide
            better compression (compression is quality per filesize)
        num_threads: str
            the number of threads to use for encoding
        overwrite: bool
            Overwrites 'resized_video_file_path' if True; does not overwrite if False

        Returns
        -------
        VideoFile or None
            the cropped video if successful; None if unsuccessful
        """
        self.assert_valid_media_file(original_video_file, VideoFile)
        if overwrite:
            self._file_system_manager.assert_parent_dir_exists(
                VideoFile(resized_video_file_path) # Use specific type for constructor
            )
        else:
            self._file_system_manager.assert_valid_path_for_new_fs_object(
                resized_video_file_path
            )
        self._file_system_manager.assert_paths_not_equal(
            original_video_file.path,
            resized_video_file_path,
            "original_video_file path",
            "resized_video_file_path",
        )

        cropped_video_files: list[VideoFile] = []
        # Create a temporary directory to store intermediate segment files
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            logging.debug(f"Created temporary directory: {tmp_dir_name}")
            for i, segment in enumerate(segments):
                # Generate a unique path within the temporary directory for each segment
                cropped_segment_path = os.path.join(
                    tmp_dir_name,
                    f"{uuid.uuid4().hex}_segment_{i}.mp4"
                )

                cropped_video_file = self.crop_video(
                    original_video_file=original_video_file,
                    cropped_video_file_path=cropped_segment_path,
                    x=segment["x"],
                    y=segment["y"],
                    width=width,
                    height=height,
                    start_time=segment["start_time"],
                    end_time=segment["end_time"],
                    audio_codec=audio_codec,
                    video_codec=video_codec,
                    crf=crf,
                    preset=preset,
                    num_threads=num_threads,
                    overwrite=True, # Always overwrite temp files
                )

                if cropped_video_file is None:
                    logging.error(
                        f"Error cropping video segment {i} with info: {segment}"
                    )
                    # Cleanup already created temp files in case of error
                    # for f in cropped_video_files: # Not strictly necessary with TemporaryDirectory
                    #    if os.path.exists(f.path): f.delete()
                    return None
                else:
                    cropped_video_files.append(cropped_video_file)

            if not cropped_video_files:
                logging.error("No video segments were successfully cropped.")
                return None

            # Concatenate cropped segments
            final_resized_video_file = self.concatenate(
                media_files=cropped_video_files,
                concatenated_media_file_path=resized_video_file_path,
                overwrite=overwrite,
            )
            # TemporaryDirectory and its contents are automatically cleaned up here

        if final_resized_video_file:
            final_resized_video_file.assert_exists()
            return final_resized_video_file
        else:
            return None


    def instantiate_as_temporal_media_file(
        self, media_file_path: str
    ) -> TemporalMediaFile:
        """
        Returns the media file as the correct type (e.g. VideoFile, AudioFile, etc.)

        Parameters
        ----------
        media_file_path: str
            Absolute path to the media file to instantiate

        Returns
        -------
        MediaFile
            the media file as the correct type (e.g. VideoFile, AudioFile, etc.)
        """
        media_file = TemporalMediaFile(media_file_path)
        media_file.assert_exists()

        if media_file.has_audio_stream() and media_file.has_video_stream():
            media_file = AudioVideoFile(media_file.path)
        elif media_file.has_audio_stream():
            media_file = AudioFile(media_file.path)
        else:
            msg = "File '{}' must be a AudioFile, or AudioVideoFile not {}." "".format(
                media_file.path, type(media_file)
            )
            logging.error(msg)
            raise MediaEditorError(msg)

        media_file.assert_exists()
        return media_file

    def check_valid_media_file(
        self, media_file: MediaFile, media_file_type
    ) -> str or None:
        """
        Checks if media_file is of the proper type and exists in the file system.
        Returns None if so, a descriptive error message if not.

        Parameters
        ----------
        media_file: MediaFile
            the media file to check
        media_file_type
            the type of media file to check for (e.g. VideoFile, AudioFile, ImageFile)

        Returns
        -------
        str or None
            None if media_file is of the proper type and exists in the file system, a
            descriptive error message if not
        """
        msg = self._type_checker.check_type(media_file, "media_file", media_file_type)
        if msg is not None:
            return msg

        msg = media_file.check_exists()
        if msg is not None:
            return msg

        return None

    def is_valid_media_file(self, media_file: MediaFile, media_file_type) -> bool:
        """
        Returns True if media_file is of the proper type and exists in the file system.
        Returns False if not.

        Parameters
        ----------
        media_file: MediaFile
            the media file to check
        media_file_type
            the type of media file to check for (e.g. VideoFile, AudioFile, ImageFile)

        Returns
        -------
        bool
            True if media_file is of the proper type and exists in the file system,
            False if not
        """
        return self.check_valid_media_file(media_file, media_file_type) is None

    def assert_valid_media_file(self, media_file: MediaFile, media_file_type) -> None:
        """
        Raises an error media_file is of the proper type and exists in the file system.
        Raises an error if not.

        Parameters
        ----------
        media_file: MediaFile
            the media file to check
        media_file_type
            the type of media file to check for (e.g. VideoFile, AudioFile, ImageFile)

        Raises
        ------
        MediaEditorError: media_file is not of the proper type or does not exist in the
            file system
        """
        msg = self.check_valid_media_file(media_file, media_file_type)
        if msg is not None:
            raise MediaEditorError(msg)

    def _check_valid_trim_times(
        self,
        media_file: TemporalMediaFile,
        start_time: float,
        end_time: float,
    ) -> str or None:
        """
        Checks if start_time and end_time are valid times to trim the media file.
        Returns None if so, a descriptive error message if not.

        Parameters
        ----------
        media_file: TemporalMediaFile
            the media file to check
        start_time: float
            the time in seconds the trimmed media file begins
        end_time: float
            the time in seconds the trimmed media file ends

        Returns
        -------
        str or None
            None if start_time and end_time are valid for the media file, a descriptive
            error message if not
        """
        if start_time is None or end_time is None: # Allow None for full duration operations
            return None
        if start_time < 0:
            return "Start second ({} seconds) cannot be negative.".format(start_time)
        if end_time < 0: # Should be end_time, not start_time
            return "End second ({} seconds) cannot be negative.".format(end_time)
        if start_time > end_time:
            return (
                "Start second ({} seconds) cannot exceed end second ({} seconds)."
                "".format(start_time, end_time)
            )

        duration = media_file.get_duration()
        if duration == -1: # Allow processing if duration is unknown, ffmpeg might handle it
            logging.warning(
                "Can't retrieve video duration from media file '{}'. Attempting to "
                "trim with given start_time ({}) and end_time ({}) regardless."
                "".format(media_file.path, start_time, end_time)
            )
            return None
        # Allow start_time == duration for specific edge cases (e.g. empty clip)
        if start_time > duration:
            return (
                "Start second ({} seconds) cannot exceed video duration ({} seconds)."
                "".format(start_time, duration)
            )
        # Allow end_time to slightly exceed duration due to precision, ffmpeg clips at actual end
        if end_time > duration + 1:  # Keep a small buffer for float precision
            logging.warning(
                "End second ({} seconds) slightly exceeds video duration ({} seconds)."
                " FFmpeg will clip at actual end of video.".format(end_time, duration)
            )

        return None

    def _is_valid_trim_times(
        self,
        media_file: TemporalMediaFile,
        start_time: float,
        end_time: float,
    ) -> bool:
        """
        Returns True if start_time and end_time are valid times to trim the media file.
        Returns False if not.

        Parameters
        ----------
        media_file: TemporalMediaFile
            the media file to check
        start_time: float
            the time in seconds the trimmed media file begins
        end_time: float
            the time in seconds the trimmed media file ends

        Returns
        -------
        bool
            True if start_time and end_time are valid for the media file, False if not
        """
        return self._check_valid_trim_times(media_file, start_time, end_time) is None

    def _assert_valid_trim_times(
        self,
        media_file: TemporalMediaFile,
        start_time: float,
        end_time: float,
    ) -> None:
        """
        Raises an error if start_time and end_time are not valid times to trim the media
        file. Raises an error if not.

        Parameters
        ----------
        media_file: TemporalMediaFile
            the media file to check
        start_time: float
            the time in seconds the trimmed media file begins
        end_time: float
            the time in seconds the trimmed media file ends

        Raises
        ------
        MediaEditorError: start_time and end_time are not valid for the media file
        """
        msg = self._check_valid_trim_times(media_file, start_time, end_time)
        if msg is not None:
            raise MediaEditorError(msg)

    def _create_media_file_of_same_type(
        self,
        file_path_to_create_media_file_from: str,
        media_file_to_copy_type_of: MediaFile,
    ) -> MediaFile:
        """
        Creates a MediaFile object with the same type as 'media_file_to_copy_type_of'
        from the file at 'file_path_to_create_media_file_from'

        Parameters
        ----------
        file_path_to_create_media_file_from: str
            absolute path to the file to create a MediaFile object from
        media_file_to_copy_type_of: MediaFile
            the media file to copy the type of

        Returns
        -------
        MediaFile
            the media file at 'file_path_to_create_media_file_from' as an MediaFile
            object of the same type as 'media_file_to_copy_type_of''
        """
        self._type_checker.assert_type(
            media_file_to_copy_type_of, "media_file_to_copy_type_of", MediaFile
        )

        if isinstance(media_file_to_copy_type_of, ImageFile):
            created_file = ImageFile(file_path_to_create_media_file_from)
        elif isinstance(media_file_to_copy_type_of, AudioFile):
            created_file = AudioFile(file_path_to_create_media_file_from)
        elif isinstance(media_file_to_copy_type_of, VideoFile): # Order matters: VideoFile before AudioVideoFile
            created_file = VideoFile(file_path_to_create_media_file_from)
        elif isinstance(media_file_to_copy_type_of, AudioVideoFile):
            created_file = AudioVideoFile(file_path_to_create_media_file_from)
        elif isinstance(media_file_to_copy_type_of, TemporalMediaFile): # Fallback for generic TemporalMediaFile
             created_file = TemporalMediaFile(file_path_to_create_media_file_from)
        else:
            # Fallback to MediaFile if specific type is unknown or not handled above
            logging.warning(
                 "Unknown specific media file type for '{}'. Defaulting to MediaFile. "
                 "Considered type: {}"
                 .format(media_file_to_copy_type_of.path, type(media_file_to_copy_type_of))
            )
            created_file = MediaFile(file_path_to_create_media_file_from)


        return created_file
