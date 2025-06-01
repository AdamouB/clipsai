"""
Diarize an audio file using pyannote/speaker-diarization-3.1

Notes
-----
- Real-time factor is around 2.5% using one Nvidia Tesla V100 SXM2 GPU (for the neural
inference part) and one Intel Cascade Lake 6248 CPU (for the clustering part).
In other words, it takes approximately 1.5 minutes to process a one hour conversation.

- The technical details of the model are described in
 https://huggingface.co/pyannote/speaker-diarization-3.0

- pyannote/speaker-diarization allows setting a number of speakers to detect. Could be
viable to analyze different subsections of the video, detect the number of faces, and
use that as the number of speakers to detect.
"""
# standard library imports
import logging
import os
import uuid

# local package imports
from clipsai.media.audio_file import AudioFile
from clipsai.utils.pytorch import get_compute_device, assert_compute_device_available

# third party imports
# Attempt to import Pipeline and related, but handle if not available due to optional install
try:
    from pyannote.audio import Pipeline
    from pyannote.core.annotation import Annotation
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    # Define dummy classes if pyannote.audio is not installed, so type hints don't break
    # and the rest of the class structure can be maintained.
    class Pipeline: pass
    class Annotation: pass


import torch


class PyannoteDiarizer:
    """
    A class for diarizing audio files using pyannote.
    """

    def __init__(self, auth_token: str = None, device: str = None) -> None:
        """
        Initialize PyannoteDiarizer

        Parameters
        ----------
        auth_token: str, optional
            Authentication token for Pyannote, obtained from HuggingFace.
            If None or empty, diarization will be skipped.
        device: str
            PyTorch device to perform computations on. Ex: 'cpu', 'cuda'. Default is
            None (auto detects the correct device)

        Returns
        -------
        None
        """
        self.pipeline = None
        if not PYANNOTE_AVAILABLE:
            logging.warning(
                "pyannote.audio package not found. Speaker diarization will be skipped. "
                "Please install it if you need this feature: pip install pyannote.audio"
            )
            return

        if not auth_token:
            logging.info(
                "No pyannote auth token provided or token is empty. "
                "Speaker diarization will be skipped."
            )
            return

        if device is None:
            device = get_compute_device()

        try:
            assert_compute_device_available(device)
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=auth_token,
            ).to(torch.device(device))
            logging.debug("Pyannote using device: {}".format(self.pipeline.device))
        except Exception as e:
            logging.warning(
                f"Failed to load pyannote/speaker-diarization-3.1 pipeline "
                f"with the provided token (Hugging Face Hub error: {e}). "
                f"Speaker diarization will be skipped."
            )
            self.pipeline = None


    def diarize(
        self,
        audio_file: AudioFile,
        min_segment_duration: float = 1.5,
        time_precision: int = 6,
    ) -> list[dict]:
        """
        Diarizes the audio file.

        Parameters
        ----------
        audio_file: AudioFile
            the audio file to diarize
        time_precision: int
            The number of decimal places for rounding the start and end times of
            segments.
        min_segment_duration: float
            The minimum duration (in seconds) for a segment to be considered valid.

        Returns
        -------
        speaker_segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
        """
        if self.pipeline is None:
            logging.info(
                "Pyannote pipeline not available. Skipping speaker diarization."
            )
            return []

        if audio_file.has_file_extension("wav"):
            wav_file = audio_file
        else:
            # Generate a unique temporary path for the WAV file
            temp_dir = tempfile.gettempdir()
            temp_wav_filename = "{}{}.wav".format(
                audio_file.get_filename_without_extension(),
                str(uuid.uuid4().hex)
            )
            wav_file_path = os.path.join(temp_dir, temp_wav_filename)

            logging.debug(f"Extracting audio to temporary WAV file: {wav_file_path}")
            wav_file = audio_file.extract_audio(
                extracted_audio_file_path=wav_file_path,
                audio_codec="pcm_s16le",
                overwrite=True, # Allow overwrite for temp files
            )
            if wav_file is None:
                logging.error("Failed to extract audio to WAV for diarization.")
                return []

        try:
            pyannote_segments: Annotation = self.pipeline({"audio": wav_file.path})

            adjusted_speaker_segments = self._adjust_segments(
                pyannote_segments=pyannote_segments,
                min_segment_duration=min_segment_duration,
                duration=audio_file.get_duration(),
                time_precision=time_precision,
            )
        finally:
            # Clean up temporary WAV file if it was created
            if not audio_file.has_file_extension("wav") and wav_file is not None:
                logging.debug(f"Deleting temporary WAV file: {wav_file.path}")
                wav_file.delete()


        return adjusted_speaker_segments

    def _adjust_segments(
        self,
        pyannote_segments: Annotation,
        min_segment_duration: float,
        duration: float,
        time_precision: int,
    ) -> list[dict]:
        """
        Adjusts and merges speaker segments to achieve an unbroken, non-overlapping
        sequence of speaker segments with at least one person speaking in each segment.

        Parameters
        ----------
        pyannote_segments: Annotation
            the pyannote speaker segments
        duration: float
            duration of the audio being diarized.
        time_precision: int
            The number of decimal places for rounding the start and end times of
            segments.
        min_segment_duration: float
            The minimum duration (in seconds) for a segment to be considered valid.

        Returns
        -------
        speaker_segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
        """
        cur_end_time = None
        cur_speaker = None
        cur_start_time = 0.000
        adjusted_speaker_segments = []
        unique_speakers: set[int] = set()

        if pyannote_segments is None: # Should not happen if pipeline worked
             return []

        for segment, _, speaker_label in pyannote_segments.itertracks(True):
            next_start_time = segment.start
            next_end_time = segment.end
            if speaker_label.split("_")[1] == "":
                next_speaker = None
            else:
                next_speaker = int(speaker_label.split("_")[1])

            if next_end_time - next_start_time < min_segment_duration:
                continue

            if cur_speaker is None: # First speaker segment
                # If the audio starts with silence before the first speaker
                if next_start_time > cur_start_time and cur_start_time == 0.0:
                     adjusted_speaker_segments.append({
                         "speakers": [], # No one speaking
                         "start_time": round(cur_start_time, time_precision),
                         "end_time": round(next_start_time, time_precision),
                     })
                cur_speaker = next_speaker
                cur_start_time = next_start_time # Actual start of this speaker
                cur_end_time = next_end_time
                continue

            if cur_speaker == next_speaker:
                cur_end_time = max(cur_end_time, next_end_time)
                continue

            # Gap between current segment and next different speaker segment
            if next_start_time > cur_end_time:
                # Add current speaker segment
                if cur_speaker is not None:
                    speakers = [cur_speaker]; unique_speakers.add(cur_speaker)
                else: speakers = []
                adjusted_speaker_segments.append({
                    "speakers": speakers,
                    "start_time": round(cur_start_time, time_precision),
                    "end_time": round(cur_end_time, time_precision),
                })
                # Add silence segment for the gap
                adjusted_speaker_segments.append({
                    "speakers": [],
                    "start_time": round(cur_end_time, time_precision),
                    "end_time": round(next_start_time, time_precision),
                })
            # Overlap or direct continuation by different speaker
            else: # next_start_time <= cur_end_time
                adjusted_end_time = min(cur_end_time, next_start_time) # End current segment where new one starts
                if cur_speaker is not None:
                    speakers = [cur_speaker]; unique_speakers.add(cur_speaker)
                else: speakers = []
                if adjusted_end_time > cur_start_time: # Ensure segment has duration
                    adjusted_speaker_segments.append({
                        "speakers": speakers,
                        "start_time": round(cur_start_time, time_precision),
                        "end_time": round(adjusted_end_time, time_precision),
                    })

            cur_speaker = next_speaker
            cur_start_time = next_start_time # Start of the new speaker's segment
            cur_end_time = next_end_time


        # Add the very last segment
        if cur_start_time < duration : # Ensure there's a segment to add
            if cur_speaker is not None:
                speakers = [cur_speaker]; unique_speakers.add(cur_speaker)
            else: speakers = []
            adjusted_speaker_segments.append({
                "speakers": speakers,
                "start_time": round(cur_start_time, time_precision),
                "end_time": round(duration, time_precision), # Segment goes to the end of audio
            })

        # Filter out zero-duration segments that might have been created by adjustments
        adjusted_speaker_segments = [
            s for s in adjusted_speaker_segments if s["end_time"] > s["start_time"]
        ]

        adjusted_speaker_segments = self._relabel_speakers(
            adjusted_speaker_segments, unique_speakers
        )
        return adjusted_speaker_segments

    def _relabel_speakers(
        self, speaker_segments: list[dict], unique_speakers: set[int]
    ) -> dict[int, int]:
        """
        Relabels speaker segments so that the speaker labels are contiguous.

        Some speakers may have been skipped if their segments were too short. Thus,
        we could end up with a set of speaker labels like {0, 1, 3}. This function
        relabels the speakers to remove gaps so that our set of speaker labels would
        be contiguous, e.g. {0, 1, 2}.

        Parameters
        ----------
        speaker_segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
        unique_speakers: set[int]
            set of unique speaker labels in the speaker segments

        Returns
        -------
        updated_speaker_segments: list[dict]
            list of speaker segments where the speakers are relabeled so that the
            speaker labels are contiguous. Each dictionary contains the following keys:
                speakers: list[int]
                    list of speaker numbers for the speakers talking in the segment
                start_time: float
                    start time of the segment in seconds
                end_time: float
                    end time of the segment in seconds
        """
        if not unique_speakers: # Handles empty set or if all segments had no speaker
            return speaker_segments

        sorted_unique_speakers = sorted(list(unique_speakers))

        # Check if relabeling is needed (i.e. if labels are already 0, 1, 2, ...)
        if all(sorted_unique_speakers[i] == i for i in range(len(sorted_unique_speakers))):
            return speaker_segments

        relabel_speaker_map = {old_label: new_label for new_label, old_label in enumerate(sorted_unique_speakers)}

        for segment in speaker_segments:
            relabeled_speakers = []
            for speaker in segment["speakers"]:
                if speaker in relabel_speaker_map: # Should always be true if speaker was in unique_speakers
                    relabeled_speakers.append(relabel_speaker_map[speaker])
                # else: # Should not happen if unique_speakers was built correctly
                #    relabeled_speakers.append(speaker) # Keep original if somehow not in map
            segment["speakers"] = relabeled_speakers

        return speaker_segments

    def cleanup(self) -> None:
        """
        Remove the diarization pipeline from memory and explicity free up GPU memory.
        """
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.debug("Pyannote pipeline cleaned up.")
        else:
            logging.debug("No Pyannote pipeline to clean up.")

# Need to import tempfile for the diarize method's temporary WAV file handling
import tempfile
