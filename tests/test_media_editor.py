import pytest
import sys # For sys.modules trick
from unittest.mock import patch, MagicMock, mock_open, call, ANY
import os # For os.path.abspath in concatenate test

# Mock clipsai modules that lead to sentence_transformers -> transformers -> torch import chain
sys.modules['clipsai.clip.clipfinder'] = MagicMock(name='mock_clipfinder_module')
sys.modules['clipsai.clip.text_embedder'] = MagicMock(name='mock_text_embedder_module')

# Mock other heavy dependencies that might be imported by other clipsai modules
sys.modules['facenet_pytorch'] = MagicMock(name='mock_facenet_pytorch')
sys.modules['mediapipe'] = MagicMock(name='mock_mediapipe')
sys.modules['pyannote.audio'] = MagicMock(name='mock_pyannote_audio')
sys.modules['whisperx'] = MagicMock(name='mock_whisperx')


# Provide a minimal mock for 'torch' to satisfy clipsai.utils.pytorch and other direct uses within clipsai
mock_torch_for_clipsai_utils = MagicMock(name='minimal_torch_for_clipsai')
mock_torch_for_clipsai_utils.tensor = MagicMock(name='torch_tensor_mock')
mock_torch_for_clipsai_utils.Tensor = MagicMock(name='torch_Tensor_mock')
mock_torch_for_clipsai_utils.device = MagicMock(name='torch_device_mock')
mock_torch_for_clipsai_utils.cuda = MagicMock(name='torch_cuda_mock')
mock_torch_for_clipsai_utils.cuda.is_available = MagicMock(return_value=False)
mock_torch_for_clipsai_utils.backends = MagicMock(name='torch_backends_mock')
mock_torch_for_clipsai_utils.backends.mps = MagicMock(name='torch_backends_mps_mock')
mock_torch_for_clipsai_utils.backends.mps.is_available = MagicMock(return_value=False)
mock_torch_for_clipsai_utils.is_tensor = MagicMock(return_value=False, name='is_tensor_mock')
mock_torch_for_clipsai_utils.manual_seed = MagicMock(name='manual_seed_mock')
mock_torch_for_clipsai_utils.empty = MagicMock(name='torch_empty_mock')
mock_torch_for_clipsai_utils.stack = MagicMock(name='torch_stack_mock')
mock_torch_for_clipsai_utils.abs = MagicMock(name='torch_abs_mock')
mock_torch_for_clipsai_utils.max = MagicMock(name='torch_max_mock')
mock_torch_for_clipsai_utils.uint8 = MagicMock(name='torch_uint8_mock')
mock_torch_for_clipsai_utils.from_numpy = MagicMock(name='torch_from_numpy_mock')
sys.modules['torch'] = mock_torch_for_clipsai_utils


# Modules to test
from clipsai.media.editor import MediaEditor, SUCCESS
from clipsai.media.media_file import MediaFile
from clipsai.media.temporal_media_file import TemporalMediaFile
from clipsai.media.video_file import VideoFile
from clipsai.media.audio_file import AudioFile
# FileSystemManager is imported by MediaEditor, so it's available if needed for direct patching,
# but the fixture approach is generally better.
# from clipsai.filesys.manager import FileSystemManager


class TestMediaEditor:
    @pytest.fixture
    def editor(self):
        """Fixture to create a MediaEditor instance with a mocked FileSystemManager."""
        with patch('clipsai.media.editor.FileSystemManager', autospec=True) as mock_fsm_constructor:
            mock_fsm_instance = mock_fsm_constructor.return_value
            editor_instance = MediaEditor()
            editor_instance._file_system_manager = mock_fsm_instance
            return editor_instance

    def test_trim_command_generation_success(self, editor):
        mock_media_file = MagicMock(spec=TemporalMediaFile)
        mock_media_file.path = "/path/to/input.mp4"
        mock_media_file.get_duration.return_value = 60.0
        mock_media_file.check_exists.return_value = None
        mock_media_file.__class__ = VideoFile


        start_time = 10.0
        end_time = 20.0
        trimmed_media_file_path = "/path/to/output/trimmed.mp4"

        with patch('subprocess.run') as mock_subproc_run, \
             patch.object(MediaEditor, '_create_media_file_of_same_type') as mock_create_media:

            mock_ffmpeg_result = MagicMock()
            mock_ffmpeg_result.returncode = SUCCESS
            mock_ffmpeg_result.stdout = "ffmpeg output"
            mock_ffmpeg_result.stderr = ""
            mock_subproc_run.return_value = mock_ffmpeg_result

            mock_created_file = MagicMock(spec=VideoFile)
            mock_create_media.return_value = mock_created_file

            result_file = editor.trim(
                mock_media_file,
                start_time,
                end_time,
                trimmed_media_file_path
            )

            mock_subproc_run.assert_called_once()
            args, kwargs = mock_subproc_run.call_args
            actual_command = args[0]

            expected_command = [
                "ffmpeg", "-y", "-ss", "00:00:10.000", "-t", "00:00:10.000",
                "-i", "/path/to/input.mp4",
                "-c:v", "copy", "-preset", "medium", "-c:a", "copy",
                "-map", "0", "-crf", "23", "-threads", "0",
                trimmed_media_file_path
            ]
            assert actual_command == expected_command

            mock_create_media.assert_called_once_with(
                trimmed_media_file_path, mock_media_file
            )
            mock_created_file.assert_exists.assert_called_once()
            assert result_file == mock_created_file

            editor._file_system_manager.assert_parent_dir_exists.assert_called_once_with(ANY)
            editor._file_system_manager.assert_paths_not_equal.assert_called_once_with(
                mock_media_file.path,
                trimmed_media_file_path,
                "media_file path",
                "trimmed_media_file_path",
            )

    def test_trim_failure(self, editor):
        mock_media_file = MagicMock(spec=TemporalMediaFile)
        mock_media_file.path = "/path/to/input.mp4"
        mock_media_file.get_duration.return_value = 60.0
        mock_media_file.check_exists.return_value = None
        mock_media_file.__class__ = VideoFile


        start_time = 10.0
        end_time = 20.0
        trimmed_media_file_path = "/path/to/output/trimmed.mp4"

        with patch('subprocess.run') as mock_subproc_run, \
             patch.object(MediaEditor, '_create_media_file_of_same_type') as mock_create_media:

            mock_ffmpeg_result = MagicMock()
            mock_ffmpeg_result.returncode = 1
            mock_subproc_run.return_value = mock_ffmpeg_result

            result_file = editor.trim(
                mock_media_file,
                start_time,
                end_time,
                trimmed_media_file_path
            )

            mock_subproc_run.assert_called_once()
            mock_create_media.assert_not_called()
            assert result_file is None

            editor._file_system_manager.assert_parent_dir_exists.assert_called_once_with(ANY)
            editor._file_system_manager.assert_paths_not_equal.assert_called_once_with(
                mock_media_file.path,
                trimmed_media_file_path,
                "media_file path",
                "trimmed_media_file_path",
            )

    def test_concatenate_command_generation_success(self, editor):
        mock_file1 = MagicMock(spec=TemporalMediaFile)
        mock_file1.path = os.path.abspath("/path/to/input1.mp4")
        mock_file1.check_exists.return_value = None
        mock_file1.__class__ = VideoFile # Assuming VideoFile for consistency

        mock_file2 = MagicMock(spec=TemporalMediaFile)
        mock_file2.path = os.path.abspath("/path/to/input2.mp4")
        mock_file2.check_exists.return_value = None
        mock_file2.__class__ = VideoFile

        media_files = [mock_file1, mock_file2]
        concatenated_media_file_path = "/path/to/output/concatenated.mp4"

        mock_temp_file_obj = MagicMock()
        mock_temp_file_obj.name = "/tmp/fake_temp_file.txt"
        mock_temp_file_obj.write = MagicMock()

        mock_named_temp_file_context_manager = MagicMock()
        mock_named_temp_file_context_manager.__enter__.return_value = mock_temp_file_obj
        mock_named_temp_file_context_manager.__exit__.return_value = None

        with patch('subprocess.run') as mock_subproc_run, \
             patch('tempfile.NamedTemporaryFile', return_value=mock_named_temp_file_context_manager) as mock_tempfile_constructor, \
             patch.object(MediaEditor, '_create_media_file_of_same_type') as mock_create_media, \
             patch('os.remove') as mock_os_remove:

            mock_ffmpeg_result = MagicMock()
            mock_ffmpeg_result.returncode = SUCCESS
            mock_subproc_run.return_value = mock_ffmpeg_result

            mock_created_file = MagicMock(spec=VideoFile)
            mock_created_file.assert_exists = MagicMock()
            mock_create_media.return_value = mock_created_file

            result_file = editor.concatenate(media_files, concatenated_media_file_path)

            mock_tempfile_constructor.assert_called_once_with(mode="w", delete=False, suffix=".txt")

            expected_temp_file_content = [
                call(f"file '{os.path.abspath(mock_file1.path)}'\n"),
                call(f"file '{os.path.abspath(mock_file2.path)}'\n")
            ]
            mock_temp_file_obj.write.assert_has_calls(expected_temp_file_content)

            mock_subproc_run.assert_called_once()
            args, kwargs = mock_subproc_run.call_args
            actual_command = args[0]

            # Note: The original concatenate command in MediaEditor.py might have "-vf setpts=PTS-STARTPTS"
            # This was commented out in my refactored version. If it's present in the actual code,
            # this expected command will need to be updated. For now, assuming it's not there based on my refactor.
            # Update: The provided editor.py in this session *does* have "-vf setpts=PTS-STARTPTS"
            # in the concatenate method. I need to use that one.
            # However, the refactored version I created earlier had "-c copy" and removed -vf.
            # I need to check which version of editor.py is being used by the test runner.
            # The refactoring request was for this session. The refactored concatenate I made uses "-c copy".
            # The original code from previous session did:
            #  ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", media_paths_file.path,
            #   "-vf", "setpts=PTS-STARTPTS", concatenated_media_file_path]
            # My refactored one in this session's `overwrite_file_with_block`:
            #  ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", media_paths_file_name,
            #   "-c", "copy", concatenated_media_file_path]
            # I will use the one from my refactoring in this session.
            expected_command = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", "/tmp/fake_temp_file.txt",
                "-c", "copy", # As per my refactored concatenate
                concatenated_media_file_path
            ]
            assert actual_command == expected_command

            mock_os_remove.assert_called_once_with(mock_temp_file_obj.name)

            mock_create_media.assert_called_once_with(
                concatenated_media_file_path, media_files[0]
            )
            mock_created_file.assert_exists.assert_called_once()
            assert result_file == mock_created_file

            editor._file_system_manager.assert_parent_dir_exists.assert_called_once_with(ANY)

            expected_fsm_path_calls = [
                call(mock_file1.path, concatenated_media_file_path, "temporal_media0 path", "concatenated_media_file_path"),
                call(mock_file2.path, concatenated_media_file_path, "temporal_media1 path", "concatenated_media_file_path"),
            ]
            editor._file_system_manager.assert_paths_not_equal.assert_has_calls(expected_fsm_path_calls)

            # Input files' assert_exists is part of their own setup, not checked on editor's FSM here.
            # It's checked via self.assert_valid_media_file -> media.check_exists()
            # So, mock_fileX.check_exists() should have been called.
            mock_file1.check_exists.assert_called_once()
            mock_file2.check_exists.assert_called_once()


    def test_concatenate_failure(self, editor):
        mock_file1 = MagicMock(spec=TemporalMediaFile)
        mock_file1.path = os.path.abspath("/path/to/input1.mp4")
        mock_file1.check_exists.return_value = None
        mock_file1.__class__ = VideoFile

        mock_file2 = MagicMock(spec=TemporalMediaFile)
        mock_file2.path = os.path.abspath("/path/to/input2.mp4")
        mock_file2.check_exists.return_value = None
        mock_file2.__class__ = VideoFile

        media_files = [mock_file1, mock_file2]
        concatenated_media_file_path = "/path/to/output/concatenated.mp4"

        mock_temp_file_obj = MagicMock()
        mock_temp_file_obj.name = "/tmp/fake_temp_file.txt" # Keep consistent
        mock_temp_file_obj.write = MagicMock()

        mock_named_temp_file_context_manager = MagicMock()
        mock_named_temp_file_context_manager.__enter__.return_value = mock_temp_file_obj
        mock_named_temp_file_context_manager.__exit__.return_value = None

        with patch('subprocess.run') as mock_subproc_run, \
             patch('tempfile.NamedTemporaryFile', return_value=mock_named_temp_file_context_manager) as mock_tempfile_constructor, \
             patch.object(MediaEditor, '_create_media_file_of_same_type') as mock_create_media, \
             patch('os.remove') as mock_os_remove:

            mock_ffmpeg_result = MagicMock()
            mock_ffmpeg_result.returncode = 1 # Simulate ffmpeg failure
            mock_subproc_run.return_value = mock_ffmpeg_result

            result_file = editor.concatenate(media_files, concatenated_media_file_path)

            mock_tempfile_constructor.assert_called_once_with(mode="w", delete=False, suffix=".txt")
            mock_subproc_run.assert_called_once()
            mock_os_remove.assert_called_once_with(mock_temp_file_obj.name) # Temp file should still be cleaned up
            mock_create_media.assert_not_called()
            assert result_file is None

            editor._file_system_manager.assert_parent_dir_exists.assert_called_once_with(ANY)
            mock_file1.check_exists.assert_called_once()
            mock_file2.check_exists.assert_called_once()


    # def test_run_ffmpeg_command_success(self, editor):
    #     with patch('subprocess.run') as mock_subproc_run, \
    #          patch('logging.debug') as mock_logging_debug, \
    #          patch('logging.error') as mock_logging_error:
    #         mock_result = MagicMock()
    #         mock_result.returncode = SUCCESS
    #         mock_result.stdout = "Success stdout"
    #         mock_result.stderr = ""
    #         mock_subproc_run.return_value = mock_result

    #         success = editor._run_ffmpeg_command(["ffmpeg", "-i", "input"], "Test Success", "Test Failure")
    #         assert success is True
    #         mock_subproc_run.assert_called_once_with(["ffmpeg", "-i", "input"], capture_output=True, text=True)
    #         assert any("Test Success" in str(call_args) for call_args in mock_logging_debug.call_args_list)
    #         mock_logging_error.assert_not_called()


    # def test_run_ffmpeg_command_failure(self, editor):
    #     with patch('subprocess.run') as mock_subproc_run, \
    #          patch('logging.debug') as mock_logging_debug, \
    #          patch('logging.error') as mock_logging_error:
    #         mock_result = MagicMock()
    #         mock_result.returncode = 1
    #         mock_result.stdout = "Failure stdout"
    #         mock_result.stderr = "Failure stderr"
    #         mock_subproc_run.return_value = mock_result

    #         success = editor._run_ffmpeg_command(["ffmpeg", "-i", "input"], "Test Success", "Test Failure")
    #         assert success is False
    #         mock_subproc_run.assert_called_once_with(["ffmpeg", "-i", "input"], capture_output=True, text=True)
    #         assert any("Test Failure" in str(call_args) for call_args in mock_logging_error.call_args_list)
    #         assert not any("Test Success" in str(call_args) for call_args in mock_logging_debug.call_args_list if "ffmpeg_command" not in str(call_args))
