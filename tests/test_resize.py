import pytest
from unittest.mock import patch, MagicMock, call
import logging # For asserting log messages
import sys
import numpy # For the torch mock if it uses numpy

# --- Start of Mocks for External Heavy Dependencies ---
# These must be defined BEFORE any clipsai imports to prevent real import attempts

# Mock for facenet_pytorch.MTCNN
mock_facenet_pytorch_pkg = MagicMock(name='mock_facenet_pytorch_package')
mock_mtcnn_instance = MagicMock(name='mock_mtcnn_instance')
mock_mtcnn_instance.detect.return_value = (None, None)
mock_facenet_pytorch_pkg.MTCNN = MagicMock(return_value=mock_mtcnn_instance, name='MockMTCNN_class')
sys.modules['facenet_pytorch'] = mock_facenet_pytorch_pkg

# Mock for mediapipe.solutions.face_mesh.FaceMesh
mock_mp_pkg = MagicMock(name='mock_mediapipe_package')
mock_mp_solutions_module = MagicMock(name='mock_mp_solutions_module')
mock_mp_face_mesh_module = MagicMock(name='mock_mp_face_mesh_module')
mock_face_mesh_instance = MagicMock(name='mock_face_mesh_instance')
mock_face_mesh_results = MagicMock()
mock_face_mesh_results.multi_face_landmarks = None
mock_face_mesh_instance.process.return_value = mock_face_mesh_results
mock_mp_face_mesh_module.FaceMesh = MagicMock(return_value=mock_face_mesh_instance, name='MockFaceMesh_class')
mock_mp_solutions_module.face_mesh = mock_mp_face_mesh_module
mock_mp_pkg.solutions = mock_mp_solutions_module
sys.modules['mediapipe'] = mock_mp_pkg
sys.modules['mediapipe.solutions'] = mock_mp_solutions_module
sys.modules['mediapipe.solutions.face_mesh'] = mock_mp_face_mesh_module

# Mock other problematic direct imports from clipsai submodules
sys.modules['whisperx'] = MagicMock(name='mock_whisperx')
sys.modules['pyannote.audio'] = MagicMock(name='mock_pyannote_audio')
sys.modules['pyannote.core'] = MagicMock(name='mock_pyannote_core') # If pyannote.audio imports it
sys.modules['sentence_transformers'] = MagicMock(name='mock_sentence_transformers_package')
sys.modules['transformers.activations'] = MagicMock(name='mock_transformers_activations_for_resizer')
sys.modules['clipsai.clip.text_embedder'] = MagicMock(name='mock_text_embedder_module_for_sentence_transformers_import')


# Comprehensive Torch Mock
class MockTorchTensor:
    def __init__(self, data, dtype=None, device=None):
        if isinstance(data, MockTorchTensor): self.data = data.data
        elif not isinstance(data, (list, numpy.ndarray)): self.data = [data] if not hasattr(data, '__iter__') else list(data)
        else: self.data = data
        self.dtype = dtype; self.device = device
        self.shape = numpy.array(self.data).shape if self.data is not None else (0,)
    def cpu(self): self.device = 'cpu'; return self
    def numpy(self): return numpy.array(self.data)
    def item(self):
        if hasattr(self.data, 'item'): return self.data.item()
        if isinstance(self.data, list) and len(self.data) == 1 and isinstance(self.data[0], (int, float)): return self.data[0]
        if isinstance(self.data, list) and len(self.data) == 1 and hasattr(self.data[0], '__len__') and len(self.data[0]) == 1: return self.data[0][0]
        return self.data
    def size(self, dim=None): s = self.shape; return s if dim is None else s[dim]
    def float(self): return self
    def to(self, dev_or_dtype):
        if isinstance(dev_or_dtype, str): self.device = dev_or_dtype
        else: self.dtype = dev_or_dtype
        return self
    def __getitem__(self, item):
        if isinstance(self.data, numpy.ndarray): return MockTorchTensor(self.data[item])
        if isinstance(item, slice) or isinstance(item, int): return MockTorchTensor(self.data[item])
        raise TypeError(f"MockTorchTensor basic __getitem__ doesn't support item type: {type(item)}")
    def __len__(self): return len(self.data) if self.data is not None else 0
    @property
    def T(self): return MockTorchTensor(numpy.array(self.data).T.tolist())

mock_torch_module = MagicMock(name='mock_torch_module_for_resize_test')
mock_spec = MagicMock(name='torch_spec_mock'); mock_spec.loader = MagicMock(name='torch_loader_mock'); mock_spec.origin = "mocked_torch"; mock_spec.name = "torch"
mock_torch_module.__spec__ = mock_spec
mock_torch_module.__version__ = "1.13.1"
mock_torch_module.tensor = lambda data, dtype=None, device=None: MockTorchTensor(data, dtype, device)
mock_torch_module.Tensor = MockTorchTensor
mock_torch_module.is_tensor = lambda obj: isinstance(obj, MockTorchTensor)
mock_torch_module.randn = lambda *size, **kwargs: MockTorchTensor(numpy.random.randn(*size))
mock_torch_module.from_numpy = lambda x: MockTorchTensor(x.tolist() if isinstance(x, numpy.ndarray) else x)
mock_linalg_module = MagicMock(name="torch_linalg_mock"); mock_linalg_module.norm = MagicMock(return_value=mock_torch_module.tensor([1.0])); mock_torch_module.linalg = mock_linalg_module; sys.modules['torch.linalg'] = mock_linalg_module
class MinimalMockTorchNNModule(object):
    def __init__(self, *args, **kwargs): self._init_mock_call = MagicMock(name="MinimalMockTorchNNModule__init__called"); self._init_mock_call(*args, **kwargs)
    parameters = MagicMock(return_value=iter([])); to = MagicMock(); register_buffer = MagicMock(); register_parameter = MagicMock(); add_module = MagicMock(); apply = MagicMock(); cuda = MagicMock(); cpu = MagicMock(); type = MagicMock(); float = MagicMock(); double = MagicMock(); half = MagicMock(); bfloat16 = MagicMock(); state_dict = MagicMock(return_value={}); load_state_dict = MagicMock(); forward = MagicMock(return_value=MagicMock(name="MinimalForwardOutput"))
class MockNNSequential(object):
    def __init__(self, *args, **kwargs):
        self._modules_dict = {}
        self._ordered_modules = []
        if args and isinstance(args[0], dict):
            self._modules_dict = args[0]
            self._ordered_modules = list(args[0].values())
        elif args:
            self._ordered_modules = list(args)
            for i, module in enumerate(args):
                self._modules_dict[str(i)] = module
        self.init_mock_call = MagicMock(name="MockNNSequential__init__called")
        self.init_mock_call(*args, **kwargs)
    def __getitem__(self, idx):
        if isinstance(idx, int): return self._ordered_modules[idx]
        return self._modules_dict.get(str(idx))
    def __setitem__(self, idx, module):
        self._modules_dict[str(idx)] = module
        if not isinstance(idx, int): self._ordered_modules = list(self._modules_dict.values())
    def __len__(self): return len(self._ordered_modules) if self._ordered_modules else len(self._modules_dict)
    to = MagicMock(); children = MagicMock(return_value=iter([])); named_children = MagicMock(return_value=iter([])); parameters = MagicMock(return_value=iter([]))

mock_nn_pkg = MagicMock(name="torch_nn_package_mock")
mock_nn_pkg.Module = MinimalMockTorchNNModule
mock_nn_pkg.Sequential = MockNNSequential
mock_functional_module = MagicMock(name="torch_nn_functional_mock")
mock_functional_module.normalize = MagicMock(side_effect=lambda x, p=2, dim=1, eps=1e-12: x)
mock_functional_module.gelu = MagicMock(return_value=MockTorchTensor([]))
mock_functional_module.relu = MagicMock(return_value=MockTorchTensor([]))
mock_nn_pkg.functional = mock_functional_module
mock_torch_module.nn = mock_nn_pkg
sys.modules['torch.nn'] = mock_nn_pkg
sys.modules['torch.nn.functional'] = mock_functional_module

mock_torch_module.device = MagicMock(side_effect=lambda x: x)
mock_torch_module.cuda = MagicMock(name="torch_cuda_mock"); mock_torch_module.cuda.is_available = MagicMock(return_value=False)
mock_torch_module.backends = MagicMock(name="torch_backends_mock"); mock_torch_module.backends.mps = MagicMock(name="torch_backends_mps_mock"); mock_torch_module.backends.mps.is_available = MagicMock(return_value=False)
class MockBatchSampler(object):
    def __init__(self, *args, **kwargs):
        pass
class MockDataset(object):
    def __init__(self, *args, **kwargs):
        pass
    def __getitem__(self, idx):
        raise IndexError
    def __len__(self):
        return 0

mock_utils_data_module = MagicMock(name='torch_utils_data_mock');
mock_utils_data_distributed_module = MagicMock(name='torch_utils_data_distributed_mock');
mock_utils_data_distributed_module.DistributedSampler = MagicMock(name='DistributedSampler_mock');
mock_utils_data_module.distributed = mock_utils_data_distributed_module;
mock_utils_data_module.BatchSampler = MockBatchSampler;
mock_utils_data_module.Dataset = MockDataset;
sys.modules['torch.utils.data.distributed'] = mock_utils_data_distributed_module

mock_utils_module = MagicMock(name='torch_utils_mock');
mock_utils_module.data = mock_utils_data_module;
mock_utils_module._pytree = MagicMock(name='torch_utils_pytree_mock');
mock_utils_module.checkpoint = MagicMock(name='torch_utils_checkpoint_mock');
mock_torch_module.utils = mock_utils_module
sys.modules['torch.utils'] = mock_utils_module;
sys.modules['torch.utils.data'] = mock_utils_data_module;
sys.modules['torch.utils._pytree'] = mock_utils_module._pytree;
sys.modules['torch.utils.checkpoint'] = mock_utils_module.checkpoint

mock_distributed_module = MagicMock(name='torch_distributed_mock');
mock_distributed_module.is_available = MagicMock(return_value=False);
mock_distributed_module.is_initialized = MagicMock(return_value=False);
mock_distributed_module.tensor = MagicMock(name='torch_distributed_tensor_mock') ;
mock_torch_module.distributed = mock_distributed_module
sys.modules['torch.distributed'] = mock_distributed_module;
sys.modules['torch.distributed.tensor'] = mock_distributed_module.tensor

mock_distributions_module = MagicMock(name='torch_distributions_mock');
mock_distributions_module.constraints = MagicMock(name='torch_distributions_constraints_mock');
mock_torch_module.distributions = mock_distributions_module
sys.modules['torch.distributions'] = mock_distributions_module;
sys.modules['torch.distributions.constraints'] = mock_distributions_module.constraints

mock_optim_module = MagicMock(name='torch_optim_mock');
mock_optim_module.Optimizer = MagicMock(name='Optimizer_class_mock');
mock_lr_scheduler_module = MagicMock(name='torch_optim_lr_scheduler_mock');
mock_lr_scheduler_module.LambdaLR = MagicMock(name='LambdaLR_mock');
mock_optim_module.lr_scheduler = mock_lr_scheduler_module;
mock_torch_module.optim = mock_optim_module
sys.modules['torch.optim'] = mock_optim_module;
sys.modules['torch.optim.lr_scheduler'] = mock_lr_scheduler_module

mock_multiprocessing_module = MagicMock(name='torch_multiprocessing_mock');
mock_multiprocessing_module.get_context = MagicMock();
mock_torch_module.multiprocessing = mock_multiprocessing_module;
sys.modules['torch.multiprocessing'] = mock_multiprocessing_module

mock_activations_module = MagicMock(name='mock_transformers_activations');
ACT2FN_dict_mock = {};
mock_gelu_activation = MagicMock(name="gelu_activation_mock_in_ACT2FN_dict", return_value=MockTorchTensor([]));
ACT2FN_dict_mock["gelu"] = mock_gelu_activation;
mock_relu_activation = MagicMock(name="relu_activation_mock_in_ACT2FN_dict", return_value=MockTorchTensor([]));
ACT2FN_dict_mock["relu"] = mock_relu_activation;
mock_activations_module.ACT2FN = ACT2FN_dict_mock;
mock_activations_module.get_activation = MagicMock(name="get_activation_mock", side_effect=lambda activation_string: ACT2FN_dict_mock.get(activation_string));
sys.modules['transformers.activations'] = mock_activations_module

sys.modules['torch'] = mock_torch_module
# --- End of Enhanced Torch Mock ---

# Mock clipsai.utils.pytorch as Resizer and other modules use it
mock_clipsai_pytorch_utils = MagicMock(name='mock_clipsai_pytorch_utils')
mock_clipsai_pytorch_utils.get_compute_device.return_value = 'cpu'
mock_clipsai_pytorch_utils.assert_compute_device_available = MagicMock()
mock_clipsai_pytorch_utils.assert_valid_torch_device = MagicMock() # Needed by transcriber
sys.modules['clipsai.utils.pytorch'] = mock_clipsai_pytorch_utils


from clipsai.resize.resizer import Resizer
from clipsai.media.video_file import VideoFile
from clipsai.resize.crops import Crops
from clipsai.resize.segment import Segment


def test_resizer_resize_no_speaker_segments_fallback():
    """
    Tests that Resizer.resize returns a default center crop when speaker_segments is empty.
    """
    resizer = Resizer(device='cpu')

    mock_video = MagicMock(spec=VideoFile)
    mock_video.get_width_pixels.return_value = 1920
    mock_video.get_height_pixels.return_value = 1080
    mock_video.get_duration.return_value = 60.0

    empty_speaker_segments = []
    scene_changes = [10.0, 20.0, 30.0]
    aspect_ratio = (9, 16)

    with patch('logging.info') as mock_log_info:
        crops_result = resizer.resize(
            video_file=mock_video,
            speaker_segments=empty_speaker_segments,
            scene_changes=scene_changes,
            aspect_ratio=aspect_ratio
        )

    assert isinstance(crops_result, Crops)
    assert len(crops_result.segments) == 1

    fallback_segment = crops_result.segments[0]
    assert isinstance(fallback_segment, Segment)
    assert fallback_segment.speakers == []
    assert fallback_segment.start_time == 0.0
    assert fallback_segment.end_time == 60.0

    expected_resize_width = 607
    expected_resize_height = 1080
    expected_x = (1920 - 607) // 2
    expected_y = (1080 - 1080) // 2

    assert fallback_segment.x == expected_x
    assert fallback_segment.y == expected_y

    assert crops_result.original_width == 1920
    assert crops_result.original_height == 1080
    assert crops_result.crop_width == expected_resize_width
    assert crops_result.crop_height == expected_resize_height

    mock_log_info.assert_any_call(
        "No speaker segments provided or diarization skipped. "
        "Applying default center crop for the entire video."
    )
