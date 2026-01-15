import pytest
from unittest.mock import MagicMock, patch
import torch

from vllm_kunlun.distributed.kunlun_communicator import KunlunCommunicator
from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase

class TestKunlunCommunicator:
    
    @pytest.fixture
    def mock_dependencies(self):
        """
        创建一个 Fixture 来准备初始化所需的 Mock 对象。
        """
        cpu_group = MagicMock()
        device_group = MagicMock()
        device = MagicMock()
        unique_name = "test_comm"
        return cpu_group, device_group, device, unique_name

    @pytest.fixture
    def communicator(self, mock_dependencies):
        """
        创建一个经过 Mock 处理的 KunlunCommunicator 实例。
        我们需要 Patch 掉 __init__ 中的 cuda 操作和父类初始化。
        """
        cpu_group, device_group, device, unique_name = mock_dependencies

        # Patch 掉 DeviceCommunicatorBase 的 __init__ 和 all_reduce (防止 warmup 真的执行)
        with patch.object(DeviceCommunicatorBase, '__init__', return_value=None) as mock_base_init, \
             patch.object(DeviceCommunicatorBase, 'all_reduce', return_value=None) as mock_base_all_reduce, \
             patch('torch.cuda.Stream', return_value=MagicMock()) as mock_stream, \
             patch('torch.cuda.device', return_value=MagicMock()) as mock_device_ctx, \
             patch('torch.zeros', return_value=MagicMock()):
            
            comm = KunlunCommunicator(device, device_group, cpu_group, unique_name)
            
            # 这里的 available 属性在源码中未定义，可能是基类属性，
            # 我们手动 Mock 它以便测试 change_state
            comm.available = True 
            
            return comm

    def test_init(self, mock_dependencies):
        """
        测试初始化过程：
        1. 验证父类 __init__ 被调用
        2. 验证 Stream 被创建
        3. 验证执行了 Warmup (all_reduce)
        """
        cpu_group, device_group, device, unique_name = mock_dependencies

        with patch.object(DeviceCommunicatorBase, '__init__', return_value=None) as mock_base_init, \
             patch.object(DeviceCommunicatorBase, 'all_reduce', return_value=None) as mock_base_all_reduce, \
             patch('torch.cuda.Stream') as mock_stream_cls, \
             patch('torch.cuda.device') as mock_device_ctx, \
             patch('torch.zeros') as mock_zeros:

            # 执行初始化
            comm = KunlunCommunicator(device, device_group, cpu_group, unique_name)

            # 验证父类初始化
            # 修正点：因为是显式调用父类 __init__(self, ...)，所以 Mock 收到的第一个参数是 comm 实例本身
            mock_base_init.assert_called_once_with(comm, cpu_group, device, device_group, unique_name)
            
            # 验证创建了 Stream
            mock_stream_cls.assert_called_once()
            
            # 验证 Warmup 逻辑：创建了 tensor 并调用了 all_reduce
            mock_zeros.assert_called_once()
            mock_base_all_reduce.assert_called_once()
            
            # 验证状态初始化
            assert comm.ca_comm is None
            assert comm.disabled is False

    @patch.object(DeviceCommunicatorBase, 'all_reduce')
    def test_all_reduce(self, mock_base_method, communicator):
        """测试 all_reduce 委托调用"""
        input_tensor = MagicMock()
        communicator.all_reduce(input_tensor)
        mock_base_method.assert_called_once_with(communicator, input_tensor)

    @patch.object(DeviceCommunicatorBase, 'all_gather')
    def test_all_gather(self, mock_base_method, communicator):
        """测试 all_gather 委托调用"""
        input_tensor = MagicMock()
        dim = 1
        communicator.all_gather(input_tensor, dim)
        mock_base_method.assert_called_once_with(communicator, input_tensor, dim)

    @patch.object(DeviceCommunicatorBase, 'gather')
    def test_gather(self, mock_base_method, communicator):
        """测试 gather 委托调用"""
        input_tensor = MagicMock()
        dst = 0
        dim = 1
        communicator.gather(input_tensor, dst, dim)
        mock_base_method.assert_called_once_with(communicator, input_tensor, dst, dim)

    @patch.object(DeviceCommunicatorBase, 'send')
    def test_send(self, mock_base_method, communicator):
        """测试 send 委托调用"""
        tensor = MagicMock()
        dst = 1
        communicator.send(tensor, dst)
        mock_base_method.assert_called_once_with(communicator, tensor, dst)

    @patch.object(DeviceCommunicatorBase, 'recv')
    def test_recv(self, mock_base_method, communicator):
        """测试 recv 委托调用"""
        size = (10,)
        dtype = torch.float32
        src = 0
        communicator.recv(size, dtype, src)
        mock_base_method.assert_called_once_with(communicator, size, dtype, src)

    def test_destroy(self, communicator):
        """测试 destroy 方法 (虽然它目前是 pass)"""
        try:
            communicator.destroy()
        except Exception as e:
            pytest.fail(f"destroy() raised check exception: {e}")

    def test_change_state_explicit(self, communicator):
        """
        测试 change_state 上下文管理器: 显式指定 enable 和 stream
        """
        original_stream = communicator.stream
        new_stream = MagicMock()
        
        # 测试：启用 (enable=True -> disabled=False) 并更换 stream
        with communicator.change_state(enable=True, stream=new_stream):
            assert communicator.disabled is False
            assert communicator.stream == new_stream
        
        # 退出上下文后，应该恢复原状
        assert communicator.stream == original_stream
        assert communicator.disabled is False  # 初始状态是 False

        # 测试：禁用 (enable=False -> disabled=True)
        with communicator.change_state(enable=False, stream=new_stream):
            assert communicator.disabled is True
        
        assert communicator.disabled is False # 恢复

    def test_change_state_defaults(self, communicator):
        """
        测试 change_state 上下文管理器: 使用默认值 (None)
        """
        original_stream = communicator.stream
        communicator.available = False # 模拟底层属性
        
        # 测试：enable=None (应该取 self.available 的值，即 False -> disabled=True)
        # 测试：stream=None (应该保持原 stream)
        with communicator.change_state(enable=None, stream=None):
            assert communicator.disabled is True # 因为 available 是 False
            assert communicator.stream == original_stream
        
        # 恢复
        assert communicator.disabled is False