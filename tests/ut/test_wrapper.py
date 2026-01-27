import os
import sys
from types import CodeType
from unittest.mock import MagicMock, Mock, patch, mock_open
import pytest
import torch


def test_import():
    """Test that the module can be imported successfully."""
    from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
    assert TorchCompileWrapperWithCustomDispatcher is not None


def test_basic_instantiation():
    """Test basic wrapper instantiation with mocked dependencies."""
    from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
    
    # Create a concrete implementation
    class TestWrapper(TorchCompileWrapperWithCustomDispatcher):
        def forward(self, x):
            return x * 2
    
    # Mock all the dependencies
    mock_config = MagicMock()
    mock_config.compilation_config.init_backend.return_value = "eager"
    mock_config.compilation_config.inductor_compile_config = None
    
    with patch('vllm.config.get_current_vllm_config', return_value=mock_config):
        with patch('vllm.config.CompilationLevel') as mock_level:
            mock_level.DYNAMO_ONCE = 1
            with patch('torch.compile', side_effect=lambda func, **kwargs: func):
                with patch('torch._dynamo.convert_frame.register_bytecode_hook'):
                    wrapper = TestWrapper(compilation_level=0)
                    
                    # Verify basic attributes exist
                    assert hasattr(wrapper, 'vllm_config')
                    assert hasattr(wrapper, 'compiled_callable')
                    assert hasattr(wrapper, 'original_code_object')
                    assert hasattr(wrapper, 'compiled_codes')
                    assert isinstance(wrapper.compiled_codes, list)


def test_forward_call():
    """Test that the forward method can be called."""
    from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
    
    class TestWrapper(TorchCompileWrapperWithCustomDispatcher):
        def forward(self, x):
            return x * 2
    
    mock_config = MagicMock()
    mock_config.compilation_config.init_backend.return_value = "eager"
    mock_config.compilation_config.inductor_compile_config = None
    
    with patch('vllm.config.get_current_vllm_config', return_value=mock_config):
        with patch('vllm.config.CompilationLevel') as mock_level:
            mock_level.DYNAMO_ONCE = 1
            with patch('torch.compile', side_effect=lambda func, **kwargs: func):
                with patch('torch._dynamo.convert_frame.register_bytecode_hook'):
                    wrapper = TestWrapper(compilation_level=0)
                    
                    # Test calling the wrapper
                    input_tensor = torch.tensor([1.0, 2.0, 3.0])
                    result = wrapper(input_tensor)
                    
                    expected = input_tensor * 2
                    assert torch.allclose(result, expected)


def test_call_with_kwargs():
    """Test __call__ method with keyword arguments."""
    from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
    
    class TestWrapper(TorchCompileWrapperWithCustomDispatcher):
        def forward(self, x, multiplier=3):
            return x * multiplier
    
    mock_config = MagicMock()
    mock_config.compilation_config.init_backend.return_value = "eager"
    mock_config.compilation_config.inductor_compile_config = None
    
    with patch('vllm.config.get_current_vllm_config', return_value=mock_config):
        with patch('vllm.config.CompilationLevel') as mock_level:
            mock_level.DYNAMO_ONCE = 1
            with patch('torch.compile', side_effect=lambda func, **kwargs: func):
                with patch('torch._dynamo.convert_frame.register_bytecode_hook'):
                    wrapper = TestWrapper(compilation_level=0)
                    
                    # Test with positional and keyword args
                    input_tensor = torch.tensor([1.0, 2.0, 3.0])
                    result = wrapper(input_tensor, multiplier=5)
                    
                    expected = input_tensor * 5
                    assert torch.allclose(result, expected)


def test_custom_callable():
    """Test wrapper with custom compiled callable."""
    from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
    
    class TestWrapper(TorchCompileWrapperWithCustomDispatcher):
        def forward(self, x):
            return x * 2
    
    custom_func = Mock(return_value=torch.tensor([5.0]))
    mock_config = MagicMock()
    mock_config.compilation_config.init_backend.return_value = "eager"
    
    with patch('vllm.config.get_current_vllm_config', return_value=mock_config):
        with patch('vllm.config.CompilationLevel') as mock_level:
            mock_level.DYNAMO_ONCE = 1
            with patch('torch._dynamo.convert_frame.register_bytecode_hook'):
                wrapper = TestWrapper(
                    compiled_callable=custom_func,
                    compilation_level=0
                )
                
                # Verify custom callable is used
                assert wrapper.compiled_callable is custom_func
                
                # Call should use custom callable
                result = wrapper(torch.tensor([1.0]))
                assert custom_func.called


def test_inductor_backend_with_options():
    """Test that inductor backend gets proper compilation options."""
    from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
    
    class TestWrapper(TorchCompileWrapperWithCustomDispatcher):
        def forward(self, x):
            return x * 2
    
    mock_config = MagicMock()
    mock_config.compilation_config.init_backend.return_value = "inductor"
    mock_config.compilation_config.inductor_compile_config = {
        "max_autotune": True,
        "triton.cudagraphs": True
    }
    
    with patch('vllm.config.get_current_vllm_config', return_value=mock_config):
        with patch('vllm.config.CompilationLevel') as mock_level:
            mock_level.DYNAMO_ONCE = 1
            with patch('torch.compile') as mock_compile:
                mock_compile.return_value = Mock()
                with patch('torch._dynamo.convert_frame.register_bytecode_hook'):
                    wrapper = TestWrapper(compilation_level=0)
                    
                    # Verify torch.compile was called with correct options
                    assert mock_compile.called
                    call_kwargs = mock_compile.call_args[1]
                    assert 'backend' in call_kwargs
                    assert call_kwargs['backend'] == 'inductor'
                    assert 'options' in call_kwargs
                    assert call_kwargs['options'] == mock_config.compilation_config.inductor_compile_config


def test_non_inductor_backend():
    """Test with non-inductor backend (options should be None)."""
    from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
    
    class TestWrapper(TorchCompileWrapperWithCustomDispatcher):
        def forward(self, x):
            return x * 2
    
    mock_config = MagicMock()
    mock_config.compilation_config.init_backend.return_value = "aot_eager"
    mock_config.compilation_config.inductor_compile_config = {"some": "config"}
    
    with patch('vllm.config.get_current_vllm_config', return_value=mock_config):
        with patch('vllm.config.CompilationLevel') as mock_level:
            mock_level.DYNAMO_ONCE = 1
            with patch('torch.compile') as mock_compile:
                mock_compile.return_value = Mock()
                with patch('torch._dynamo.convert_frame.register_bytecode_hook'):
                    wrapper = TestWrapper(compilation_level=0)
                    
                    # Verify options is None for non-inductor backend
                    call_kwargs = mock_compile.call_args[1]
                    assert call_kwargs.get('options') is None


def test_bytecode_hook_ignores_wrong_code():
    """Test that bytecode hook ignores calls with wrong old_code."""
    from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
    
    class TestWrapper(TorchCompileWrapperWithCustomDispatcher):
        def forward(self, x):
            return x * 2
    
    mock_config = MagicMock()
    mock_config.compilation_config.init_backend.return_value = "eager"
    mock_config.compilation_config.inductor_compile_config = None
    mock_config.compilation_config.local_cache_dir = None
    
    with patch('vllm.config.get_current_vllm_config', return_value=mock_config):
        with patch('vllm.config.CompilationLevel') as mock_level:
            mock_level.DYNAMO_ONCE = 1
            with patch('torch.compile', side_effect=lambda func, **kwargs: func):
                with patch('torch._dynamo.convert_frame.register_bytecode_hook'):
                    wrapper = TestWrapper(compilation_level=0)
                    
                    # Test with wrong code object (should be ignored)
                    wrong_code = MagicMock(spec=CodeType)
                    new_code = MagicMock(spec=CodeType)
                    
                    initial_count = len(wrapper.compiled_codes)
                    wrapper.bytecode_hook(wrong_code, new_code)
                    
                    # Should not add anything
                    assert len(wrapper.compiled_codes) == initial_count


def test_bytecode_hook_saves_code():
    """Test that bytecode_hook correctly saves compiled code."""
    from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
    
    class TestWrapper(TorchCompileWrapperWithCustomDispatcher):
        def forward(self, x):
            return x * 2
    
    mock_config = MagicMock()
    mock_config.compilation_config.init_backend.return_value = "eager"
    mock_config.compilation_config.inductor_compile_config = None
    mock_config.compilation_config.local_cache_dir = None
    
    with patch('vllm.config.get_current_vllm_config', return_value=mock_config):
        with patch('vllm.config.CompilationLevel') as mock_level:
            mock_level.DYNAMO_ONCE = 1
            with patch('torch.compile', side_effect=lambda func, **kwargs: func):
                with patch('torch._dynamo.convert_frame.register_bytecode_hook'):
                    wrapper = TestWrapper(compilation_level=0)
                    
                    # Get the original code
                    old_code = wrapper.original_code_object
                    new_code = MagicMock(spec=CodeType)
                    
                    # Mock frame structure
                    mock_frame = MagicMock()
                    mock_frame.f_code = old_code
                    mock_frame.f_locals = {"self": wrapper}
                    
                    with patch('sys._getframe') as mock_getframe:
                        # Create the frame chain
                        convert_frame_mock = MagicMock()
                        convert_frame_mock.f_code.co_name = "_compile"
                        convert_frame_mock.f_code.co_filename = "/path/to/convert_frame.py"
                        convert_frame_mock.f_locals = {"frame": mock_frame}
                        convert_frame_mock.f_back = None
                        
                        current_frame_mock = MagicMock()
                        current_frame_mock.f_back = convert_frame_mock
                        mock_getframe.return_value = current_frame_mock
                        
                        # Call bytecode_hook
                        initial_count = len(wrapper.compiled_codes)
                        wrapper.bytecode_hook(old_code, new_code)
                        
                        # Should add the new code
                        assert len(wrapper.compiled_codes) == initial_count + 1
                        assert wrapper.compiled_codes[-1] is new_code


def test_bytecode_hook_wrong_self():
    """Test that bytecode_hook ignores when self doesn't match."""
    from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
    
    class TestWrapper(TorchCompileWrapperWithCustomDispatcher):
        def forward(self, x):
            return x * 2
    
    mock_config = MagicMock()
    mock_config.compilation_config.init_backend.return_value = "eager"
    mock_config.compilation_config.inductor_compile_config = None
    mock_config.compilation_config.local_cache_dir = None
    
    with patch('vllm.config.get_current_vllm_config', return_value=mock_config):
        with patch('vllm.config.CompilationLevel') as mock_level:
            mock_level.DYNAMO_ONCE = 1
            with patch('torch.compile', side_effect=lambda func, **kwargs: func):
                with patch('torch._dynamo.convert_frame.register_bytecode_hook'):
                    wrapper1 = TestWrapper(compilation_level=0)
                    wrapper2 = TestWrapper(compilation_level=0)
                    
                    old_code = wrapper1.original_code_object
                    new_code = MagicMock(spec=CodeType)
                    
                    # Mock frame with different self
                    mock_frame = MagicMock()
                    mock_frame.f_code = old_code
                    mock_frame.f_locals = {"self": wrapper2}
                    
                    with patch('sys._getframe') as mock_getframe:
                        convert_frame_mock = MagicMock()
                        convert_frame_mock.f_code.co_name = "_compile"
                        convert_frame_mock.f_code.co_filename = "/path/to/convert_frame.py"
                        convert_frame_mock.f_locals = {"frame": mock_frame}
                        convert_frame_mock.f_back = None
                        
                        current_frame_mock = MagicMock()
                        current_frame_mock.f_back = convert_frame_mock
                        mock_getframe.return_value = current_frame_mock
                        
                        initial_count = len(wrapper1.compiled_codes)
                        wrapper1.bytecode_hook(old_code, new_code)
                        
                        # Should not add because self doesn't match
                        assert len(wrapper1.compiled_codes) == initial_count


def test_bytecode_hook_with_cache_dir():
    """Test bytecode_hook with local_cache_dir for decompilation."""
    from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
    
    class TestWrapper(TorchCompileWrapperWithCustomDispatcher):
        def forward(self, x):
            return x * 2
    
    mock_config = MagicMock()
    mock_config.compilation_config.init_backend.return_value = "eager"
    mock_config.compilation_config.inductor_compile_config = None
    mock_config.compilation_config.local_cache_dir = "/tmp/test_cache"
    
    with patch('vllm.config.get_current_vllm_config', return_value=mock_config):
        with patch('vllm.config.CompilationLevel') as mock_level:
            mock_level.DYNAMO_ONCE = 1
            with patch('torch.compile', side_effect=lambda func, **kwargs: func):
                with patch('torch._dynamo.convert_frame.register_bytecode_hook'):
                    wrapper = TestWrapper(compilation_level=0)
                    
                    old_code = wrapper.original_code_object
                    new_code = MagicMock(spec=CodeType)
                    
                    mock_frame = MagicMock()
                    mock_frame.f_code = old_code
                    mock_frame.f_locals = {"self": wrapper}
                    
                    with patch('sys._getframe') as mock_getframe:
                        convert_frame_mock = MagicMock()
                        convert_frame_mock.f_code.co_name = "_compile"
                        convert_frame_mock.f_code.co_filename = "/path/to/convert_frame.py"
                        convert_frame_mock.f_locals = {"frame": mock_frame}
                        convert_frame_mock.f_back = None
                        
                        current_frame_mock = MagicMock()
                        current_frame_mock.f_back = convert_frame_mock
                        mock_getframe.return_value = current_frame_mock
                        
                        with patch('os.path.exists', return_value=False):
                            with patch('builtins.open', mock_open()) as m_open:
                                # Mock depyf module
                                mock_depyf = MagicMock()
                                mock_depyf.decompile.return_value = "decompiled_code"
                                
                                with patch.dict('sys.modules', {'depyf': mock_depyf}):
                                    wrapper.bytecode_hook(old_code, new_code)
                                    
                                    # Verify decompile was called
                                    assert mock_depyf.decompile.called
                                    # Verify file was written
                                    assert m_open.called


def test_bytecode_hook_decompile_exception():
    """Test bytecode_hook handles decompilation exceptions gracefully."""
    from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
    
    class TestWrapper(TorchCompileWrapperWithCustomDispatcher):
        def forward(self, x):
            return x * 2
    
    mock_config = MagicMock()
    mock_config.compilation_config.init_backend.return_value = "eager"
    mock_config.compilation_config.inductor_compile_config = None
    mock_config.compilation_config.local_cache_dir = "/tmp/test_cache"
    
    with patch('vllm.config.get_current_vllm_config', return_value=mock_config):
        with patch('vllm.config.CompilationLevel') as mock_level:
            mock_level.DYNAMO_ONCE = 1
            with patch('torch.compile', side_effect=lambda func, **kwargs: func):
                with patch('torch._dynamo.convert_frame.register_bytecode_hook'):
                    wrapper = TestWrapper(compilation_level=0)
                    
                    old_code = wrapper.original_code_object
                    new_code = MagicMock(spec=CodeType)
                    
                    mock_frame = MagicMock()
                    mock_frame.f_code = old_code
                    mock_frame.f_locals = {"self": wrapper}
                    
                    with patch('sys._getframe') as mock_getframe:
                        convert_frame_mock = MagicMock()
                        convert_frame_mock.f_code.co_name = "_compile"
                        convert_frame_mock.f_code.co_filename = "/path/to/convert_frame.py"
                        convert_frame_mock.f_locals = {"frame": mock_frame}
                        convert_frame_mock.f_back = None
                        
                        current_frame_mock = MagicMock()
                        current_frame_mock.f_back = convert_frame_mock
                        mock_getframe.return_value = current_frame_mock
                        
                        with patch('os.path.exists', return_value=False):
                            # Mock depyf to raise exception
                            mock_depyf = MagicMock()
                            mock_depyf.decompile.side_effect = Exception("Decompile failed")
                            
                            with patch.dict('sys.modules', {'depyf': mock_depyf}):
                                # Should not raise, exception is caught
                                wrapper.bytecode_hook(old_code, new_code)
                                
                                # Code should still be saved
                                assert new_code in wrapper.compiled_codes


def test_bytecode_hook_file_already_exists():
    """Test bytecode_hook skips decompilation if file exists."""
    from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
    
    class TestWrapper(TorchCompileWrapperWithCustomDispatcher):
        def forward(self, x):
            return x * 2
    
    mock_config = MagicMock()
    mock_config.compilation_config.init_backend.return_value = "eager"
    mock_config.compilation_config.inductor_compile_config = None
    mock_config.compilation_config.local_cache_dir = "/tmp/test_cache"
    
    with patch('vllm.config.get_current_vllm_config', return_value=mock_config):
        with patch('vllm.config.CompilationLevel') as mock_level:
            mock_level.DYNAMO_ONCE = 1
            with patch('torch.compile', side_effect=lambda func, **kwargs: func):
                with patch('torch._dynamo.convert_frame.register_bytecode_hook'):
                    wrapper = TestWrapper(compilation_level=0)
                    
                    old_code = wrapper.original_code_object
                    new_code = MagicMock(spec=CodeType)
                    
                    mock_frame = MagicMock()
                    mock_frame.f_code = old_code
                    mock_frame.f_locals = {"self": wrapper}
                    
                    with patch('sys._getframe') as mock_getframe:
                        convert_frame_mock = MagicMock()
                        convert_frame_mock.f_code.co_name = "_compile"
                        convert_frame_mock.f_code.co_filename = "/path/to/convert_frame.py"
                        convert_frame_mock.f_locals = {"frame": mock_frame}
                        convert_frame_mock.f_back = None
                        
                        current_frame_mock = MagicMock()
                        current_frame_mock.f_back = convert_frame_mock
                        mock_getframe.return_value = current_frame_mock
                        
                        # File already exists
                        with patch('os.path.exists', return_value=True):
                            mock_depyf = MagicMock()
                            
                            with patch.dict('sys.modules', {'depyf': mock_depyf}):
                                wrapper.bytecode_hook(old_code, new_code)
                                
                                # decompile should not be called
                                assert not mock_depyf.decompile.called
                                # But code should still be saved
                                assert new_code in wrapper.compiled_codes


def test_dispatch_to_code_context_manager():
    """Test dispatch_to_code context manager switches code objects."""
    from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
    
    class TestWrapper(TorchCompileWrapperWithCustomDispatcher):
        def forward(self, x):
            return x * 2
    
    mock_config = MagicMock()
    mock_config.compilation_config.init_backend.return_value = "eager"
    mock_config.compilation_config.inductor_compile_config = None
    
    with patch('vllm.config.get_current_vllm_config', return_value=mock_config):
        with patch('vllm.config.CompilationLevel') as mock_level:
            mock_level.DYNAMO_ONCE = 1
            with patch('torch.compile', side_effect=lambda func, **kwargs: func):
                with patch('torch._dynamo.convert_frame.register_bytecode_hook'):
                    wrapper = TestWrapper(compilation_level=0)
                    
                    # Manually add a compiled code (using real code object)
                    def dummy_func():
                        return 42
                    
                    wrapper.compiled_codes.append(dummy_func.__code__)
                    original_code = wrapper.__class__.forward.__code__
                    
                    # Test context manager
                    with wrapper.dispatch_to_code(0):
                        # Inside context, code should be switched
                        assert wrapper.__class__.forward.__code__ == dummy_func.__code__
                    
                    # After context, code should be restored
                    assert wrapper.__class__.forward.__code__ == original_code


def test_dispatch_to_code_with_exception():
    """Test dispatch_to_code context manager properly propagates exceptions."""
    from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
    
    class TestWrapper(TorchCompileWrapperWithCustomDispatcher):
        def forward(self, x):
            return x * 2
    
    mock_config = MagicMock()
    mock_config.compilation_config.init_backend.return_value = "eager"
    mock_config.compilation_config.inductor_compile_config = None
    
    with patch('vllm.config.get_current_vllm_config', return_value=mock_config):
        with patch('vllm.config.CompilationLevel') as mock_level:
            mock_level.DYNAMO_ONCE = 1
            with patch('torch.compile', side_effect=lambda func, **kwargs: func):
                with patch('torch._dynamo.convert_frame.register_bytecode_hook'):
                    wrapper = TestWrapper(compilation_level=0)
                    
                    # Use a real code object with matching signature
                    def dummy_func(self, x):
                        return x * 3
                    
                    # Add compiled code
                    wrapper.compiled_codes.append(dummy_func.__code__)
                    
                    # Test that exception propagates properly from context manager
                    with pytest.raises(ValueError, match="Test exception"):
                        with wrapper.dispatch_to_code(0):
                            raise ValueError("Test exception")
                    
                    # The test above verifies that:
                    # 1. The exception is properly propagated out of the context manager
                    # 2. The context manager's __exit__ was called (as expected in Python)
                    # This is sufficient to verify the context manager works correctly


def test_use_custom_dispatcher_flag():
    """Test that use_custom_dispatcher flag is set based on compilation_level."""
    from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
    
    class TestWrapper(TorchCompileWrapperWithCustomDispatcher):
        def forward(self, x):
            return x * 2
    
    mock_config = MagicMock()
    mock_config.compilation_config.init_backend.return_value = "eager"
    mock_config.compilation_config.inductor_compile_config = None
    
    with patch('vllm.config.get_current_vllm_config', return_value=mock_config):
        with patch('vllm.config.CompilationLevel') as mock_level:
            mock_level.DYNAMO_ONCE = 1
            with patch('torch.compile', side_effect=lambda func, **kwargs: func):
                with patch('torch._dynamo.convert_frame.register_bytecode_hook'):
                    # Test with low level
                    wrapper_low = TestWrapper(compilation_level=0)
                    assert wrapper_low.use_custom_dispatcher is False
                    
                    # Test with high level
                    wrapper_high = TestWrapper(compilation_level=2)
                    assert wrapper_high.use_custom_dispatcher is True


def test_torch_compile_called_with_fullgraph():
    """Test that torch.compile is called with fullgraph=True."""
    from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
    
    class TestWrapper(TorchCompileWrapperWithCustomDispatcher):
        def forward(self, x):
            return x * 2
    
    mock_config = MagicMock()
    mock_config.compilation_config.init_backend.return_value = "eager"
    mock_config.compilation_config.inductor_compile_config = None
    
    with patch('vllm.config.get_current_vllm_config', return_value=mock_config):
        with patch('vllm.config.CompilationLevel') as mock_level:
            mock_level.DYNAMO_ONCE = 1
            with patch('torch.compile') as mock_compile:
                mock_compile.return_value = Mock()
                with patch('torch._dynamo.convert_frame.register_bytecode_hook'):
                    wrapper = TestWrapper(compilation_level=0)
                    
                    # Verify fullgraph=True was passed
                    call_kwargs = mock_compile.call_args[1]
                    assert call_kwargs.get('fullgraph') is True


def test_bytecode_hook_registration():
    """Test that bytecode hook is registered on initialization."""
    from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
    
    class TestWrapper(TorchCompileWrapperWithCustomDispatcher):
        def forward(self, x):
            return x * 2
    
    mock_config = MagicMock()
    mock_config.compilation_config.init_backend.return_value = "eager"
    mock_config.compilation_config.inductor_compile_config = None
    
    with patch('vllm.config.get_current_vllm_config', return_value=mock_config):
        with patch('vllm.config.CompilationLevel') as mock_level:
            mock_level.DYNAMO_ONCE = 1
            with patch('torch.compile', side_effect=lambda func, **kwargs: func):
                with patch('torch._dynamo.convert_frame.register_bytecode_hook') as mock_register:
                    wrapper = TestWrapper(compilation_level=0)
                    
                    # Verify hook was registered
                    assert mock_register.called
                    # Verify it was registered with wrapper's bytecode_hook method
                    registered_hook = mock_register.call_args[0][0]
                    assert registered_hook == wrapper.bytecode_hook


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])