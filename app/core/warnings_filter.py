"""
Warnings filter to suppress known deprecation warnings from dependencies.
"""
import warnings
import os


def suppress_known_warnings():
    """Suppress known deprecation warnings from dependencies."""
    
    # Suppress pkg_resources deprecation warning
    warnings.filterwarnings("ignore", 
                          message="pkg_resources is deprecated",
                          category=DeprecationWarning)
    
    # Suppress torchaudio deprecation warnings
    warnings.filterwarnings("ignore", 
                          message=".*list_audio_backends has been deprecated.*",
                          category=UserWarning)
    
    warnings.filterwarnings("ignore", 
                          message=".*StreamingMediaDecoder has been deprecated.*",
                          category=UserWarning)
    
    warnings.filterwarnings("ignore", 
                          message=".*function's implementation will be changed to use torchaudio.load_with_torchcodec.*",
                          category=UserWarning)
    
    # Suppress torch CUDA amp deprecation warnings
    warnings.filterwarnings("ignore", 
                          message=".*torch.cuda.amp.custom_fwd.*is deprecated.*",
                          category=FutureWarning)
    
    # Suppress speechbrain pretrained module deprecation
    warnings.filterwarnings("ignore", 
                          message=".*Module 'speechbrain.pretrained' was deprecated.*",
                          category=UserWarning)