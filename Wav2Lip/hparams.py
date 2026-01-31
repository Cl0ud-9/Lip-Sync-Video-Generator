
import os
from typing import List, Any

def get_image_list(data_root: str, split: str) -> List[str]:
    """
    Reads a filelist from the filelists/ directory and returns absolute paths to images.
    
    Args:
        data_root (str): The root directory where images are stored.
        split (str): The name of the split (e.g., 'train', 'val', 'test').
        
    Returns:
        List[str]: A list of file paths.
    """
    filelist = []

    with open(f'filelists/{split}.txt') as f:
        for line in f:
            line = line.strip()
            if ' ' in line: 
                line = line.split()[0]
            filelist.append(os.path.join(data_root, line))

    return filelist


class HParams:
    """
    Hyperparameters container class.
    Allows access to parameters via dot notation.
    """
    def __init__(self, **kwargs: Any):
        self.data = {}
        for key, value in kwargs.items():
            self.data[key] = value

    def __getattr__(self, key: str) -> Any:
        if key not in self.data:
            raise AttributeError(f"'HParams' object has no attribute {key}")
        return self.data[key]

    def set_hparam(self, key: str, value: Any) -> None:
        self.data[key] = value

    def values(self):
        return self.data


# Default hyperparameters
hparams = HParams(
    num_mels=80,             # Number of mel-spectrogram channels
    rescale=True,            # Whether to rescale audio prior to preprocessing
    rescaling_max=0.9,       # Rescaling value
    
    use_lws=False,           # Use LWS for STFT and phase reconstruction
    
    n_fft=800,               # FFT window size
    hop_size=200,            # Hop size (200 @ 16kHz = 12.5 ms)
    win_size=800,            # Window size (800 @ 16kHz = 50 ms)
    sample_rate=16000,       # 16000Hz
    
    frame_shift_ms=None,     # Can replace hop_size parameter
    
    # Mel and Linear spectrograms normalization/scaling and clipping
    signal_normalization=True,
    allow_clipping_in_normalization=True, 
    symmetric_mels=True,     # Scale data to be symmetric around 0
    max_abs_value=4.,        # Max absolute value of data
    
    # Spectrogram Pre-Emphasis
    preemphasize=True,
    preemphasis=0.97,
    
    # Limits
    min_level_db=-100,
    ref_level_db=20,
    fmin=55,                 # Set to 55 for male speakers, 95 for female (approx)
    fmax=7600,
    
    # Training parameters
    img_size=96,
    fps=25,
    
    batch_size=16,
    initial_learning_rate=1e-4,
    nepochs=20000,           # Reasonable number instead of 200...0
    num_workers=16,
    checkpoint_interval=3000,
    eval_interval=3000,
    save_optimizer_state=True,
    
    syncnet_wt=0.0,
    syncnet_batch_size=64,
    syncnet_lr=1e-4,
    syncnet_eval_interval=10000,
    syncnet_checkpoint_interval=10000,
    
    disc_wt=0.07,
    disc_initial_learning_rate=1e-4,
)


def hparams_debug_string() -> str:
    """
    Returns a formatted string containing all hyperparameters.
    """
    values = hparams.values()
    hp_list = [f"  {name}: {values[name]}" for name in sorted(values) if name != "sentences"]
    return "Hyperparameters:\n" + "\n".join(hp_list)
