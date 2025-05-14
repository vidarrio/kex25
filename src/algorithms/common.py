import os
import torch

# Debug levels
DEBUG_NONE = 0
DEBUG_CRITICAL = 1
DEBUG_INFO = 2
DEBUG_VERBOSE = 3
DEBUG_ALL = 4
DEBUG_SPECIFIC = 5

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get path to models directory (src/models)
def get_model_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, '..', 'models')
    return models_dir

# GPU monitoring
def check_gpu_usage():
    if torch.cuda.is_available():
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

# Process count determination
def determine_process_count(n_jobs):
    """
    Determine the number of processes to use for multiprocessing.
    """
    if n_jobs is None:
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB

            if gpu_mem < 4:
                n_jobs = 4
            elif gpu_mem < 8:
                n_jobs = 8
            elif gpu_mem < 16:
                n_jobs = 16
            else:
                n_jobs = 16
        else:
            # Use CPU count if no GPU available
            n_jobs = max(1, min(16, os.cpu_count() - 2))

    return n_jobs