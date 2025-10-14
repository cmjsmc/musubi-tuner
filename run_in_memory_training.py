# run_hybrid_training.py
import io
import argparse
import sys
import os
import tempfile
import atexit
import shutil

# Third-party libraries must be installed
try:
    import py7zr
    from pyfakefs.fake_filesystem_patcher import Patcher
except ImportError:
    print("Error: Required packages are not installed.")
    print("Please run: pip install pyfakefs py7zr")
    sys.exit(1)

# --- Import the refactored scripts ---
try:
    from musubi_tuner import qwen_image_cache_text_encoder_outputs as cache_encoder_script
    from musubi_tuner import qwen_image_cache_latents as cache_latents_script
    from src.musubi_tuner import qwen_image_train_network as train_network_script
except ImportError as e:
    print(f"Error importing training modules: {e}")
    print("Please ensure this script is run from the correct directory or adjust sys.path.")
    sys.exit(1)


def setup_real_directories():
    """
    Creates temporary directories on the real filesystem for outputs and caches.
    Returns the paths to these directories.
    """
    # Create a single temporary base directory for the session
    # This makes cleanup easier and keeps all outputs organized.
    session_dir = tempfile.mkdtemp()
    
    real_output_dir = os.path.join(session_dir, "outputs")
    real_cache_dir = os.path.join(session_dir, "caches")
    
    os.makedirs(real_output_dir, exist_ok=True)
    os.makedirs(real_cache_dir, exist_ok=True)
    
    print(f"Real disk storage configured:")
    print(f"  - Final models will be saved to: {real_output_dir}")
    print(f"  - Caches will be saved to:      {real_cache_dir}")

    # Register a cleanup function to remove the temp directory on exit
    def cleanup():
        print(f"\nCleaning up temporary directory: {session_dir}")
        shutil.rmtree(session_dir, ignore_errors=True)
    
    atexit.register(cleanup)

    return real_output_dir, real_cache_dir


def define_training_parameters(real_output_dir, real_cache_dir):
    """
    Define arguments for the pipeline. Paths now distinguish between
    virtual (in-memory) and real (on-disk) locations.
    """
    args = argparse.Namespace()

    # --- Virtual (In-Memory) Paths ---
    virtual_workspace = "/workspace"
    virtual_model_dir = f"{virtual_workspace}/models"

    # --- Real (On-Disk) Paths ---
    # These paths were created on the actual filesystem.
    args.output_dir = real_output_dir
    # The dataset config will point to this real directory for caches.
    args.cache_dir = real_cache_dir
    
    # --- File/Directory Configs ---
    args.dataset_config = f"{virtual_workspace}/dataset_config.toml"
    args.pretrained_model_name_or_path = f"{virtual_model_dir}/qwen-vl-moe-v2.fp16.safetensors"
    args.vae = f"{virtual_model_dir}/qwen-vl-moe-vae.fp16.safetensors"
    args.text_encoder = f"{virtual_model_dir}/Qwen2.5-0.5B-VL-Chat"
    args.logging_dir = os.path.join(real_output_dir, "logs") # Also on real disk
    args.sample_prompts = f"{virtual_workspace}/prompts.txt"
    args.output_name = "qwen_lora_model"

    # --- Caching & Training Arguments (same as before) ---
    args.batch_size = 4
    args.num_workers = 2
    args.max_resolution = "1024,1024"
    args.skip_existing = False
    args.keep_cache = True
    args.debug_mode = None
    args.disable_cudnn_backend = False
    args.max_train_steps = 1500
    args.learning_rate = 1e-4
    args.optimizer_type = "AdamW8bit"
    args.network_dim = 128
    args.network_alpha = 64
    args.lr_scheduler = "cosine"
    args.save_every_n_steps = 500
    args.save_model_as = "safetensors"
    args.network_module = "networks.lora"
    args.gradient_checkpointing = True
    args.mixed_precision = "bf16"
    args.log_with = "tensorboard"
    args.sample_every_n_steps = 250
    args.config_file = None
    args.edit = True
    args.edit_plus = False
    args.fp8_vl = False
    args.fp8_scaled = False
    args.num_layers = None
    args.split_attn = True 
    args.device = "cuda"
    args.console_width = None
    args.console_back = None
    args.console_num_images = None
    args.vae_dtype = None

    return args


def get_archive_data_in_memory(cache_dir_path: str) -> io.BytesIO:
    """
    Creates a dummy 7z archive in memory.
    Crucially, the dataset_config.toml inside the archive is configured
    to use the real disk path for its cache directory.
    """
    in_memory_7z = io.BytesIO()
    password = None # No password for this example

    # This TOML content will be written into the in-memory archive.
    # It tells the scripts where to read source images (virtual path)
    # and where to write the caches (real path).
    dataset_toml_content = f'''
[general]
resolution = [1024, 1024]
shuffle_caption = true

[[datasets]]
root_dir = "/workspace/datasets/train_data"
cache_dir = "{cache_dir_path}"
'''.encode('utf-8')

    with py7zr.SevenZipFile(in_memory_7z, 'w', password=password) as archive:
        files_to_add = {
            "workspace/dataset_config.toml": dataset_toml_content,
            "workspace/prompts.txt": b'{"prompt": "a beautiful landscape"}',
            "workspace/datasets/train_data/img1.png": b"dummy_png_bytes_1",
            "workspace.datasets/train_data/img1.txt": b"caption 1",
            # Placeholders for models so path checks in scripts don't fail
            "workspace/models/qwen-vl-moe-vae.fp16.safetensors": b"",
            "workspace/models/qwen-vl-moe-v2.fp16.safetensors": b"",
            "workspace/models/Qwen2.5-0.5B-VL-Chat/config.json": b'{"status": "ok"}',
        }
        archive.writed(files_to_add)

    in_memory_7z.seek(0)
    return in_memory_7z, password


def main():
    """Main orchestrator for the hybrid in-memory/on-disk training pipeline."""
    
    # 1. Create the directories on the real disk that we will write to.
    real_output_dir, real_cache_dir = setup_real_directories()

    # 2. Define parameters, using the real disk paths where needed.
    args = define_training_parameters(real_output_dir, real_cache_dir)
    
    # 3. Create the in-memory archive, injecting the real cache path into its config.
    archive_data_stream, archive_password = get_archive_data_in_memory(args.cache_dir)

    print("\nStarting in-memory filesystem patcher with hybrid mode...")
    with Patcher() as patcher:
        fs = patcher.fs

        # 4. Mount the real directories into the virtual filesystem.
        # Any I/O to these paths will now pass through to the real disk.
        fs.add_real_directory(real_output_dir)
        fs.add_real_directory(real_cache_dir)
        print(f"Mounted real directories into virtual filesystem.")

        # 5. Extract the archive into the virtual root.
        # The NSFW images/text now exist only in memory at '/workspace/datasets'.
        with py7zr.SevenZipFile(archive_data_stream, 'r', password=archive_password) as archive:
            archive.extractall(path="/")
        print("Archive extracted into memory.")
        
        # --- Execute Pipeline ---
        print("\n--- STAGE 1: Caching Text Encoder Outputs ---")
        # Reads from /workspace/datasets (memory), writes to args.cache_dir (disk).
        cache_encoder_script.main_exec(args)
        print("--- STAGE 1 COMPLETE ---\n")

        print("--- STAGE 2: Caching Latents ---")
        # Reads from /workspace/datasets (memory), writes to args.cache_dir (disk).
        cache_latents_script.main_exec(args)
        print("--- STAGE 2 COMPLETE ---\n")
        
        print("--- STAGE 3: Training Network ---")
        # Reads caches from args.cache_dir (disk), writes model to args.output_dir (disk).
        train_network_script.main_exec(args)
        print("--- STAGE 3 COMPLETE ---\n")

        print(">>> Hybrid training pipeline finished successfully. <<<")
        
        final_model_path = os.path.join(args.output_dir, f"{args.output_name}.{args.save_model_as}")
        print(f"\nFinal model saved to real disk at: {final_model_path}")


if __name__ == "__main__":
    main()
