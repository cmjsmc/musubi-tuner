# run_secure_hybrid_training.py
import io
import argparse
import sys
import os
import tempfile
import atexit
import shutil
import hashlib
from collections import defaultdict

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
    """Creates temporary directories on the real filesystem for outputs and caches."""
    session_dir = tempfile.mkdtemp()
    real_output_dir = os.path.join(session_dir, "outputs")
    real_cache_dir = os.path.join(session_dir, "caches")
    os.makedirs(real_output_dir, exist_ok=True)
    os.makedirs(real_cache_dir, exist_ok=True)
    
    print(f"Real disk storage configured:")
    print(f"  - Final models will be saved to: {real_output_dir}")
    print(f"  - Caches will be saved to:      {real_cache_dir}")

    def cleanup():
        print(f"\nCleaning up temporary directory: {session_dir}")
        shutil.rmtree(session_dir, ignore_errors=True)
    
    atexit.register(cleanup)
    return real_output_dir, real_cache_dir


def define_training_parameters(real_output_dir, real_cache_dir):
    """Defines arguments, distinguishing between virtual and real paths."""
    args = argparse.Namespace()
    
    # --- Virtual (In-Memory) Paths ---
    virtual_workspace = "/workspace"
    virtual_model_dir = f"{virtual_workspace}/models"
    # This path is where the anonymized dataset will reside in memory.
    args.virtual_dataset_dir = f"{virtual_workspace}/datasets/train_data"

    # --- Real (On-Disk) Paths ---
    args.output_dir = real_output_dir
    args.cache_dir = real_cache_dir
    
    # --- File/Directory Configs ---
    args.dataset_config = f"{virtual_workspace}/dataset_config.toml"
    args.pretrained_model_name_or_path = f"{virtual_model_dir}/qwen-vl-moe-v2.fp16.safetensors"
    args.vae = f"{virtual_model_dir}/qwen-vl-moe-vae.fp16.safetensors"
    args.text_encoder = f"{virtual_model_dir}/Qwen2.5-0.5B-VL-Chat"
    args.logging_dir = os.path.join(real_output_dir, "logs")
    args.sample_prompts = f"{virtual_workspace}/prompts.txt"
    args.output_name = "qwen_lora_model"

    # --- Other Training Arguments ---
    args.batch_size = 4
    args.num_workers = 2
    args.max_resolution = "1024,1024"
    args.skip_existing = False; args.keep_cache = True; args.debug_mode = None
    args.disable_cudnn_backend = False; args.max_train_steps = 1500
    args.learning_rate = 1e-4; args.optimizer_type = "AdamW8bit"
    args.network_dim = 128; args.network_alpha = 64; args.lr_scheduler = "cosine"
    args.save_every_n_steps = 500; args.save_model_as = "safetensors"
    args.network_module = "networks.lora"; args.gradient_checkpointing = True
    args.mixed_precision = "bf16"; args.log_with = "tensorboard"
    args.sample_every_n_steps = 250; args.config_file = None; args.edit = True
    args.edit_plus = False; args.fp8_vl = False; args.fp8_scaled = False
    args.num_layers = None; args.split_attn = True; args.device = "cuda"
    args.console_width = None; args.console_back = None
    args.console_num_images = None; args.vae_dtype = None

    return args


def get_archive_data_in_memory(cache_dir_path: str, virtual_dataset_dir: str) -> io.BytesIO:
    """Creates a dummy 7z archive in memory with the correct config."""
    in_memory_7z = io.BytesIO()
    
    dataset_toml_content = f'''
[general]
resolution = [1024, 1024]
shuffle_caption = true

[[datasets]]
root_dir = "{virtual_dataset_dir}"
cache_dir = "{cache_dir_path}"
'''.encode('utf-8')

    with py7zr.SevenZipFile(in_memory_7z, 'w') as archive:
        files_to_add = {
            "workspace/dataset_config.toml": dataset_toml_content,
            "workspace/prompts.txt": b'{"prompt": "a beautiful landscape"}',
            "workspace/datasets/train_data/sensitive_image_name_1.png": b"dummy_png_1",
            "workspace/datasets/train_data/sensitive_image_name_1.txt": b"sensitive caption 1",
            "workspace/datasets/train_data/another_sensitive_name.png": b"dummy_png_2",
            "workspace/datasets/train_data/another_sensitive_name.txt": b"sensitive caption 2",
            "workspace/models/qwen-vl-moe-vae.fp16.safetensors": b"",
            "workspace/models/qwen-vl-moe-v2.fp16.safetensors": b"",
            "workspace/models/Qwen2.5-0.5B-VL-Chat/config.json": b'{"status": "ok"}',
        }
        archive.writed(files_to_add)
        
    in_memory_7z.seek(0)
    return in_memory_7z, None # No password


def anonymize_virtual_dataset(root_dir: str):
    """
    Walks a virtual directory, renaming files to a hash of their basename.
    This is performed entirely in memory on the fake filesystem.
    """
    print(f"\nAnonymizing in-memory filenames inside '{root_dir}'...")
    if not os.path.exists(root_dir):
        print(f"Warning: Virtual directory '{root_dir}' not found. Skipping anonymization.")
        return

    files_by_basename = defaultdict(list)
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            basename, ext = os.path.splitext(filename)
            full_path = os.path.join(dirpath, filename)
            files_by_basename[basename].append((full_path, ext))

    renamed_count = 0
    for basename, files in files_by_basename.items():
        # Use a cryptographic hash for robust, collision-resistant names
        hashed_basename = hashlib.sha256(basename.encode('utf-8')).hexdigest()
        
        for full_path, ext in files:
            dirpath = os.path.dirname(full_path)
            new_path = os.path.join(dirpath, f"{hashed_basename}{ext}")
            os.rename(full_path, new_path)
            renamed_count += 1
            # print(f"  - Renamed {os.path.basename(full_path)} -> {os.path.basename(new_path)}")

    print(f"Anonymization complete. Renamed {renamed_count} files in memory.")


def main():
    """Main orchestrator for the secure hybrid training pipeline."""
    real_output_dir, real_cache_dir = setup_real_directories()
    args = define_training_parameters(real_output_dir, real_cache_dir)
    archive_data_stream, archive_password = get_archive_data_in_memory(
        args.cache_dir, args.virtual_dataset_dir
    )

    with Patcher() as patcher:
        fs = patcher.fs
        fs.add_real_directory(real_output_dir)
        fs.add_real_directory(real_cache_dir)
        
        print("\nExtracting archive into virtual filesystem...")
        with py7zr.SevenZipFile(archive_data_stream, 'r', password=archive_password) as archive:
            archive.extractall(path="/")
        print("Extraction complete.")

        # --- NEW ANONYMIZATION STEP ---
        # Rename the sensitive source files in memory before any scripts see them.
        anonymize_virtual_dataset(args.virtual_dataset_dir)
        
        # --- Execute Pipeline ---
        print("\n--- STAGE 1: Caching Text Encoder Outputs ---")
        cache_encoder_script.main_exec(args)
        print("--- STAGE 1 COMPLETE ---\n")

        print("--- STAGE 2: Caching Latents ---")
        cache_latents_script.main_exec(args)
        print("--- STAGE 2 COMPLETE ---\n")
        
        print("--- STAGE 3: Training Network ---")
        train_network_script.main_exec(args)
        print("--- STAGE 3 COMPLETE ---\n")

        print(">>> Secure hybrid training pipeline finished successfully. <<<")
        final_model_path = os.path.join(args.output_dir, f"{args.output_name}.{args.save_model_as}")
        print(f"\nFinal model and caches saved to real disk in: {os.path.dirname(real_output_dir)}")
        print(f"All sensitive source data was processed exclusively in memory.")


if __name__ == "__main__":
    main()
