# run_in_memory_training.py
import io
import argparse
import sys
import os

# Third-party libraries must be installed
try:
    import py7zr
    from pyfakefs.fake_filesystem_patcher import Patcher
except ImportError:
    print("Error: Required packages are not installed.")
    print("Please run: pip install pyfakefs py7zr")
    sys.exit(1)

# --- Import the refactored scripts ---
# This assumes the script is run from a location where these modules are accessible.
# You may need to add the parent directories to sys.path if you get ModuleNotFoundError.
# For example: sys.path.insert(0, '/path/to/your/project')
try:
    from musubi_tuner import qwen_image_cache_text_encoder_outputs as cache_encoder_script
    from musubi_tuner import qwen_image_cache_latents as cache_latents_script
    from src.musubi_tuner import qwen_image_train_network as train_network_script
except ImportError as e:
    print(f"Error importing training modules: {e}")
    print("Please ensure this script is run from the correct directory or adjust sys.path.")
    sys.exit(1)


def define_training_parameters():
    """
    Define all arguments for the training pipeline in one place.
    Paths are set to their intended locations in the VIRTUAL filesystem.
    """
    args = argparse.Namespace()

    # --- Virtual Paths ---
    # These paths exist only in memory. The archive will be extracted here.
    virtual_workspace = "/workspace"
    virtual_dataset_dir = f"{virtual_workspace}/datasets/train_data"
    virtual_output_dir = f"{virtual_workspace}/output"
    virtual_model_dir = f"{virtual_workspace}/models"
    
    # --- File/Directory Configs ---
    args.dataset_config = f"{virtual_workspace}/dataset_config.toml"
    args.output_dir = virtual_output_dir
    args.output_name = "qwen_lora_model"
    args.pretrained_model_name_or_path = f"{virtual_model_dir}/qwen-vl-moe-v2.fp16.safetensors"
    args.vae = f"{virtual_model_dir}/qwen-vl-moe-vae.fp16.safetensors"
    args.text_encoder = f"{virtual_model_dir}/Qwen2.5-0.5B-VL-Chat"
    args.logging_dir = f"{virtual_output_dir}/logs"
    args.sample_prompts = f"{virtual_workspace}/prompts.txt"

    # --- Caching Script Arguments ---
    args.batch_size = 4
    args.num_workers = 2
    args.max_resolution = "1024,1024"
    args.skip_existing = False
    args.keep_cache = True
    args.debug_mode = None
    args.disable_cudnn_backend = False
    
    # --- Training Script Arguments ---
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
    args.config_file = None # We are defining config here, not loading from file.

    # --- Qwen-Image Specific Arguments ---
    args.edit = True
    args.edit_plus = False
    args.fp8_vl = False
    args.fp8_scaled = False
    args.num_layers = None
    args.split_attn = True 
    
    # --- Common Arguments for all scripts ---
    # Device needs to be available for all scripts.
    args.device = "cuda"
    
    # These args are expected by some parsers but might not be used.
    # We define them to prevent parsing errors.
    args.console_width = None
    args.console_back = None
    args.console_num_images = None
    args.vae_dtype = None # Handled automatically in the train script

    return args


def get_archive_data_in_memory() -> io.BytesIO:
    """
    Placeholder to retrieve your 7z archive's byte data.
    This creates a dummy 7z file in memory for demonstration.
    Replace this with your actual data fetching logic (e.g., from a network).
    """
    in_memory_7z = io.BytesIO()
    password = "your-archive-password"  # Can be None if not password-protected

    print("Creating a dummy .7z archive in memory...")
    with py7zr.SevenZipFile(in_memory_7z, 'w', password=password) as archive:
        # Create dummy data as dictionaries {filename: content_in_bytes}
        files_to_add = {
            "workspace/dataset_config.toml": b"""
[general]
resolution = [1024, 1024]
shuffle_caption = true

[[datasets]]
root_dir = "/workspace/datasets/train_data"
""",
            "workspace/prompts.txt": b'{"prompt": "a beautiful landscape"}',
            "workspace/datasets/train_data/img1.png": b"dummy_png_bytes_for_image1",
            "workspace/datasets/train_data/img1.txt": b"caption for image 1",
            "workspace/datasets/train_data/img2.png": b"dummy_png_bytes_for_image2",
            "workspace/datasets/train_data/img2.txt": b"caption for image 2",
            # Create empty placeholder files for models so that path checks don't fail
            "workspace/models/qwen-vl-moe-vae.fp16.safetensors": b"",
            "workspace/models/qwen-vl-moe-v2.fp16.safetensors": b"",
            "workspace/models/Qwen2.5-0.5B-VL-Chat/config.json": b'{"status": "ok"}',
        }
        archive.writed(files_to_add)

    print("Dummy archive created successfully.")
    in_memory_7z.seek(0)
    return in_memory_7z, password


def main():
    """Main orchestrator for the in-memory training pipeline."""
    args = define_training_parameters()
    archive_data_stream, archive_password = get_archive_data_in_memory()

    print("\nStarting in-memory filesystem patcher...")
    with Patcher() as patcher:
        fs = patcher.fs

        print("Extracting .7z archive into virtual filesystem...")
        with py7zr.SevenZipFile(archive_data_stream, 'r', password=archive_password) as archive:
            archive.extractall(path="/")
        print("Extraction complete. All files are now in memory.")

        # Verify a file exists in the virtual filesystem
        if not fs.exists(args.dataset_config):
            print(f"FATAL: Virtual file '{args.dataset_config}' not found after extraction.")
            return

        # Explicitly create output directories in the virtual filesystem
        fs.create_dir(args.output_dir)
        fs.create_dir(args.logging_dir)

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

        print(">>> In-memory training pipeline finished successfully. <<<")
        
        # Optionally, retrieve the final model from the virtual filesystem
        final_model_path = os.path.join(args.output_dir, f"{args.output_name}.{args.save_model_as}")
        final_real_model_path = os.path.join(args.output_dir, f"{args.output_name}.{args.save_model_as}")
        if fs.exists(final_model_path):
            file_obj = fs.get_object(final_model_path)
            model_bytes = file_obj.contents
            print(f"\nSuccessfully retrieved final model '{final_model_path}' ({len(model_bytes)} bytes) from memory.")

    print("\nFilesystem patcher has been deactivated. Real filesystem is back in effect.")
    with open(final_model_path, "wb") as f_out:
                f_out.write(model_bytes)

if __name__ == "__main__":
    main()
