# run_secure_training.py
import sys
import os
import io
import argparse
import tempfile
import atexit
import shutil
import hashlib
from collections import defaultdict
from enum import Enum

# --- Enhanced Import Validation ---
try:
    import toml
except ImportError:
    print(f"Error: 'toml' not found. Run: {sys.executable} -m pip install toml", file=sys.stderr); sys.exit(1)
try:
    import gnupg
except ImportError:
    print(f"Error: 'python-gnupg' not found. Run: {sys.executable} -m pip install python-gnupg", file=sys.stderr); sys.exit(1)
try:
    from pyfakefs.fake_filesystem_unittest import Patcher
except ImportError:
    print(f"Error: 'pyfakefs' not found. Run: {sys.executable} -m pip install pyfakefs", file=sys.stderr); sys.exit(1)

# --- Training script import validation ---
try:
    from musubi_tuner import qwen_image_cache_text_encoder_outputs as cache_encoder_script
    from musubi_tuner import qwen_image_cache_latents as cache_latents_script
    from src.musubi_tuner import qwen_image_train_network as train_network_script
except ImportError as e:
    print(f"Error importing training modules: {e}", file=sys.stderr); sys.exit(1)

# (Helper classes remain unchanged)
class DynamoBackend(Enum):
    NO = "NO"; EAGER = "eager"; AOT_EAGER = "aot_eager"; INDUCTOR = "inductor"
    AOT_NVFUSER = "aot_nvfuser"; NVFUSER = "nvfuser"; OFI = "ofi"; FX2TRT = "fx2trt"
    ONNXRT = "onnxrt"; IPEX = "ipex"
def int_or_float(value):
    if '%' in value: return float(value.strip('%')) / 100.0
    try: v = float(value); return int(v) if v.is_integer() else v
    except ValueError: raise argparse.ArgumentTypeError(f"Invalid value: {value}")

# (prepare_real_directories, load_and_configure_toml_in_memory, anonymize_virtual_dataset remain unchanged)
def prepare_real_directories(args):
    if args.output_dir is None:
        temp_session_dir = tempfile.mkdtemp()
        args.output_dir = os.path.join(temp_session_dir, "outputs")
        print(f"No output directory specified. Using temporary location: {args.output_dir}")
        atexit.register(lambda: (print(f"\nCleaning up temp dir: {temp_session_dir}"), shutil.rmtree(temp_session_dir, ignore_errors=True)))
    else: print(f"Using user-specified output directory: {args.output_dir}")
    if args.cache_dir is None:
        args.cache_dir = os.path.join(args.output_dir, "caches")
        print(f"No cache directory specified. Using default location: {args.cache_dir}")
    else: print(f"Using user-specified cache directory: {args.cache_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

def load_and_configure_toml_in_memory(fs, real_toml_path, real_cache_dir, virtual_dataset_dir, virtual_config_path):
    print(f"\nLoading and configuring '{real_toml_path}' for in-memory use...")
    with open(real_toml_path, 'r') as f: config = toml.load(f)
    if 'datasets' not in config or not isinstance(config['datasets'], list): raise ValueError("Dataset TOML must contain a 'datasets' list.")
    for ds in config['datasets']: ds['cache_directory'] = real_cache_dir; ds['image_directory'] = virtual_dataset_dir
    fs.create_file(virtual_config_path, contents=toml.dumps(config))
    print(f"In-memory config created at '{virtual_config_path}'.")

def anonymize_virtual_dataset(fs, root_dir: str):
    print(f"\nAnonymizing in-memory filenames inside '{root_dir}'...")
    if not fs.exists(root_dir): print(f"Warning: Virtual directory '{root_dir}' not found. Skipping anonymization."); return
    files_by_basename = defaultdict(list)
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            basename, ext = os.path.splitext(filename); full_path = os.path.join(dirpath, filename)
            files_by_basename[basename].append((full_path, ext))
    renamed_count = 0
    for basename, files in files_by_basename.items():
        hashed_basename = hashlib.sha256(basename.encode('utf-8')).hexdigest()
        for full_path, ext in files:
            dirpath = os.path.dirname(full_path); new_path = os.path.join(dirpath, f"{hashed_basename}{ext}")
            fs.rename(full_path, new_path); renamed_count += 1
    print(f"Anonymization complete. Renamed {renamed_count} files in memory without exposing filenames.")

def run_pipeline(args):
    """Main orchestrator using GPG decryption and tarfile extraction."""
    prepare_real_directories(args)
    
    virtual_workspace = "/workspace"
    virtual_model_dir = f"{virtual_workspace}/models"
    virtual_dataset_dir = f"{virtual_workspace}/datasets/train_data" # This path must match what's inside your .tar.gz
    virtual_config_path = f"{virtual_workspace}/dataset_config.toml"
    
    # --- DECRYPTION STEP (BEFORE PATCHER) ---
    print("\nDecrypting archive into memory (this may take a moment)...")
    try:
        gpg = gnupg.GPG()
        with open(args.archive_path, 'rb') as f:
            decrypted_data = gpg.decrypt_file(f, passphrase=args.password)
        
        if not decrypted_data.ok:
            raise RuntimeError(f"GPG decryption failed: {decrypted_data.status}")
        
        # This is the raw .tar.gz content, held entirely in a bytes object
        tar_gz_bytes = decrypted_data.data
        print("Archive decrypted successfully into memory.")
    except Exception as e:
        print(f"FATAL: Failed to decrypt archive: {e}", file=sys.stderr)
        print("Check if GPG is installed, the file path is correct, and the password is right.", file=sys.stderr)
        sys.exit(1)

    # --- IN-MEMORY EXTRACTION (INSIDE PATCHER) ---
    with Patcher() as patcher:
        fs = patcher.fs
        
        print("\nMounting real paths into virtual filesystem...")
        virtual_vae_path = f"{virtual_model_dir}/vae.safetensors"
        virtual_dit_path = f"{virtual_model_dir}/dit_model.safetensors"
        virtual_text_encoder_path = f"{virtual_model_dir}/text_encoder.safetensors"
        
        fs.add_real_file(args.vae_path, target_path=virtual_vae_path)
        fs.add_real_file(args.pretrained_model_name_or_path, target_path=virtual_dit_path)
        fs.add_real_file(args.text_encoder_path, target_path=virtual_text_encoder_path)
        fs.add_real_file(args.dataset_config_path, target_path=args.dataset_config_path)
        fs.add_real_directory(args.output_dir)
        fs.add_real_directory(args.cache_dir)
        print("Mounting complete.")

        print("\nExtracting in-memory tar.gz into virtual filesystem...")
        import tarfile
        with tarfile.open(fileobj=io.BytesIO(tar_gz_bytes), mode='r:gz') as tar:
            tar.extractall(path=virtual_dataset_dir)
        print("Extraction to virtual filesystem complete.")

        load_and_configure_toml_in_memory(fs, args.dataset_config_path, args.cache_dir, virtual_dataset_dir, virtual_config_path)
        anonymize_virtual_dataset(fs, virtual_dataset_dir)
        
        # --- PIPELINE EXECUTION (REMAINS THE SAME) ---
        print("\n--- STAGE 1: Caching Text Encoder Outputs ---")
        cache_encoder_args = argparse.Namespace(
            dataset_config=virtual_config_path, text_encoder=virtual_text_encoder_path,
            batch_size=4, fp8_vl=True, device=args.device, num_workers=args.max_data_loader_n_workers,
            skip_existing=False, keep_cache=True, edit=args.edit, edit_plus=args.edit_plus,
            max_resolution=getattr(args, 'max_resolution', '1024,1024')
        )
        cache_encoder_script.main_exec(cache_encoder_args)
        print("--- STAGE 1 COMPLETE ---\n")

        print("--- STAGE 2: Caching Latents ---")
        cache_latents_args = argparse.Namespace(
            dataset_config=virtual_config_path, vae=virtual_vae_path, device=args.device, batch_size=4,
            num_workers=args.max_data_loader_n_workers, max_resolution=getattr(args, 'max_resolution', '1024,1024'),
            skip_existing=False, keep_cache=True, edit=args.edit, edit_plus=args.edit_plus,
            debug_mode=None, console_width=None, console_back=None, console_num_images=None,
            disable_cudnn_backend=False, vae_dtype=args.vae_dtype
        )
        cache_latents_script.main_exec(cache_latents_args)
        print("--- STAGE 2 COMPLETE ---\n")
        
        print("--- STAGE 3: Training Network ---")
        args.dataset_config = virtual_config_path
        args.vae = virtual_vae_path
        args.text_encoder = virtual_text_encoder_path
        args.pretrained_model_name_or_path = virtual_dit_path
        args.dit = virtual_dit_path
        train_network_script.main_exec(args)
        print("--- STAGE 3 COMPLETE ---\n")

        print(">>> Secure hybrid training pipeline finished successfully. <<<")
        print(f"\nFinal model and logs saved to: {args.output_dir}")
        print(f"Caches saved to: {args.cache_dir}")

def setup_main_parser():
    parser = argparse.ArgumentParser(description="Securely run a Qwen-Image training pipeline using GPG and an in-memory filesystem.")
    # (The comprehensive argparse setup from the previous response remains identical here,
    # with one change: --archive_path now points to the .tar.gz.gpg file)
    g_paths = parser.add_argument_group('Path Arguments')
    g_paths.add_argument("--archive_path", type=str, required=True, help="Path to the .tar.gz.gpg archive with dataset.")
    g_paths.add_argument("--dataset_config_path", type=str, required=True, help="Path to the dataset .toml config file on your real disk.")
    g_paths.add_argument("--vae_path", type=str, required=True, help="Path to the VAE model file (.safetensors).")
    g_paths.add_argument("--text_encoder_path", type=str, required=True, help="Path to the text encoder model file (.safetensors).")
    g_paths.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Path to the pretrained DiT model file (.safetensors).")
    g_paths.add_argument("--output_dir", type=str, default=None, help="Directory to save final models and logs. (Default: a temporary directory)")
    g_paths.add_argument("--cache_dir", type=str, default=None, help="Directory to save caches. (Default: a 'caches' subfolder in the output directory)")
    g_paths.add_argument("--password", type=str, required=True, help="Password for the GPG-encrypted archive.")
    # ... all other arguments from the previous version remain the same ...
    g_paths.add_argument("--network_weights", type=str, default=None, help="Path to pretrained weights for the network (LoRA, etc).")
    g_paths.add_argument("--base_weights", type=str, default=None, nargs="*", help="Network weights to merge into the model before training.")
    g_paths.add_argument("--sample_prompts", type=str, default=None, help="File with prompts for generating sample images.")
    g_paths.add_argument("--config_file", type=str, default=None, help="Load hyperparameters from a .toml file instead of command line.")
    g_train = parser.add_argument_group('Core Training Arguments')
    g_train.add_argument("--output_name", type=str, default="qwen_lora_model", help="Basename for saved model files.")
    g_train.add_argument("--max_resolution", type=str, default="1024,1024", help="Max resolution for caching.")
    g_train.add_argument("--max_train_steps", type=int, default=1600, help="Total number of training steps.")
    g_train.add_argument("--max_train_epochs", type=int, default=None, help="Training epochs (overrides max_train_steps).")
    g_train.add_argument("--max_data_loader_n_workers", type=int, default=8, help="Max num workers for DataLoader.")
    g_train.add_argument("--persistent_data_loader_workers", action="store_true", help="Persistent DataLoader workers.")
    g_train.add_argument("--seed", type=int, default=None, help="Random seed for training.")
    g_train.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients.")
    g_train.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision training.")
    g_train.add_argument("--device", type=str, default="cuda", help="Device to train on ('cuda' or 'cpu').")
    g_train.add_argument("--vae_dtype", type=str, default="bfloat16", help="Data type for VAE (e.g., float16, bfloat16).")
    g_net = parser.add_argument_group('Network Configuration')
    g_net.add_argument("--network_module", type=str, required=True, help="Network module to use (e.g., 'networks.lora').")
    g_net.add_argument("--network_dim", type=int, default=128, help="Dimension of the network (rank for LoRA).")
    g_net.add_argument("--network_alpha", type=float, default=64, help="Alpha for LoRA weight scaling.")
    g_net.add_argument("--network_dropout", type=float, default=None, help="Dropout rate for network.")
    g_net.add_argument("--network_args", type=str, default=None, nargs="*", help="Additional key=value arguments for the network.")
    g_net.add_argument("--dim_from_weights", action="store_true", help="Automatically determine dim (rank) from network_weights.")
    g_net.add_argument("--scale_weight_norms", type=float, default=None, help="Scale the weight of each key pair to help prevent overtraining.")
    g_net.add_argument("--base_weights_multiplier", type=float, default=None, nargs="*", help="Multiplier for base_weights.")
    g_opt = parser.add_argument_group('Optimizer and LR Scheduler')
    g_opt.add_argument("--optimizer_type", type=str, default="AdamW8bit", help="Optimizer to use.")
    g_opt.add_argument("--optimizer_args", type=str, default=None, nargs="*", help="Additional arguments for the optimizer.")
    g_opt.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    g_opt.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm (0 for no clipping).")
    g_opt.add_argument("--lr_scheduler", type=str, default="cosine", help="Learning rate scheduler.")
    g_opt.add_argument("--lr_warmup_steps", type=int_or_float, default=0, help="Number of warmup steps for LR scheduler.")
    g_opt.add_argument("--lr_decay_steps", type=int_or_float, default=0, help="Number of decay steps for LR scheduler.")
    g_opt.add_argument("--lr_scheduler_num_cycles", type=int, default=1, help="Number of cycles for cosine with restarts.")
    g_opt.add_argument("--lr_scheduler_power", type=float, default=1, help="Power for polynomial scheduler.")
    g_opt.add_argument("--lr_scheduler_timescale", type=int, default=None, help="Inverse sqrt timescale for inverse sqrt scheduler.")
    g_opt.add_argument("--lr_scheduler_min_lr_ratio", type=float, default=None, help="Minimum learning rate as a ratio of the initial LR.")
    g_opt.add_argument("--lr_scheduler_type", type=str, default="", help="Custom scheduler module.")
    g_opt.add_argument("--lr_scheduler_args", type=str, default=None, nargs="*", help="Additional arguments for scheduler.")
    g_qwen = parser.add_argument_group('Qwen-Image & Timestep Arguments')
    g_qwen.add_argument("--edit", action="store_true", default=True, help="Enable training for Qwen-Image-Edit.")
    g_qwen.add_argument("--edit_plus", action="store_true", help="Enable training for Qwen-Image-Edit-2509.")
    g_qwen.add_argument("--num_layers", type=int, default=None, help="Number of layers in the DiT model (default is 60).")
    g_qwen.add_argument("--timestep_sampling", choices=["sigma", "uniform", "sigmoid", "shift", "flux_shift", "qwen_shift", "logsnr", "qinglong_flux", "qinglong_qwen"], default="sigma", help="Method to sample timesteps.")
    g_qwen.add_argument("--discrete_flow_shift", type=float, default=1.0, help="Discrete flow shift for the Euler Discrete Scheduler.")
    g_qwen.add_argument("--weighting_scheme", type=str, default="none", choices=["logit_normal", "mode", "cosmap", "sigma_sqrt", "none"], help="Weighting scheme for timestep distribution.")
    g_perf = parser.add_argument_group('Performance & Optimization')
    g_perf.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    g_perf.add_argument("--gradient_checkpointing_cpu_offload", action="store_true", help="Enable CPU offloading of activation for gradient checkpointing.")
    g_perf.add_argument("--sdpa", action="store_true", help="Use sdpa for CrossAttention (PyTorch 2.0+).")
    g_perf.add_argument("--flash_attn", action="store_true", help="Use FlashAttention for CrossAttention.")
    g_perf.add_argument("--flash3", action="store_true", help="Use FlashAttention 3 for CrossAttention.")
    g_perf.add_argument("--sage_attn", action="store_true", help="Use SageAttention.")
    g_perf.add_argument("--xformers", action="store_true", help="Use xformers for CrossAttention.")
    g_perf.add_argument("--split_attn", action="store_true", help="Use split attention calculation.")
    g_perf.add_argument("--fp8_base", action="store_true", help="Use fp8 for base model.")
    g_perf.add_argument("--fp8_scaled", action="store_true", help="Use scaled fp8 for DiT model.")
    g_perf.add_argument("--dynamo_backend", type=str, default="NO", choices=[e.value for e in DynamoBackend], help="Dynamo backend type.")
    g_perf.add_argument("--dynamo_mode", type=str, default=None, choices=["default", "reduce-overhead", "max-autotune"], help="Dynamo mode.")
    g_perf.add_argument("--blocks_to_swap", type=int, default=None, help="Number of blocks to swap in the model.")
    g_perf.add_argument("--img_in_txt_in_offloading", action="store_true", help="Offload img_in and txt_in to CPU.")
    g_perf.add_argument("--fp8_vl", action="store_true", help="Use fp8 for TE model.")
    g_perf.add_argument("--num_timestep_buckets", type=int, default=5, help="Number of tm buck.")
    g_ddp = parser.add_argument_group('Distributed Training (DDP) Arguments')
    g_ddp.add_argument("--ddp_timeout", type=int, default=None, help="DDP timeout in minutes.")
    g_ddp.add_argument("--ddp_gradient_as_bucket_view", action="store_true", help="Enable gradient_as_bucket_view for DDP.")
    g_ddp.add_argument("--ddp_static_graph", action="store_true", help="Enable static_graph for DDP.")
    g_log = parser.add_argument_group('Logging, Saving & Metadata')
    g_log.add_argument("--logging_dir", type=str, default=None, help="Enable logging and output TensorBoard log to this directory.")
    g_log.add_argument("--log_with", type=str, default=None, choices=["tensorboard", "wandb", "all"], help="Logging tool to use.")
    g_log.add_argument("--log_prefix", type=str, default=None, help="Add prefix for each log directory.")
    g_log.add_argument("--wandb_api_key", type=str, default=None, help="WandB API key.")
    g_log.add_argument("--save_model_as", type=str, default="safetensors", choices=["safetensors", "ckpt", "pt"], help="Format to save the model in.")
    g_log.add_argument("--save_every_n_steps", type=int, default=500, help="Save a checkpoint every N steps.")
    g_log.add_argument("--save_every_n_epochs", type=int, default=None, help="Save a checkpoint every N epochs.")
    g_log.add_argument("--save_last_n_epochs", type=int, default=None, help="Keep only the last N checkpoints (epochs).")
    g_log.add_argument("--save_last_n_steps", type=int, default=None, help="Keep only checkpoints from the last N steps.")
    g_log.add_argument("--save_state", action="store_true", help="Save optimizer state with checkpoints.")
    g_log.add_argument("--save_state_on_train_end", action="store_true", help="Save state at the end of training.")
    g_log.add_argument("--sample_every_n_steps", type=int, default=None, help="Generate sample images every N steps.")
    g_log.add_argument("--sample_every_n_epochs", type=int, default=None, help="Generate sample images every N epochs.")
    g_log.add_argument("--no_metadata", action="store_true", help="Do not save metadata in the output model.")
    g_log.add_argument("--training_comment", type=str, default=None, help="Arbitrary comment string stored in metadata.")

    return parser

if __name__ == "__main__":
    parser = setup_main_parser()
    args = parser.parse_args()
    run_pipeline(args)
