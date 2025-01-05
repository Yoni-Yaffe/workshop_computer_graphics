import yaml
import os
import sys
import argparse


def set_cache_directories(cache_dir):
    """
    Sets the cache directories for models and data.
    """
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers_cache")
    os.environ["HF_HOME"] = os.path.join(cache_dir, "huggingface_home")
    os.environ["DIFFUSERS_CACHE"] = os.path.join(cache_dir, "diffusers_cache")
    os.environ["TORCH_HOME"] = os.path.join(cache_dir, "torch_home")
    print("Environment variables for cache directories set:")
    print(f"TRANSFORMERS_CACHE={os.environ['TRANSFORMERS_CACHE']}")
    print(f"HF_HOME={os.environ['HF_HOME']}")
    print(f"DIFFUSERS_CACHE={os.environ['DIFFUSERS_CACHE']}")
    print(f"TORCH_HOME={os.environ['TORCH_HOME']}")

def run_textual_inversion(yaml_path, logdir):
    """
    Runs the textual inversion script with parameters from a YAML configuration file.

    Parameters:
        yaml_path (str): Path to the YAML configuration file.
    """
    
    # Load YAML configuration
    with open(yaml_path, "r") as file:
        yaml_config = yaml.safe_load(file)
    print("Started main")
    cache_dir_path = yaml_config.get("cache_dir", "/vol/scratch/jonathany2/cache")
    set_cache_directories(cache_dir_path)
    # we intentionally import here in order to set the env variables first
    from textual_inversion_decomposed import main as textual_inversion_main
    config = yaml_config["train_params"]
    print("Config:", config)
    # Extract parameters from YAML
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    train_data_dir = config.get("train_data_dir", "input_concepts")
    
    parent_data_dir = config.get("parent_data_dir")
    node = config.get("node", "v0")
    test_name = config.get("test_name", "test_run")
    max_train_steps = config.get("max_train_steps", 1000)
    validation_steps = config.get("validation_steps", 100)
    placeholder_token = config.get("placeholder_token", "<*> <&>")
    validation_prompt = config.get("validation_prompt", "<*>,<&>,<*> <&>")
    norm_loss = config.get("norm_loss", False)
    norm_loss_beta = config.get("norm_loss_beta", 0.005)
    seed = config.get("seed", 123)
    print("config:", config)
    # Construct arguments as if passed from the command line
    output_dir = os.path.join(logdir, f"outputs/{parent_data_dir}/{node}/{test_name}_seed{seed}/")
    sys.argv = [
        "textual_inversion_decomposed.py",
        "--train_data_dir", f"{train_data_dir}/{parent_data_dir}/{node}",
        "--placeholder_token", placeholder_token,
        "--validation_prompt", validation_prompt,
        "--output_dir", output_dir,
        "--seed", str(seed),
        "--max_train_steps", str(max_train_steps),
        "--validation_steps", str(validation_steps),
        "--norm_loss_beta", str(norm_loss_beta),
    ]
    if norm_loss:
        sys.argv.append("--norm_loss")
    

    print(f"Running with arguments: {' '.join(sys.argv)}")

    # Call the main function of the script
    textual_inversion_main()

# Example usage
# run_textual_inversion("config.yaml")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default=None)
    parser.add_argument('--yaml_config', default=None)
    args = parser.parse_args()
    
    logdir = args.logdir
    yaml_path = args.yaml_config
    run_textual_inversion(yaml_path, logdir)