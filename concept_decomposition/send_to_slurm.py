import os
import yaml
from datetime import datetime
import argparse

DEFAULT_LOG_DIR = '/vol/scratch/jonathany2/runs'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', default='config.yaml')
    args = parser.parse_args()
    with open(args.yaml_config, 'r') as fp:
        config = yaml.safe_load(fp)
    slurm_config = config['slurm_params']
    sbatch_command = 'sbatch'
    if 'logdir' in config:
        logdir_base = config['logdir']
    else:
        logdir_base = DEFAULT_LOG_DIR
    run_name = f"{datetime.now().strftime('%y%m%d-%H%M%S')}_{config['run_name']}"
    logdir = os.path.join(logdir_base, run_name) # ckpts and midi will be saved here
    # logdir = f"/vol/scratch/jonathany/runs/{datetime.now().strftime('%y%m%d-%H%M%S')}_{config['run_name']}" # ckpts and midi will be saved here
    config['logdir'] = logdir
    slurm_config['output'] = os.path.join(logdir, slurm_config['output'])
    slurm_config['error'] = os.path.join(logdir, slurm_config['error'])
    os.makedirs(logdir, exist_ok=True)
    new_yaml_path = os.path.join(logdir, 'run_config.yaml')
    with open(new_yaml_path, 'w') as fp:
        yaml.dump(config, fp)
    
    for param in slurm_config:
        sbatch_command += f' --{param}="{slurm_config[param]}"'
    sbatch_command += f" {config['command']} --logdir {logdir} --yaml_config {new_yaml_path}"
    local_command = f"python3 main.py --logdir {logdir} --yaml_config {new_yaml_path}"
    if 'local' in config and config['local']:
        os.system(local_command)
    else:
        os.system(sbatch_command)
    