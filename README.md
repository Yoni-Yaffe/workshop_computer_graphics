# Concept Decomposition - Forked Repository

This repository is a fork of the [original concept decomposition repository](https://github.com/google/inspiration_tree/tree/main), with modifications primarily in the `textual_inversion_decomposed.py` script. The key difference in this fork is the incorporation of additional regularization losses to enhance the learning process.

## Modifications in This Fork

The primary enhancement introduced in this repository is the integration of:

- **Cosine Regularization Loss**: Ensures that the learned embeddings maintain a structured relationship with the predefined initial embeddings.
- **Norm Regularization Loss**: Controls the magnitude of the learned embeddings to prevent them from diverging too far from a desired norm.

These losses are implemented within the training loop in `textual_inversion_decomposed.py` and can be enabled or disabled via command-line arguments or YAML configuration.

## Dependencies

To install the required dependencies, use the `requirements.txt` file provided in the `concept_decomposition` directory:

```bash
pip install -r concept_decomposition/requirements.txt
```

## Usage

### Running Training

To train the model using a YAML configuration file:

```bash
python main.py --logdir path_to_logs --yaml_config path_to_config.yaml
```

### Running with SLURM

If using SLURM, you can submit the job using the `send_to_slurm.py` script:

```bash
python send_to_slurm.py --yaml_config path_to_config.yaml
```

This script:

- Reads the YAML configuration file.
- Sets up logging directories.
- Generates and submits an SLURM job with the appropriate command.
- Runs locally if specified in the configuration.

### Playing with Learned Embeddings

After training, the provided Jupyter Notebook `inspiration_tree_playground.ipynb` can be used to visualize and analyze the learned embeddings interactively.

### YAML Configuration

The YAML file should define key training parameters, such as:

```yaml
train_params:
  train_data_dir: "input_concepts"
  parent_data_dir: "data_parent"
  node: "v0"
  test_name: "test_run"
  max_train_steps: 1000
  validation_steps: 100
  placeholder_token: "<*> <&>"
  validation_prompt: "<*>,<&>,<*> <&>"
  norm_loss: true
  cosine_loss: true
  norm_loss_beta: 0.005
  cosine_loss_beta: 0.01
  norm_loss_desired_norm: 1.0
  initializer_token: "object object"
  seed: 123
```

## Output Structure

After training, results will be saved in:

```
outputs/
├── parent_data_dir/
│   ├── node/
│   │   ├── test_run_seed123/
│   │   │   ├── learned_embeds.bin
│   │   │   ├── samples/
│   │   │   │   ├── validation_images.jpg
│   │   │   ├── logs/
│   │   │   ├── checkpoints/
```

## Notes

- The regularization losses can be adjusted using the `norm_loss_beta` and `cosine_loss_beta` parameters.
- Ensure your dataset is properly formatted and available at `train_data_dir`.
- Modify hyperparameters in the YAML file to optimize training.
- Monitor training progress via logged losses and generated validation images.

For further details, refer to the comments within `textual_inversion_decomposed.py` explaining the loss integration.

