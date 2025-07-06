# Deep-Generative-Replay-Framework

This repository provides a comprehensive PyTorch-based framework for research in Online Continual Learning (OCL). It focuses on **Deep Generative Replay (DGR)**, where a generative model is trained to produce samples from previously seen tasks to alleviate catastrophic forgetting. In addition to DGR, the framework includes implementations of numerous other prominent CL algorithms, enabling robust comparison and analysis across various learning scenarios.

## Features

*   **Wide Range of CL Agents:** A modular agent-based design allows for easy comparison between different CL strategies.
*   **Diverse CL Scenarios:** Supports standard benchmarks like New Classes (NC) and New Instances/Domains (NI), including non-stationary data drifts (blur, noise, occlusion).
*   **Standard Datasets:** Includes data loaders and scenarios for CIFAR-10, CIFAR-100, Mini-ImageNet, CORE50, and OpenLORIS.
*   **Flexible Configuration:** A YAML-based configuration system allows for easy control over experiments, agents, and datasets.
*   **Advanced Generative Models:** Features a Conditional VAE for DGR and an implementation of a Conditional Neural Process Mixture (CNDPM).

## Implemented Agents

This framework provides implementations for the following continual learning agents:

-   **Generative Replay:**
    -   **Deep Generative Replay (DGR):** Utilizes a Conditional VAE to generate samples from past tasks.
    -   **Conditional Neural DPM (CNDPM):** A non-parametric Bayesian approach for CL.
-   **Rehearsal Methods:**
    -   **Experience Replay (ER):** The baseline replay method.
    -   **Averaged Gradient Episodic Memory (A-GEM):** Constrains updates using gradients from replayed samples.
    -   **Maximally Interfered Retrieval (MIR):** Intelligently selects samples from the buffer that are most likely to be forgotten.
    -   **Supervised Contrastive Replay (SCR):** Leverages contrastive learning on buffered and current data.
    -   **GDumb (Greedy Sampler and Dumb Learner):** Periodically retrains the model from scratch on the buffer.
-   **Regularization Methods:**
    -   **Elastic Weight Consolidation (EWC++):** Penalizes changes to weights important for previous tasks.
    -   **Learning without Forgetting (LwF):** Uses knowledge distillation from the previous model state.
-   **Hybrid Methods:**
    -   **iCaRL:** Combines rehearsal with a nearest-class-mean classifier.

## Repository Structure

```
.
├── agents/             # Implementations of all continual learning agents
├── config/             # YAML configuration files for experiments
├── continuum/          # Data loading and continual learning scenario management
├── experiment/         # Scripts for running experiments and hyperparameter tuning
├── models/             # Network architectures (ResNet, CVAE, CNDPM)
└── utils/              # Utility functions, buffer management, and losses
```

## Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/ujjwal895/Deep-Generative-Replay-Framework.git
    cd Deep-Generative-Replay-Framework
    ```
2.  Install the required dependencies. A virtual environment is recommended.
    ```bash
    # (Create and activate your virtual environment)
    pip install torch torchvision numpy pyyaml scikit-learn kornia psutil
    ```
3.  Download the datasets. The scripts in `continuum/dataset_scripts` will automatically download some datasets into a `./datasets` directory when first run. For others like CORE50 or Mini-ImageNet, you may need to download them manually and place them in the appropriate folder.

## Running Experiments

The framework uses a flexible configuration system to run experiments by combining multiple YAML files. The primary script for execution is `experiment/run.py`.

You can create a `main.py` wrapper script to parse command-line arguments and pass them to the experiment runner.

**Example `main.py`:**

```python
import yaml
from types import SimpleNamespace
from experiment.run import multiple_run_tune_separate
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, nargs='+', 
                        help='Paths to YAML configuration files.')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save the result pickle file.')
    cli_args = parser.parse_args()

    # Combine parameters from all specified config files
    params = {}
    tune_params = {}
    for conf_file in cli_args.config:
        with open(conf_file, 'r') as f:
            config = yaml.safe_load(f)
            # Differentiate between regular and tunable parameters
            if 'learning_rate' in config['parameters'] and isinstance(config['parameters']['learning_rate'], list):
                 tune_params.update(config['parameters'])
            else:
                 params.update(config['parameters'])

    default_params = SimpleNamespace(**params)
    
    # Run the experiment
    multiple_run_tune_separate(default_params, tune_params, cli_args.save_path)

if __name__ == '__main__':
    main()
```

**To run an experiment (e.g., ER with a 1k buffer on CIFAR-100 NC):**

```bash
python main.py --config config_CVPR/general.yml \
                        config_CVPR/data/cifar100/cifar100_nc.yml \
                        config_CVPR/agent/er/er_1k.yml
```

-   `config_CVPR/general.yml`: Defines general experiment parameters like number of runs and optimizer settings.
-   `config_CVPR/data/cifar100/cifar100_nc.yml`: Specifies the dataset (CIFAR-100) and the continual learning scenario (New Classes).
-   `config_CVPR/agent/er/er_1k.yml`: Configures the agent (Experience Replay) and its specific hyperparameters (e.g., memory size).

Results, including accuracy matrices and timings, are saved as `.pkl` files in the `result/` directory, organized by dataset.
