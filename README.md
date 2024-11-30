# ManyObjectiveDrugDesign

GitHub repository for a paper "Integrating Transformers and Many-Objective Metaheuristics for Cancer Drug Design".

## Installation

There is a multi-part installation process, due to the use of QuickVina2-GPU for molecular docking, and Python for ADMET modelling and molecular generation models.

To begin, use:

```
git clone https://github.com/Pixelatory/ManyObjectiveDrugDesign
```

Download the files at https://brocku-my.sharepoint.com/:f:/g/personal/na16dg_brocku_ca/EpEij4XhbOtEkA8mp-EcS04Bge6ynSrlZk5BuX7x9-rYLQ?e=ZWwKd4 and put them in ManyObjectiveDrugDesign/. In addition, extract the allmolgen zip file.

### QuickVina2-GPU

1. QuickVina2-GPU can already be found in QuickVinaTwoGPU/
2. Follow QuickVina2-GPU installation instructions from https://github.com/DeltaGroupNJUPT/QuickVina2-GPU

Note: Installation can be tested by running "./QuickVina2-GPU --config input_file_example/2bm2_config.txt".
There will be many lines printed to console, with the last lines taking the form:
"Writing output ... done.
Vina-GPU total runtime = 1.014 s"

### Python Environment

environment.yml

OR

```
conda create -n [ENV_NAME] conda-forge::meeko pytorch torchvision torchaudio pytorch-cuda=12.1 conda-forge::rdkit==2023.09.6 -c pytorch -c nvidia

conda activate [ENV_NAME]

pip install selfies positional-encodings[pytorch] tensorboard scipy lightning seaborn tqdm pandas matplotlib scikit-learn openTSNE pymoo umap-learn psutil
```

### Usage

1. Train a model using `train.py`. Various arguments are at the heading of this python file and are configurable.
2. Use `generate_optimizer_boundary.py` to generate the boundaries of the search space. This script gathers the latent vectors from the entire dataset, and finds the vectors corresponding to the maximum and minimum on each latent dimension. These become the boundaries of the search. Arguments to configure which model save file to use are found in the heading.
3. Use `optimizer.py` alongside `nogui.m` inside PlatEMO 4.2 to optimize. Note that you should run `nogui.m` first before running `optimizer.py`.
