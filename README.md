# BoltzGAT
A statistical thermodynamic-inpired GNN to learning temperature-dependent representations

# Installation
Following the following steps to install the dependencies:
```bash
conda install -y pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
                 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch_geometric
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
conda install -c rdkit rdkit==2023.09.6
pip install easydict
pip install scikit-learn
```

# Usage
## Download the processed databases
Please download the processed databases from [Google Drive](https://drive.google.com/drive/folders/1jSPmP2uwgJyAga5AHrerDuKdtsS5UL8v?usp=sharing) to `database/processed`.

## Pure property prediction
```bash
python run.py --datasets {dataset names} --model {model name}
# Example:
python run.py --datasets Cp_G Cp_L Cp_S --model BoltzGAT
```

## Mixture viscosity prediction
```bash
python run_mixture.py --dataset viscosity_mixture --model BoltzGAT --pretrained_path {path to pure viscosity model or None}
```
## Solubility prediction
```bash
python run_mixture.py --dataset BigSolDBv2 --model BoltzGAT
```