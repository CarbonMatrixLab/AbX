# AbX: Antibody Design Using a Score-based Diffusion Model Guided by Evolutionary, Physical, and Geometric Constraints

![AbX Logo](./figures/AbX.png)

T. Zhu, M. Ren, H. Zhang. Antibody Design Using a Score-based Diffusion Model Guided by Evolutionary. ICML 2024. <br>
[Link to Paper at ICML 2024](https://openreview.net/pdf?id=1YsQI04KaN)

## Installation

If you encounter any issues with the installation or would like to report a bug, please feel free to open an issue on GitHub at https://github.com/CarbonMatrixLab/AbX/issues.

### Setting up the AbX Environment

To install AbX, it is recommended to create a Conda environment and install the necessary dependencies by following these steps:

```bash
git clone git@github.com:CarbonMatrixLab/AbX.git 
conda env create -f environment.yml
```

PyRosetta is required to relax the generated structures and compute binding energy. Please refer to the installation guide provided [here](https://www.pyrosetta.org/) for further instructions.

### Dataset Preparation

Antibody-antigen structures and associated summary files can be retrieved from the SAbDab database. The dataset and accompanying files can be downloaded from the following links: 
- [Dataset](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/archive/all/)
- [Summary Files](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all/)

Extract `all_structures.zip` into the `data` directory.

To preprocess the structure data into `.npz` format, use the `preprocess_data.py` script:

```bash
python preprocess_data.py --cpu 100 --summary_file ./data/sabdab_summary_all.tsv --data_dir ./data/mmcif --output_dir ./data/npz --data_mode mmcif
```

We recommend using the `mmCIF` format for PDB structures, as it provides comprehensive information.

### Pre-trained Models

1. Download the **AbX-DiffAb** and **AbX-RAbD** model weights [here](https://zenodo.org/records/14577013?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImFkYzk5NzZhLWNjM2ItNGYzNS1hNTQyLWMxMzQyNDIyYTY2NSIsImRhdGEiOnt9LCJyYW5kb20iOiJiMmUyNzkzNmYxNDI1MDE5YzYyNzBlNzVkMWQ3MzQwNCJ9.P1juTzvnN6HzglwgUfOnb551MODvUGx6NgwvWnrhI51W_jl8nulAW7c5mnkEsgs_O6sp03G8NiowEs88y_ohlw), and place them in the `./trained_model` directory.
2. Download the **ESM2** model weights from [here](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt) and the **contact regressor** weights from [here](https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t36_3B_UR50D-contact-regression.pt), and save these files in the `./trained_model` directory.

## Usage Instructions

### Co-Design of CDRs in DiffAb Test Dataset

To perform co-design of CDRs using the DiffAb test dataset, use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py  \
    --model ./trained_model/abx_diffab.ckpt \
    --model_features ./config/config_data_feature.json \
    --model_config ./config/config_model.json \
    --batch_size 1 \
    --num_samples 100 \
    --name_idx ./test_data/diffab_test.idx \
    --data_dir  ./data/npz \
    --output_dir ./output/DiffAb_design \
    --mode design
```

### Co-Design of CDRs in RAbD Test Dataset

For co-design using the RAbD test dataset, execute the following:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py  \
    --model ./trained_model/abx_rabd.ckpt \
    --model_features ./config/config_data_feature.json \
    --model_config ./config/config_model.json \
    --batch_size 1 \
    --num_samples 100 \
    --name_idx ./test_data/RAbD_test.idx \
    --data_dir  ./data/npz \
    --output_dir ./output/RAbD_design \
    --mode design
```

### CDR Optimization in DiffAb Test Dataset

To optimize CDRs in the DiffAb test dataset, run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py  \
    --model ./trained_model/abx_diffab.ckpt \
    --model_features ./config/config_data_feature.json \
    --model_config ./config/config_model.json \
    --batch_size 1 \
    --num_samples 100 \
    --name_idx ./test_data/diffab_test.idx \
    --data_dir  ./data/npz \
    --output_dir ./output/DiffAb_optimize \
    --mode optimize
```

Modify the `generate_area` and `optimize_steps` parameters to adjust the target regions and optimization steps.

### Generating Trajectories

To generate a trajectory during the design of CDRs in the DiffAb test dataset, use the following:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py  \
    --model ./trained_model/abx_diffab.ckpt \
    --model_features ./config/config_data_feature.json \
    --model_config ./config/config_model.json \
    --batch_size 1 \
    --num_samples 100 \
    --name_idx ./test_data/diffab_test.idx \
    --data_dir  ./data/npz \
    --output_dir ./output/DiffAb_optimize \
    --mode trajectory
```

### Design CDRs given Antibody-Antigen Complex

To generate CDRs of given antibdody-antigen complexes in the PDB format, use the following:

```bash
CUDA_VISIBLE_DEVICES=0 python design.py  \
    --model ./trained_model/abx_diffab.ckpt \
    --model_features ./config/config_data_feature.json \
    --model_config ./config/config_model.json \
    --batch_size 1 \
    --num_samples 100 \
    --pdb_file  ./test_data/6ct7_H_L_S.pdb \
    --output_dir ./output/design \
    --mode design
```

The example of input antibody-antigen complexes is `6ct7_H_L_S.pdb`, where `H` is the heavy chain id, `L` is the light chain id and `S` is the antigen chain id.




### Relaxing the Designed Proteins

To relax the designed proteins using PyRosetta, run the following command and modify the relaxation regions using the `generate_area` parameter:

```bash
CUDA_VISIBLE_DEVICES=0 python relax_pdb.py  \
    --data_dir ./output/output_dir \
    --cpus 100 \
    --generate_area cdrs
```

### Metric Calculation

To compute the RMSD, AAR, and IMP metrics, use the `eval_metric.py` script as follows:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_metric.py  \
    --data_dir ./output/output_dir \
    --cpus 100 \
    --energy
```

For calculating plausibility, you may use [AntiBERTy](https://github.com/jeffreyruffolo/AntiBERTy).


## Reference

```bash
@inproceedings{
zhu2024antibody,
title={Antibody Design Using a Score-based Diffusion Model Guided by Evolutionary, Physical and Geometric Constraints},
author={Tian Zhu and Milong Ren and Haicang Zhang},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=1YsQI04KaN}
}

```

