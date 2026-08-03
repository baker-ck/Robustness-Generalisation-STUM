# Robustness-Generalisation-STUM

## STUM Results

This repository contains configurations for diverse STGNN models (DCRNN, DGCRN, GMAN, MegaCRN, MTGNN,STGODE and USTCGN) integrated with Spatio-Temporal Unitized Modelling (STUM). Test results are provided for comparison.

### Implementation notes
We use the publicly available STUM (https://github.com/RWLinno/STUM) codebase as our training framework. To ensure smooth execution, we applied minor engineering fixes (guarding optional wandb logging, correcting import paths, and fixing logging typos). The STGNNs were refactored into single model.py source files, called baselines, compatible with PyTorch. Data preprocessing, training and evaluation is handled independently by STUM.  

No changes were made to model architectures, loss functions, or optimisation procedures except:
1. TensorFlow implementations were converted to PyTorch equivalents
2. Multiple dependencies were resolved via model architecture refactoring into a single source file
3. Wrappers were added to source files for STUM compatibility
4. Engines were added for custom data-preprocessing for embeddings / training behaviour not provided by STUM
5. Factory methods were created for each STGNN to configure models with correct parameters before training in STUM pipeline

### Repo structure
```
STUM/
├── README.md
├── baseline_models/
│   └── label_pre_trained_agcrn_model.pt
|   └── pre_trained_agcrn_model.pt
|   └── pred_pre_trained_agcrn_model.pt
│   └── label_pre_trained_d2stgnn_model.pt
|   └── pre_trained_d2stgnn_model.pt
|   └── pred_pre_trained_d2stgnn_model.pt
│   └── label_pre_trained_gwnet_model.pt
|   └── pre_trained_gwnet_model.pt
|   └── pred_pre_trained_gwnet_model.pt
│   └── label_pre_trained_stae_model.pt
|   └── pre_trained_stae_model.pt
|   └── pred_pre_trained_stae_model.pt
│   └── label_pre_trained_stid_model.pt
|   └── pre_trained_stid_model.pt
|   └── pred_pre_trained_stid_model.pt
│   └── label_pre_trained_stgcn_model.pt
|   └── pre_trained_stgcn_model.pt
|   └── pred_pre_trained_stgcn_model.pt
├── enhanced_models/
│   └── # coming soon
├── stum_patches/
│   └── non_algorithmic_fixes.diff
└── stum_version.txt
└── requirements.txt
└── environment.yaml
```

### Environment
- Python: 3.10.0
- Device: MPS (Apple Silicon)
- Conda environment: exported in `environment.yaml`
- Python dependencies: listed in `requirements.txt`

### Results
- Dataset: PEMS07/2017
- Horizons: 12
- Criteria: validation loss

<table>
  <tr>
    <th>Group</th>
    <th>Value</th>
  </tr>
  <tr>
    <td rowspan="3">Experiment 1</td>
    <td>Row 1</td>
  </tr>
  <tr>
    <td>Row 2</td>
  </tr>
  <tr>
    <td>Row 3</td>
  </tr>
</table>
  
| Trained baseline | Average MAE | Average RMSE | Average MAPE |
|------------------|-------------|--------------|--------------|
| stgcn            |   XX.XXXX   |   XX.XXXX    |    XX.XXXX   |
| stum+stgcn       |   XX.XXXX   |   XX.XXXX    |    XX.XXXX   |
| $\Delta$         |   XX.XXXX   |   XX.XXXX    |    XX.XXXX   |

| dcrnn            |   XX.XXXX   |   XX.XXXX    |    XX.XXXX   |
| dgcrn            |   XX.XXXX   |   XX.XXXX    |    XX.XXXX   |
| gman             |   XX.XXXX   |   XX.XXXX    |    XX.XXXX   |
| megacrn          |   XX.XXXX   |   XX.XXXX    |    XX.XXXX   |
| mtgnn            | **19.6757**          |  **32.7257**         |  0.0827         |
| stgode    | 19.6953          |32.7480          |    0.0831      |
| ustcgn    | 19.6953          |32.7480          |    0.0831      |


(coming soon)

| Enhanced baseline | Average MAE | Average RMSE | Average MAPE |
|-----------|-----------------------------|------------------------------|-------------------------------|
| stum_agcrn  |           |          |          |
| stum_d2stgnn  |           |          |          |
| stum_gwnet  |           |          |          |
| stum_stae  |           |          |          |
| stum_stgcn   |           |           |           |
| stum_stid  |           |          |          |

### Reference

 The implementation is based on the IEEE T-ITS 2025 paper “Cross Space and Time: A Spatio-Temporal Unitized Model for Traffic Flow Forecasting” (https://arxiv.org/pdf/2411.09251), with the following citation:

```bibtex
@article{ruan2025cross,
  title={Cross space and time: A spatio-temporal unitized model for traffic flow forecasting},
  author={Ruan, Weilin and Wang, Wenzhuo and Zhong, Siru and Chen, Wei and Liu, Li and Liang, Yuxuan},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2025},
  publisher={IEEE}
}
```















