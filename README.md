# Robustness-Generalisation-STUM

This repository contains configurations for diverse STGNN models (DCRNN, DGCRN, GMAN, MegaCRN, MTGNN,STGODE and USTCGN) integrated with Spatio-Temporal Unitized Modelling (STUM). Test results are provided for comparison.

## Implementation notes
We use the publicly available STUM (https://github.com/RWLinno/STUM) codebase as our training framework. To ensure smooth execution, we applied minor engineering fixes (guarding optional wandb logging, correcting import paths, and fixing logging typos). Data preprocessing, training and evaluation is handled independently by STUM. 

No changes were made to model architectures, loss functions, or optimisation procedures except:
1. TensorFlow implementations were converted to PyTorch equivalents
2. Dimensions of tensors were reshaped for compatibility with STUM pipeline for METR-LA and PeMS-BAY data loader 
3. Multiple dependencies were resolved via model architecture refactoring into a single source file
4. Custom wrappers (classes) were added to source files for STUM compatibility and inheritance, comprising
   - a constructor, and
   - a forward(x, label) signature 
5. Custom engines (classes) were added to STUM pipeline for embeddings / necessary training behaviour not provided by STUM
6. Factory methods added to source files for model instantiation with correct parameters before training in STUM pipeline, comprising an
   - a get_engine_and_model(args) signature

## Repo structure
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

## Environment
- Python: 3.10.0
- Google Colab Pro+
- Device: A1000 (Nvidia)
- Conda environment: exported in `environment.yaml`
- Python dependencies: listed in `requirements.txt`

## Results
- Horizons: 12
- Batch size: 64
- Input dim: 3
- Output dim: 1

Criteria: validation loss

<table>
  <tr>
    <th rowspan="2"></th>
    <th rowspan="2">Model</th>
    <th colspan="3">METR-LA</th>
    <th colspan="3">PeMS-BAY</th>
  </tr>
  <tr>
    <th>Average MAE</th>
    <th>Average RMSE</th>
    <th>Average MAPE</th>
    <th>Average MAE</th>
    <th>Average RMSE</th>
    <th>Average MAPE</th>
  </tr>

  <!-- Experiment 1 -->
  <tr>
    <td rowspan="4">Experiment 1</td>
    <td>STGCN</td>
    <td>3.1107</td><td>6.2288</td><td>0.0866</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
  <tr>
    <td>STUM+STGCN (frozen = true)</td>
    <td>14.7943</td><td>18.6444</td><td>0.3637</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
    <tr>
    <td>STUM+STGCN (frozen = false)</td>
    <td>14.7943</td><td>18.6444</td><td>0.3637</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
  <tr>
    <td>$\Delta$</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>

  <!-- Experiment 2 -->
  <tr>
    <td rowspan="4">Experiment 2</td>
    <td>DCRNN</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
  <tr>
    <td>STUM+DCRNN (frozen = true)</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
    <tr>
    <td>STUM+DCRNN (frozen = false)</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
  <tr>
    <td>$\Delta$</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>

  <!-- Experiment 3 -->
  <tr>
    <td rowspan="3">Experiment 3</td>
    <td>DGCRN</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
  <tr>
    <td>STUM+DGCRN</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
  <tr>
    <td>$\Delta$</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>

  <!-- Experiment 4 -->
  <tr>
    <td rowspan="3">Experiment 4</td>
    <td>GMAN</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
  <tr>
    <td>STUM+GMAN</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
  <tr>
    <td>$\Delta$</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>

  <!-- Experiment 5 -->
  <tr>
    <td rowspan="3">Experiment 5</td>
    <td>MegaCRN</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
  <tr>
    <td>STUM+MegaCRN</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
  <tr>
    <td>$\Delta$</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>

  <!-- Experiment 6 -->
  <tr>
    <td rowspan="3">Experiment 6</td>
    <td>MTGNN</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
  <tr>
    <td>STUM+MTGNN</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
  <tr>
    <td>$\Delta$</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>

  <!-- Experiment 7 -->
  <tr>
    <td rowspan="3">Experiment 7</td>
    <td>STGODE</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
  <tr>
    <td>STUM+STGODE</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
  <tr>
    <td>$\Delta$</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>

  <!-- Experiment 8 -->
  <tr>
    <td rowspan="3">Experiment 8</td>
    <td>USTGCN</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
  <tr>
    <td>STUM+USTGCN</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
  <tr>
    <td>$\Delta$</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
</table>

## Reference

The implementation of STUM is based on the IEEE T-ITS 2025 paper “Cross Space and Time: A Spatio-Temporal Unitized Model for Traffic Flow Forecasting” (https://arxiv.org/pdf/2411.09251), with the following citation:

```bibtex
@article{ruan2025cross,
  title={Cross space and time: A spatio-temporal unitized model for traffic flow forecasting},
  author={Ruan, Weilin and Wang, Wenzhuo and Zhong, Siru and Chen, Wei and Liu, Li and Liang, Yuxuan},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2025},
  publisher={IEEE}
}
```




