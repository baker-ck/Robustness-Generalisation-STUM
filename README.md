# Robustness-Generalisation-STUM

This repository contains configurations checkpoints for diverse STGNN models (AGCRN, D2STGNN, GWNET, STAE, STID and STGCN) reproduced  with Spatio-Temporal Unitized Modelling (STUM). Test results are provided for comparison.

## Implementation notes
We use the publicly available STUM (https://github.com/RWLinno/STUM) codebase as our training framework. To ensure smooth execution, we applied minor engineering fixes (correcting import paths, guarding ./save paths and fixing logging typos). No changes were made to model architectures, loss functions, or optimisation procedures.


## Repo structure
```
STUM/
├── README.md
├── baseline_models/
|   └── pre_trained_agcrn_model.pt
|   └── pre_trained_d2stgnn_model.pt
|   └── pre_trained_gwnet_model.pt
|   └── pre_trained_stae_model.pt
|   └── pre_trained_stid_model.pt
|   └── pre_trained_stgcn_model.pt
├── baseline_training_notebooks/
|   └── acgrn.ipynb
|   └── d2stgnn.ipynb
|   └── gwnet.ipynb
|   └── stae.ipynb
|   └── stid.ipynb
|   └── stgcn.ipynb
├── stum_patches/
│   └── non_algorithmic_fixes.diff
└── requirements.txt
└── environment.yaml
```

## Environment
- Python: 3.10.0
- Google Colab Pro+
- Device: A1000 (NVIDIA) with High RAM (80GB)
- Criteria: best validation loss

## Results

<table>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="3">PeMS03</th>
    <th colspan="3">PeMS04</th>
    <th colspan="3">PeMS07</th>
    <th colspan="3">PeMS08</th>
    <th colspan="3">PeMS-BAY</th>
    <th colspan="3">METR-LA</th>
  </tr>

  <tr>
    <th>Average MAE</th>
    <th>Average MSE</th>
    <th>Average MAPE</th>
    <th>Average MAE</th>
    <th>Average MSE</th>
    <th>Average MAPE</th>
    <th>Average MAE</th>
    <th>Average MSE</th>
    <th>Average MAPE</th>
    <th>Average MAE</th>
    <th>Average MSE</th>
    <th>Average MAPE</th>
    <th>Average MAE</th>
    <th>Average MSE</th>
    <th>Average MAPE</th>
    <th>Average MAE</th>
    <th>Average MSE</th>
    <th>Average MAPE</th>
  </tr>

  <!-- Experiment 1 -->
  <tr>
    <td>AGCRN</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>

  <!-- Experiment 2 -->
  <tr>
    <td>D2STGNN</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>

  <!-- Experiment 3 -->
  <tr>
    <td>GWNET</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>

  <!-- Experiment 4 -->
  <tr>
    <td>STID</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>

  <!-- Experiment 5 -->
  <tr>
    <td>STAE</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>

  <!-- Experiment 6 -->
  <tr>
    <td>STGCN</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
</table>


## Reference

The implementation of STUM is based on the IEEE T-ITS 2025 paper “Cross Space and Time: A Spatio-Temporal Unitized Model for Traffic Flow Forecasting”, with the following citation:

```bibtex
@article{ruan2025cross,
  title={Cross space and time: A spatio-temporal unitized model for traffic flow forecasting},
  author={Ruan, Weilin and Wang, Wenzhuo and Zhong, Siru and Chen, Wei and Liu, Li and Liang, Yuxuan},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2025},
  publisher={IEEE}
}
```




