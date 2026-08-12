# Robustness-Generalisation-STUM

This repository contains configurations for diverse STGNN models (DCRNN, DGCRN, GMAN, MegaCRN, MTGNN,STGODE and USTCGN) integrated with Spatio-Temporal Unitized Modelling (STUM). Test results are provided for comparison.

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
    <th rowspan="2">Experiment #</th>
    <th rowspan="2">Model</th>
    <th colspan="3">PEMS03</th>
    <th colspan="3">PEMS04</th>
    <th colspan="3">PEMS07</th>
    <th colspan="3">PEMS08</th>
    <th colspan="3">PEMS-BAY</th>
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
    <td>Experiment 1</td>
    <td>Model 1</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>

  <!-- Experiment 2 -->
  <tr>
    <td>Experiment 2</td>
    <td>Model 2</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>

  <!-- Experiment 3 -->
  <tr>
    <td>Experiment 3</td>
    <td>Model 3</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>

  <!-- Experiment 4 -->
  <tr>
    <td>Experiment 4</td>
    <td>Model 4</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>

  <!-- Experiment 5 -->
  <tr>
    <td>Experiment 5</td>
    <td>Model 5</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>

  <!-- Experiment 6 -->
  <tr>
    <td>Experiment 6</td>
    <td>Model 6</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
</table>



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
    <td>3.1154</td><td>6.2067</td><td>0.0869</td>
    <td>1.6830</td><td>3.6650</td><td>0.0382</td>
  </tr>
  <tr>
    <td>STUM+STGCN<br>(frozen = true)</td>
    <td>10.1510</td><td>13.1192</td><td>0.2185</td>
    <td>5.1199</td><td>6.7907</td><td>0.0938</td>
  </tr>
    <tr>
    <td>STUM+STGCN<br>(frozen = false)</td>
    <td>6.4808</td><td>8.9324</td><td>0.1494</td>
    <td>2.8957</td><td>4.4890</td><td>0.0581</td>
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
    <td>STUM+DCRNN<br>(frozen = true)</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
    <tr>
    <td>STUM+DCRNN<br>(frozen = false)</td>
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
    <td rowspan="4">Experiment 3</td>
    <td>DGCRN</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
  <tr>
    <td>STUM+DGCRN<br>(frozen = true)</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
    <tr>
    <td>STUM+DGCRN<br>(frozen = false)</td>
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
    <td rowspan="4">Experiment 4</td>
    <td>GMAN</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
  <tr>
    <td>STUM+GMAN<br>(frozen = true)</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
    <tr>
    <td>STUM+GMAN<br>(frozen = false)</td>
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
    <td rowspan="4">Experiment 5</td>
    <td>MegaCRN</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
  <tr>
    <td>STUM+MegaCRN<br>(frozen = true)</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
    <tr>
    <td>STUM+MegaCRN<br>(frozen = false)</td>
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
    <td rowspan="4">Experiment 6</td>
    <td>MTGNN</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
  <tr>
    <td>STUM+MTGNN<br>(frozen = true)</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
    <tr>
    <td>STUM+MTGNN<br>(frozen = false)</td>
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
    <td rowspan="4">Experiment 7</td>
    <td>STGODE</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
  <tr>
    <td>STUM+STGODE<br>(frozen = true)</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
    <tr>
    <td>STUM+STGODE<br>(frozen = false)</td>
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
    <td rowspan="4">Experiment 8</td>
    <td>USTGCN</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
  <tr>
    <td>STUM+USTGCN<br>(frozen = true)</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
    <tr>
    <td>STUM+USTGCN<br>(frozen = false)</td>
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




