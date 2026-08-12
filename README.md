# Robustness-Generalisation-STUM

This repository contains configurations checkpoints for diverse STGNN models (AGCRN, D2STGNN, GWNET, STAE, STID and STGCN) reproduced  with Spatio-Temporal Unitized Modelling (STUM). Test results are provided for comparison.

## Implementation notes
We use the publicly available STUM (https://github.com/RWLinno/STUM) codebase as our training framework. To ensure smooth execution, we applied minor engineering fixes (correcting import paths, guarding ./save paths and fixing logging typos) to the main.py executable. No changes were made to model architectures, loss functions, or optimisation procedures.

## Repo structure
```
STUM/
├── README.md
├── baseline_models/
│   ├── agcrn/
│       ├── pre_trained_agcrn_pems03_model.pt
│       ├──  pre_trained_agcrn_pems04_model.pt
│       ├──  pre_trained_agcrn_pems07_model.pt
│       ├──  pre_trained_agcrn_pems08_model.pt
│       ├──  pre_trained_agcrn_pemsbay_model.pt
│       └── pre_trained_agcrn_metrla_model.pt
│   ├── d2stgnn/
│       ├── pre_trained_d2stgnn_pems03_model.pt
│       ├── pre_trained_d2stgnn_pems04_model.pt
│       ├── pre_trained_d2stgnn_pems07_model.pt
│       ├── pre_trained_d2stgnn_model.pt
│       ├── pre_trained_d2stgnn_pemsbay_model.pt
│       └── pre_trained_d2stgnn_metrla_model.pt
│   ├── gwnet/
|       ├── pre_trained_gwnet_pems03_model.pt
|       ├── pre_trained_gwnet_pems04_model.pt
|       └──pre_trained_gwnet_pems07_model.pt
|       └──pre_trained_gwnet_pems08_model.pt
|       └──pre_trained_gwnet_pemsbay_model.pt
|       └──pre_trained_gwnet_metrla_model.pt
|   ├── stae/
|       ├── pre_trained_stae_pems03_model.pt
|       ├── pre_trained_stae_pems04_model.pt
|       ├── pre_trained_stae_pems07_model.pt
|       └──pre_trained_stae_pems08_model.pt
|       └──pre_trained_stae_pemsbay_model.pt
|       └──pre_trained_stae_metrla_model.pt
|   ├── stid/
|       ├──pre_trained_stid_pems03_model.pt
|       └──pre_trained_stid_pems04_model.pt
|       └──pre_trained_stid_pems07_model.pt
|       └──pre_trained_stid_pems08_model.pt
|       └──pre_trained_stid_pemsbay_model.pt
|       └──pre_trained_stid_metrla_model.pt
|   ├── stcgn/
|       ├──pre_trained_stgcn_pems03_model.pt
|       └──pre_trained_stgcn_pems04_model.pt
|       └──pre_trained_stgcn_pems07_model.pt
|       └──pre_trained_stgcn_pems08_model.pt
|       ├──pre_trained_stgcn_pemsbay_model.pt
│       └──pre_trained_stgcn_metrla_model.pt
├── training_notebooks/
│   ├── acgrn.ipynb
│   ├── d2stgnn.ipynb
│   ├── gwnet.ipynb
│   ├── stae.ipynb
│   ├── stid.ipynb
│   ├── stgcn.ipynb
├── stum_patches/
│   └── main.py
├── requirements/
│   ├──  acgrn_requirements.txt
│   ├──  d2stgnn_requirements.txt
│   ├──  gwnet_requirements.txt
│   ├──  stae_requirements.txt
│   ├── stid_requirements.txt
│   └── stgcn_requirements.txt
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
    <td>17.1977</td><td>29.1553</td><td>0.1950</td>
    <td>20.7230</td><td>32.2432</td><td>0.1507</td>
    <td>24.1579</td><td>37.2138</td><td>0.1113</td>
    <td>16.4761</td><td>25.5645</td><td>0.1105</td>
    <td>1.6795</td><td>3.6302</td><td>0.0379</td>
    <td>3.1107</td><td>6.2288</td><td>0.0866</td>
  </tr>
  <tr>
    <td>AGCRN (reference)</td>
    <td>16.69</td><td>27.60</td><td>16.44%</td>
    <td>20.74</td><td>32.61</td><td>14.57%</td>
    <td>23.29</td><td>36.18</td><td>10.07%</td>
    <td>15.3</td><td>24.51</td><td>10.29%</td>
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
  <tr>
    <td>D2STGNN (reference)</td>
    <td>15.76</td><td>26.45</td><td>14.89%</td>
    <td>22.85</td><td>—</td><td>—</td>
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
  <tr>
    <td>GWNET (reference)</td>
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
  <tr>
    <td>STID (reference)</td>
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
  <tr>
    <td>STAE (reference)</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>

  <!-- Experiment 6 -->
  <tr style="background-color:#f5f7fa;">
    <td>STGCN</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
    <td>—</td><td>—</td><td>—</td>
  </tr>
  <tr style="background-color:#f5f7fa;">
    <td>STGCN (reference)</td>
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




