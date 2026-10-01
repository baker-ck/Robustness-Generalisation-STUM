# Robustness-Generalisation-STUM

This repository contains configurations and checkpoints for diverse STGNN models (AGCRN, D2STGNN, GWNET, STAE, STID, and STGCN) reproduced using Spatio-Temporal Unitized Modelling (STUM) (Ruan et al., 2025). Test results are provided for comparison on the PeMS03 and PeMS08 datasets. Note that the reported results are for the trained baseline models only and can be compared with those reported in the STUM paper and the respective original repositories. 

Future work will focus on expanded dataset testing (PeMS04, PeMS07, PeMS-BAY, and METR-LA), as well as reproducing the STUM optimisation and integration pipeline to enhance each baseline model.

## Implementation notes
We use the publicly available STUM (https://github.com/RWLinno/STUM) codebase as our training framework. To ensure smooth execution, we applied minor engineering fixes (correcting import paths, guarding ./save paths and fixing logging typos) to the main.py executable. No changes were made to model architectures, loss functions, or optimisation procedures.

## Repo structure
```
STUM/
├── README.md
├── baseline_models/
│   ├── agcrn/
│   │   ├── pre_trained_agcrn_pems03_model.pt
│   │   └── pre_trained_agcrn_pems08_model.pt
│   ├── d2stgnn/
│   │   ├── pre_trained_d2stgnn_pems03_model.pt
│   │   └── pre_trained_d2stgnn_pems08_model.pt
│   ├── gwnet/
│   │   ├── pre_trained_gwnet_pems03_model.pt
│   │   └── pre_trained_gwnet_pems08_model.pt
|   ├── stae/
│   │   ├── pre_trained_stae_pems03_model.pt
│   │   └── pre_trained_stae_pems08_model.pt
|   ├── stid/
│   │   ├── pre_trained_stid_pems03_model.pt
│   │   └── pre_trained_stid_pems08_model.pt
|   ├── stcgn/
│   │   ├── pre_trained_stgcn_pems03_model.pt
│   │   └── pre_trained_stgcn_pems08_model.pt
├── training_notebooks/
│   ├── acgrn.ipynb
│   ├── d2stgnn.ipynb
│   ├── gwnet.ipynb
│   ├── stae.ipynb
│   ├── stid.ipynb
│   └── stgcn.ipynb
├── stum_patches/
│   └── main.py
├── requirements/
│   ├── acgrn_requirements.txt
│   ├── d2stgnn_requirements.txt
│   ├── gwnet_requirements.txt
│   ├── stae_requirements.txt
│   ├── stid_requirements.txt
│   └── stgcn_requirements.txt
└── README.md
```

## Environment
- Python: 3.10.0
- Google Colab Pro+
- Device: A1000 80GB with High RAM

## Results

<table>
  <tr>
    <th rowspan="2">Dataset</th>
    <th rowspan="2">Metric</th>
    <th colspan="2">AGCRN</th>
    <th colspan="2">D2STGNN</th>
    <th colspan="2">GWNET</th>
    <th colspan="2">STAE</th>
    <th colspan="2">STID</th>
    <th colspan="2">STGCN</th>
  </tr>
  <tr>
    <th>★</th><th>◆</th>
    <th>★</th><th>◆</th>
    <th>★</th><th>◆</th>
    <th>★</th><th>◆</th>
    <th>★</th><th>◆</th>
    <th>★</th><th>◆</th>
  </tr>

  <!-- PeMS03 -->
  <tr>
    <td rowspan="3">PeMS03</td>
    <td>MAE</td>
    <td>16.83</td><td>16.69</td>
    <td>15.42</td><td>15.76</td>
    <td>14.91</td><td>15.16</td>
    <td>15.04</td><td>15.29</td>
    <td>15.19</td><td>15.33</td>
    <td>17.20</td><td>17.27</td>
  </tr>
  <tr>
    <td>RMSE</td>
    <td>27.88</td><td>27.60</td>
    <td>26.06</td><td>26.45</td>
    <td>24.77</td><td>25.82</td>
    <td>25.66</td><td>25.87</td>
    <td>25.40</td><td>27.40</td>
    <td>29.16</td><td>28.72</td>
  </tr>
  <tr>
    <td>MAPE</td>
    <td>0.1717</td><td>0.1644</td>
    <td>0.1705</td><td>0.1489</td>
    <td>0.1494</td><td>0.1611</td>
    <td>0.1618</td><td>0.1764</td>
    <td>0.1646</td><td>0.1640</td>
    <td>0.1950</td><td>0.1774</td>
  </tr>

  <!-- PeMS08 -->
  <tr>
    <td rowspan="3">PeMS08</td>
    <td>MAE</td>
    <td>17.32</td><td>15.30</td>
    <td>15.36</td><td>15.72</td>
    <td>15.08</td><td>14.92</td>
    <td>14.55</td><td>14.71</td>
    <td>14.18</td><td>15.58</td>
    <td>16.48</td><td>16.58</td>
  </tr>
  <tr>
    <td>RMSE</td>
    <td>26.00</td><td>24.51</td>
    <td>24.37</td><td>24.67</td>
    <td>24.08</td><td>23.76</td>
    <td>23.57</td><td>23.79</td>
    <td>23.30</td><td>25.89</td>
    <td>25.56</td><td>25.65</td>
  </tr>
  <tr>
    <td>MAPE</td>
    <td>0.1076</td><td>0.1029</td>
    <td>0.1172</td><td>0.1146</td>
    <td>0.0974</td><td>0.0989</td>
    <td>0.0985</td><td>0.1015</td>
    <td>0.0915</td><td>0.1033</td>
    <td>0.1105</td><td>0.1127</td>
  </tr>
</table>

<p><strong>★</strong> This study; <strong>◆</strong> Reference.</p>



## Reference

The implementation of STUM is based on the IEEE T-ITS 2025 paper “Cross Space and Time: A Spatio-Temporal Unitized Model for Traffic Flow Forecasting”, with the following citation:
<ul>
  <p>Paper: Ruan et al. (2025), Cross Space and Time: A Spatio-Temporal Unitized Model for Traffic Flow Forecasting, IEEE Transactions on Intelligent Transportation Systems. </p>
  <p>Implementation: https://github.com/RWLinno/STUM/ </p>
</ul>

```bibtex
@article{ruan2025cross,
  title={Cross space and time: A spatio-temporal unitized model for traffic flow forecasting},
  author={Ruan, Weilin and Wang, Wenzhuo and Zhong, Siru and Chen, Wei and Liu, Li and Liang, Yuxuan},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2025},
  publisher={IEEE}
}
```




