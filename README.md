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
│   │   ├── pre_trained_agcrn_pems03_model.pt
│   │   ├── pre_trained_agcrn_pems04_model.pt
│   │   ├── pre_trained_agcrn_pems07_model.pt
│   │   ├── pre_trained_agcrn_pems08_model.pt
│   │   ├── pre_trained_agcrn_pemsbay_model.pt
│   │   └── pre_trained_agcrn_metrla_model.pt
│   ├── d2stgnn/
│   │   ├── pre_trained_d2stgnn_pems03_model.pt
│   │   ├── pre_trained_d2stgnn_pems04_model.pt
│   │   ├── pre_trained_d2stgnn_pems07_model.pt
│   │   ├── pre_trained_d2stgnn_model.pt
│   │   ├── pre_trained_d2stgnn_pemsbay_model.pt
│   │   └── pre_trained_d2stgnn_metrla_model.pt
│   ├── gwnet/
│   │   ├── pre_trained_gwnet_pems03_model.pt
│   │   ├── pre_trained_gwnet_pems04_model.pt
│   │   ├── pre_trained_gwnet_pems07_model.pt
│   │   ├── pre_trained_gwnet_pems08_model.pt
│   │   ├── pre_trained_gwnet_pemsbay_model.pt
│   │   └── pre_trained_gwnet_metrla_model.pt
|   ├── stae/
│   │   ├── pre_trained_stae_pems03_model.pt
│   │   ├── pre_trained_stae_pems04_model.pt
│   │   ├── pre_trained_stae_pems07_model.pt
│   │   ├── pre_trained_stae_pems08_model.pt
│   │   ├── pre_trained_stae_pemsbay_model.pt
│   │   └──pre_trained_stae_metrla_model.pt
|   ├── stid/
│   │   ├── pre_trained_stid_pems03_model.pt
│   │   ├── pre_trained_stid_pems04_model.pt
│   │   ├── pre_trained_stid_pems07_model.pt
│   │   ├── pre_trained_stid_pems08_model.pt
│   │   ├── pre_trained_stid_pemsbay_model.pt
│   │   └── pre_trained_stid_metrla_model.pt
|   ├── stcgn/
│   │   ├── pre_trained_stgcn_pems03_model.pt
│   │   ├── pre_trained_stgcn_pems04_model.pt
│   │   ├── pre_trained_stgcn_pems07_model.pt
│   │   ├── pre_trained_stgcn_pems08_model.pt
│   │   ├── pre_trained_stgcn_pemsbay_model.pt
│   │   └── pre_trained_stgcn_metrla_model.pt
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
- Device: A1000 (NVIDIA) with High RAM (80GB)
- Criteria: best validation loss

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

  <!-- PeMS04 -->
  <tr>
    <td rowspan="3">PeMS04</td>
    <td>MAE</td>
    <td>21.15</td><td>20.74</td>
    <td>TBC</td><td>22.85</td>
    <td>19.89</td><td>19.88</td>
    <td>TBC</td><td>20.59</td>
    <td>18.35</td><td>19.58</td>
    <td>20.72</td><td>20.62</td>
  </tr>
  <tr>
    <td>RMSE</td>
    <td>32.93</td><td>32.61</td>
    <td>TBC</td><td>35.23</td>
    <td>31.46</td><td>31.37</td>
    <td>TBC</td><td>32.71</td>
    <td>29.79</td><td>31.79</td>
    <td>32.24</td><td>31.98</td>
  </tr>
  <tr>
    <td>MAPE</td>
    <td>0.1593</td><td>0.1457</td>
    <td>TBC</td><td>0.1733</td>
    <td>0.1389</td><td>0.1396</td>
    <td>TBC</td><td>0.1479</td>
    <td>0.1250</td><td>0.1338</td>
    <td>0.1507</td><td>0.1527</td>
  </tr>

  <!-- PeMS07 -->
  <tr>
    <td rowspan="3">PeMS07</td>
    <td>MAE</td>
    <td>TBC</td><td>23.29</td>
    <td>TBC</td><td>21.20</td>
    <td>22.83</td><td>22.52</td>
    <td>TBC</td><td>21.97</td>
    <td>19.7</td><td>21.52</td>
    <td>24.16</td><td>24.21</td>
  </tr>
  <tr>
    <td>RMSE</td>
    <td>TBC</td><td>36.18</td>
    <td>TBC</td><td>34.09</td>
    <td>36.75</td><td>35.97</td>
    <td>TBC</td><td>34.81</td>
    <td>32.75</td><td>36.29</td>
    <td>37.21</td><td>37.38</td>
  </tr>
  <tr>
    <td>MAPE</td>
    <td>TBC</td><td>0.1007</td>
    <td>TBC</td><td>0.0918</td>
    <td>0.0951</td><td>0.0969</td>
    <td>TBC</td><td>0.0986</td>
    <td>0.0917</td><td>0.0915</td>
    <td>0.1113</td><td>0.1131</td>
  </tr>

  <!-- PeMS08 -->
  <tr>
    <td rowspan="3">PeMS08</td>
    <td>MAE</td>
    <td>17.32</td><td>15.30</td>
    <td>TBC</td><td>15.72</td>
    <td>15.08</td><td>14.92</td>
    <td>14.55</td><td>14.71</td>
    <td>14.18</td><td>15.58</td>
    <td>16.48</td><td>16.58</td>
  </tr>
  <tr>
    <td>RMSE</td>
    <td>26.00</td><td>24.51</td>
    <td>TBC</td><td>24.67</td>
    <td>24.08</td><td>23.76</td>
    <td>23.57</td><td>23.79</td>
    <td>23.30</td><td>25.89</td>
    <td>25.56</td><td>25.65</td>
  </tr>
  <tr>
    <td>MAPE</td>
    <td>0.1076</td><td>0.1029</td>
    <td>TBC</td><td>0.1146</td>
    <td>0.0974</td><td>0.0989</td>
    <td>0.0985</td><td>0.1015</td>
    <td>0.0915</td><td>0.1033</td>
    <td>0.1105</td><td>0.1127</td>
  </tr>

  <!-- PeMS-BAY -->
  <tr>
    <td rowspan="3">PeMS-BAY</td>
    <td>MAE</td>
    <td>1.77</td><td>--</td>
    <td>TBC</td><td>--</td>
    <td>1.63</td><td>--</td>
    <td>1.59</td><td>--</td>
    <td>1.59</td><td>--</td>
    <td>1.68</td><td>--</td>
  </tr>
  <tr>
    <td>RMSE</td>
    <td>3.77</td><td>--</td>
    <td>TBC</td><td>--</td>
    <td>3.58</td><td>--</td>
    <td>3.51</td><td>--</td>
    <td>3.54</td><td>--</td>
    <td>3.63</td><td>--</td>
  </tr>
  <tr>
    <td>MAPE</td>
    <td>0.0412</td><td>--</td>
    <td>TBC</td><td>--</td>
    <td>0.0368</td><td>--</td>
    <td>0.0631</td><td>--</td>
    <td>0.0352</td><td>--</td>
    <td>0.0379</td><td>--</td>
  </tr>

  <!-- METR-LA -->
  <tr>
    <td rowspan="3">METR-LA</td>
    <td>MAE</td>
    <td>3.33</td><td>--</td>
    <td>TBC</td><td>--</td>
    <td>3.08</td><td>--</td>
    <td>3.13</td><td>--</td>
    <td>3.09</td><td>--</td>
    <td>3.11</td><td>--</td>
  </tr>
  <tr>
    <td>RMSE</td>
    <td>6.51</td><td>--</td>
    <td>TBC</td><td>--</td>
    <td>6.08</td><td>--</td>
    <td>6.33</td><td>--</td>
    <td>6.40</td><td>--</td>
    <td>6.23</td><td>--</td>
  </tr>
  <tr>
    <td>MAPE</td>
    <td>0.0944</td><td>--</td>
    <td>TBC</td><td>--</td>
    <td>0.0848</td><td>--</td>
    <td>0.0906</td><td>--</td>
    <td>0.0891</td><td>--</td>
    <td>0.0866</td><td>--</td>
  </tr>
</table>

<p><strong>★</strong> This study; <strong>◆</strong> Reference.</p>



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




