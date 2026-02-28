# mirs-tskin-nn

Training script for the **DNN-TB** experiment from Liu et al. (2022), which predicts ocean sea surface temperature (SST) directly from NOAA-20 ATMS brightness temperatures using a deep neural network. Developed during an R&D internship at NOAA.

> Liu, S., Grassotti, C., Liu, Q., Zhou, Y., & Lee, Y.-K. (2022). Improvement of MiRS Sea Surface Temperature Retrievals Using a Machine Learning Approach. *IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens.*, 15, 1857–1868. https://doi.org/10.1109/JSTARS.2022.3151002

---

## Background

The NOAA Microwave Integrated Retrieval System (MiRS) retrieves atmospheric and surface parameters from satellite microwave observations using a 1D-variational algorithm. Despite its all-weather capability, MiRS SST retrievals from cross-track instruments like NOAA-20 ATMS have known limitations: cold biases at high latitudes, scan angle dependence across the swath, and coastal artifacts.

The paper compared three machine learning approaches to correct these biases. This script implements **DNN-TB** — the experiment that predicts SST directly from the 22 ATMS brightness temperature channels, without relying on MiRS physical retrievals as inputs.

| Experiment | Inputs | Target |
|---|---|---|
| DNN-Retrieval | MiRS retrieved SST + emissivity (22 ch) + CLW + TPW + lat + lon + cos(angle) | SST residual |
| **DNN-TB** ← *this script* | Brightness temperatures (22 ch) + lat + lon + cos(angle) | SST |
| MLReg-TB | Same as DNN-TB | SST |

---

## Key findings (Liu et al. 2022)

| Method | Std dev — January | Std dev — July |
|---|---|---|
| MiRS operational | 3.22 K | 3.02 K |
| **DNN-TB** ← *this script* | **2.15 K** | **2.27 K** |
| DNN-Retrieval | 1.80 K | 1.92 K |
| MLReg-TB | 2.76 K | 3.01 K |

DNN-TB reduced retrieval error by ~30% over the operational baseline and eliminated scan angle dependence. DNN-Retrieval performed better overall because MiRS physical retrievals contain information beyond what raw brightness temperatures alone can provide. The paper recommends stratified monthly training for best operational performance.

---

## Model architecture

Two hidden layers of 200 ReLU nodes, trained with RMSprop (lr=0.001), MSE loss, and early stopping (patience=100, max 1000 epochs). Inputs normalized by training set mean and standard deviation.

```
Input (25)  →  Dense(200, ReLU)  →  Dense(200, ReLU)  →  Output (1)
[lon, lat, tbu1–tbu22, cos(zenith)]
```

---

## Usage

Update the paths at the top of `train_dnn_tb.py`:

```python
date     = "12days"
dir_in   = "/path/to/mirs/data/"
out_dir  = "/path/to/output/"
```

Then run:

```bash
python train_dnn_tb.py
```

---

## Required data files

| File pattern | Description |
|---|---|
| `mirs_img_*.nc` | Simulated MIRS IMG files |
| `obs_img_*.nc` | Observed MIRS IMG files (collocated ATMS) |

The paper trained on 12 days of NOAA-20 data, one per month in 2020 (~21 million ocean samples).

---

## Outputs

| File | Description |
|---|---|
| `Entire_Model_node200_<tag>.h5` | Full saved model |
| `model_node200_<tag>.json` | Model architecture |
| `model_node200_<tag>.h5` | Model weights |
| `history_model_node200_<tag>.csv` | Per-epoch training history |
| `MAE_model_node200_patience100.png` | MAE learning curve |
| `MSE_model_node200_patience100.png` | MSE learning curve |

---

## Requirements

```
tensorflow>=2.10
numpy
pandas
matplotlib
scikit-learn
netCDF4
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Notes

- Multi-GPU training via `tf.distribute.MirroredStrategy` is enabled by default; comment out the GPU block for single-GPU or CPU runs
- Only ocean pixels (`Sfc_type == 0`) with valid TSkin in both sim and obs are used
- Rows with any fill values (< −900) are dropped before training

---

## Contributors

Methodology, experimental design, and published results are from Liu et al. (2022). Copyright to this code implementation is held by the author.

---

## Acknowledgements

Special thanks to the NOAA MIRS team for data access and scientific guidance. This work was supported in part by NOAA grants NA19NES4320002 and NA19OAR4320073, and by the Joint Polar Satellite System.

---

## Citation

```bibtex
@article{liu2022mirs,
  author  = {Liu, Shuyan and Grassotti, Christopher and Liu, Quanhua and Zhou, Yan and Lee, Yong-Keun},
  title   = {Improvement of {MiRS} Sea Surface Temperature Retrievals Using a Machine Learning Approach},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year    = {2022},
  volume  = {15},
  pages   = {1857--1868},
  doi     = {10.1109/JSTARS.2022.3151002}
}
```