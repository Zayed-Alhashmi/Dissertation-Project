# Automated CAC Scoring from Chest CT using Classical Pipeline + CNN

This project implements an automated Coronary Artery Calcium (CAC) scoring system applied to real chest CT scans from the Stanford AIMI public dataset. It computes Agatston scores using a classical image processing pipeline and lays the groundwork for a hybrid approach that combines the classical method with a CNN-based classifier to reduce false positives. The system is validated against ground truth scores across 29 patients, with metrics including MAE, RMSE, and Pearson correlation.

---

## Installation

```bash
pip install -r requirements.txt
```

> Python 3.10+ recommended.

---

## Usage

### Classical Pipeline

Run the classical Agatston scoring pipeline on a single patient folder:

```bash
python src/score_patient.py
```

A folder picker window will open. Select the folder containing the patient's `.dcm` files. The total Agatston score will be printed to the terminal.

---

### Batch Validation (Classical)

Compare the classical pipeline against ground truth scores across all patients:

```bash
python src/validate.py
```

Select the data root folder (which contains one subfolder per patient) and the `scores.csv` ground truth file. Outputs MAE, RMSE, and Pearson r.

---

### CNN Training

_Coming soon — train the patch CNN classifier on auto-labelled DICOM data._

```bash
# python src/train.py --data_dir /path/to/patient_folders
```

---

### Hybrid Pipeline

_Coming soon — classical pipeline with CNN re-scoring to reduce false positives._

```bash
# python src/score_patient_hybrid.py
```

---

### Streamlit Demo App

_Coming soon — interactive web UI for slice-by-slice visualisation and scoring._

```bash
# streamlit run src/app.py
```

---

## Slice Visualiser

Scroll through CT slices with a red overlay highlighting detected calcium:

```bash
python src/visualize.py
```

---

## Project Structure

```
ProjectAnti/
├── src/
│   ├── load_ct.py          # DICOM loading and HU conversion
│   ├── utils.py            # All image processing (ROI, filtering, detection)
│   ├── scoring.py          # Agatston weight table and slice score
│   ├── score_patient.py    # Full pipeline orchestration
│   ├── validate.py         # Batch validation against ground truth
│   ├── cli.py              # GUI folder picker (drag-and-drop + browse)
│   └── visualize.py        # Matplotlib slice viewer with calcium overlay
├── data/                   # Patient DICOM folders (not committed)
├── requirements.txt
└── README.md
```

---

## Results

| Metric | Classical Pipeline | Hybrid Pipeline (CNN) |
|---|---|---|
| MAE | TBD | TBD |
| RMSE | TBD | TBD |
| Pearson r | TBD | TBD |

---

## Dataset

Stanford AIMI Chest CT dataset. Patient DICOM files and ground truth Agatston scores (`scores.csv`) are not included in this repository due to data governance constraints.

---

## Author

Zayed Alhashmi — Individual Project, 2025/2026
