# Does OCT help fundus photography? Cohort-aware multimodal deep learning for diabetic macular edema

This repository contains the code of the paper submitted for double-blind review.

## What the notebook does

- **Dataset audit:** shows that, in the public OCT-AND-EYE-FUNDUS-DATASET, a classifier that uses only color histograms detects diabetic macular edema (DME) with AUC 0.985, revealing an acquisition shortcut.
- **Cohort-aware protocol:** restricts the experiments to the patients whose images with and without DME share the same origin, with 5-fold patient-level cross-validation.
- **Models:** ResNet50, DenseNet121 and ViT-B/16 (ImageNet weights, full fine-tuning), with learning rate and batch size selected by grid search.
- **Tasks:** DME detection and diabetic retinopathy grading (no DR, NPDR, PDR) on fundus photographs and OCT B-scans.
- **Fusion:** late and intermediate fusion of fundus and OCT on 410 paired eyes, with paired patient bootstrap tests.
- **Extra analyses:** robustness subgroup, preprocessing ablation and Grad-CAM maps.

## How to run

1. Open `dme_fundus_oct.ipynb` in Google Colab with a GPU runtime (Runtime > Change runtime type > GPU).
2. Run the cells in order. The notebook downloads the dataset automatically from
   https://github.com/Traslational-Visual-Health-Laboratory/OCT-AND-EYE-FUNDUS-DATASET
3. Results and figures are saved to Google Drive. Set `USAR_DRIVE = False` to save them in the Colab session instead.
4. Set `MODO_RAPIDO = True` for a quick test of the pipeline (one fold, few epochs).

The full run (grid search, 5 folds, 3 networks, 2 modalities, 2 tasks, fusion and ablation) takes several hours on a single NVIDIA T4 GPU.

Comments and section titles in the notebook are in Portuguese.

## Requirements

Python 3, PyTorch, torchvision, scikit-learn, OpenCV, pandas, matplotlib and grad-cam (installed by the notebook).
