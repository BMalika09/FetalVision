# FetalVision

**Can a fetal ultrasound classifier be accurate for the wrong reasons?**

FetalVision is a computer vision project built around fetal ultrasound plane classification.

The project starts with a six-class classification task using two pretrained architectures. Beyond overall performance, the first version focuses on three questions:

- Where does the model fail?
- How confident is it when it is wrong?
- Is there evidence that predictions depend on non-anatomical visual cues?

To investigate these questions, the project combines class-level evaluation, prediction-level error analysis, Grad-CAM and controlled masking experiments.

The goal is not to build the largest possible benchmark, but to understand a small number of model behaviours properly.

## Current scope

The first version of FetalVision includes:

- patient-level train / validation / test separation;
- ResNet18 baseline;
- ConvNeXt-Tiny comparison;
- complete class-level evaluation;
- prediction-level error analysis;
- high-confidence failure analysis;
- confidence and entropy analysis;
- Grad-CAM;
- targeted shortcut-learning experiments;
- experiment tracking with Weights & Biases;
- an interactive Streamlit application.

## Future work

Possible extensions include:

- probability calibration;
- selective classification;
- robustness experiments;
- embedding and outlier analysis;
- acquisition-device subgroup analysis;
- fine-grained fetal brain classification.

These are intentionally kept outside the initial scope and will be added progressively where they provide useful additional evidence.