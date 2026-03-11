# EcoScan Desktop - Waste Classification System

Intelligent waste classification application using Transfer Learning and computer vision.

![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)

## Overview

EcoScan Desktop is a web application that automatically classifies waste into two categories:
- Plastic (PET, HDPE, etc.)
- Paper/Cardboard

It uses Transfer Learning with the MobileNetV2 architecture pre-trained on ImageNet, achieving 98.48% accuracy even with a small dataset.

## Features

- Real-time classification: Analyzes images in <50ms
- Intuitive web interface: Built with Streamlit
- Transfer Learning: Reuses ImageNet features
- High accuracy: 98.48% on training data
- Interactive mode: Upload images or use the test dataset

## Model Architecture

### Phase 1: Feature Extraction
- Epochs: 5
- Learning Rate: 1e-3
- Frozen Layers: Yes (MobileNetV2 base frozen)
- Objective: Learn waste-specific features

### Phase 2: Fine-Tuning
- Epochs: 3
- Learning Rate: 1e-5
- Last 20 unfrozen layers: Yes
- Objective: Precisely adjust to our dataset

### Layer Architecture
```
Input (224x224x3)
    |
MobileNetV2 (2.25M frozen parameters)
    |
GlobalAveragePooling2D
    |
Dense(128, relu) + Dropout(0.5)
    |
Dense(1, sigmoid)
    |
Output (Plastic: 0, Paper: 1)
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 98.48% |
| Loss | 0.0871 |
| Inference Time | <50ms |
| Model Size | 9.24 MB |
| Trainable Parameters | 164,097 |

## Installation

### Requirements
- Python 3.8+
- pip or conda

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/YourUsername/ecoscan-desktop.git
cd ecoscan-desktop

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the pre-trained model
# (See "Pre-trained Model" section below)
```

## Usage

### Run the web application
```bash
streamlit run app_residuos.py
```
Then open: http://localhost:8501

### Train the model from scratch
```bash
python clasificador_residuos.py
```

## Project Structure

```
ecoscan-desktop/
├── app_residuos.py              # Main application (Streamlit)
├── clasificador_residuos.py     # Training script
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── .gitignore                   # Ignored files
├── entrenamiento_residuos.png   # Training graphs
└── dataset_reciclaje/          # [Not included - too large]
    ├── entrenamiento/
    │   ├── plastico/
    │   └── papel/
    └── validacion/
        ├── plastico/
        └── papel/
```

## Pre-trained Model

The trained model (modelo_clasificador_residuos.h5) is NOT included in the repository due to its size (9.24 MB).

To obtain it:

1. Option A: Train locally
   ```bash
   python clasificador_residuos.py
   ```

2. Option B: Download from releases
   - Go to Releases
   - Download modelo_clasificador_residuos.h5
   - Place it in the project root

## Dataset

The dataset includes 66 training images:
- 33 plastic images
- 33 paper/cardboard images

Expected structure:
```
dataset_reciclaje/
├── entrenamiento/
│   ├── plastico/ (30 images)
│   └── papel/    (30 images)
└── validacion/
    ├── plastico/ (5 images)
    └── papel/    (5 images)
```

## Training Graphs

Performance curves are automatically generated:
- entrenamiento_residuos.png - Learning curves

## Technologies

- TensorFlow/Keras: Deep learning framework
- MobileNetV2: Pre-trained architecture
- Streamlit: Interactive web framework
- NumPy: Numerical computing
- Pandas: Data analysis
- PIL: Image processing
- Matplotlib: Visualization

## How It Works

### 1. Input
- User uploads a JPG/PNG image

### 2. Preprocessing
- Resizes to 224x224 pixels
- Normalizes values to [0, 1]
- Expands batch dimension

### 3. Inference
- Passes through MobileNetV2 (extracts features)
- Passes through custom dense layers
- Obtains probability for each class

### 4. Classification
```
If output < 0.5 then PLASTIC
If output >= 0.5 then PAPER
```

### 5. Output
- Shows predicted class
- Shows confidence (%)
- Optional technical details

## Learning Concepts

- Transfer Learning
- Feature Extraction vs Fine-Tuning
- Convolutional Neural Networks (CNN)
- Data Augmentation
- Batch Normalization
- Dropout for regularization
- Early Stopping
- Model Evaluation

## License

This project is under the MIT license. See LICENSE for details.

## Contributing

Contributions are welcome. For major changes:

1. Fork the repository
2. Create a branch for your feature (git checkout -b feature/MyFeature)
3. Commit your changes (git commit -m 'Add MyFeature')
4. Push to the branch (git push origin feature/MyFeature)
5. Open a Pull Request

## Contact

- Author: Israel Rodríguez González
- Email: israelrodgonz@gmail.com
- GitHub: @isra19dev

## Acknowledgments

- TensorFlow & Keras team
- Streamlit for the excellent library
- Waste dataset collected locally

Made with and by [Your Name]

Last updated: March 11, 2026
