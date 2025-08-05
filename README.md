# Fake or Real: The Impostor Hunt in Texts

A text authenticity detection project using Siamese neural networks with transformer models to classify whether text pairs contain real or fake content. This project focuses on detecting AI-generated or manipulated text by comparing pairs of texts using advanced deep learning techniques.

## Project Overview

This project implements a sophisticated text classification system that can distinguish between authentic and fake text content. Using a Siamese neural network architecture with BERT transformers, the model learns to identify subtle patterns that differentiate genuine text from AI-generated or manipulated content.

## Architecture

### Core Model: SiameseTextClassifier
- **Base Model**: [omykhailiv/bert-fake-news-recognition](https://huggingface.co/omykhailiv/bert-fake-news-recognition)
- **Pooling Strategies**: CLS token, mean pooling, max pooling
- **Comparison Method**: Siamese architecture comparing two text representations
- **Feature Combination**: Concatenation, difference, and element-wise product
- **Output**: Binary classification probability (real vs. fake)

### Key Components
1. **TextTruthnessDataset**: Custom PyTorch dataset for loading text pairs
2. **SiameseTextClassifier**: Main neural network model
3. **K-fold Cross-Validation**: Robust training approach with 5-fold validation
4. **Prediction Pipeline**: Test set inference and CSV output generation

## Dataset

The dataset contains:
- **Training Data**: Text pairs in `data/train/` organized as `article_XXXX/` directories
- **Test Data**: Test articles in `data/test/` with same structure
- **Labels**: `train.csv` mapping article IDs to real text IDs (1 or 2)
- **Format**: Each article contains `file_1.txt` and `file_2.txt` for comparison

### Data Structure
```
data/
├── train/
│   ├── article_0000/
│   │   ├── file_1.txt
│   │   └── file_2.txt
│   └── ...
├── test/
│   ├── article_0000/
│   │   ├── file_1.txt
│   │   └── file_2.txt
│   └── ...
└── train.csv
```

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training the Model
```bash
python transformers/siamese_transformer_trainer.py
```

This will:
- Use 5-fold cross-validation by default
- Train a BERT-based Siamese network
- Save the best model to `transformers/models/best_kfold_model.pt`
- Generate training history plots and JSON logs

### Making Predictions
```bash
python transformers/prediction.py
```

Loads the best k-fold model and generates predictions on the test set.

### Exploratory Data Analysis
```bash
python eda.py
```

Generates comprehensive EDA plots and saves them to `eda_plots/`.

## Configuration

### Training Parameters
- **Model**: `omykhailiv/bert-fake-news-recognition` (configurable via `MODEL_NAME` constant)
- **Batch Size**: 4 (optimized for GPU memory)
- **Learning Rate**: 5e-5 with warmup and decay scheduling
- **Max Sequence Length**: 512 tokens
- **Epochs**: 15 (default)
- **Data Augmentation**: Optional SynonymAug from nlpaug

## Project Structure

```
├── data/                          # Dataset files
│   ├── train/                     # Training articles
│   ├── test/                      # Test articles
│   └── train.csv                  # Training labels
├── transformers/                  # Core model implementation
│   ├── siamese_transformer_trainer.py  # Main training script
│   ├── prediction.py              # Inference utilities
│   ├── models/                    # Saved model checkpoints
│   └── predictions/               # Generated predictions
├── eda.py                         # Exploratory data analysis
├── eda_plots/                     # EDA visualizations
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Model Outputs

- **Predictions**: Timestamped prediction files in `transformers/predictions/`
- **Checkpoints**: Model weights saved with training history and validation metrics
- **Cross-validation**: Results stored in JSON format for analysis
- **Visualizations**: Training curves and performance plots

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- NLTK
- seaborn
- wordcloud
- nlpaug (optional, for data augmentation)

## Use Cases

- Text authenticity verification
- AI-generated content detection
- Academic integrity checking
- Content moderation
- Fake news detection
- Literary analysis

## Performance

The model uses k-fold cross-validation to ensure robust performance evaluation. Training history and validation metrics are automatically saved for analysis and comparison across different runs.