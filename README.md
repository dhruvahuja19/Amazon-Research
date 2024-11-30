# Amazon Video Game Recommendation System

A deep learning-based recommendation system for video games using PyTorch, implementing different negative sampling strategies for improved model performance.

## Project Overview

This repository contains implementations of a two-tower neural network architecture for video game recommendations, with three different approaches to negative sampling:

1. **Out-batch Negatives** (`/out_batch_negatives/`)
   - Traditional approach using negative samples from outside the current batch
   - Balanced sampling with oversampling of positive interactions

2. **In-batch Negatives** (`/in_batch_negatives_two_tower/`)
   - Efficient approach using other items within the same batch as negative samples
   - Two-tower architecture optimized for in-batch negative sampling

3. **Also-View Negatives** (`/also_view_negatives/`)
   - Sophisticated approach using "also viewed" items as hard negative samples
   - Leverages product relationships for more challenging training

## Model Architecture

The core model is a two-tower neural network with:

- User Tower:
  - User embedding layer
  - Multiple dense layers with ReLU activation
  - Dropout for regularization

- Item Tower:
  - Item embedding layer
  - Brand embedding layer
  - Price feature integration
  - Description text processing using embedding layer
  - Multiple dense layers with ReLU activation
  - Dropout for regularization

## Features

- Multi-modal feature processing (numerical, categorical, text)
- GPU acceleration support
- Flexible embedding dimensions
- Comprehensive evaluation metrics (AUC, accuracy)
- Modular code structure
- Both Jupyter notebook and Python script implementations

## Requirements

- Python 3.8+
- PyTorch
- torchtext
- pandas
- numpy
- scikit-learn
- tqdm

## Dataset

The model is designed to work with Amazon video game review data, expecting the following features:
- User ID
- Item ID
- Brand ID
- Price
- Product Description
- Interaction Label (1 for positive interaction, 0 for negative)

## Usage

Each implementation directory contains:
1. A Jupyter notebook (`general_nn.ipynb`) for interactive development
2. A Python script (`general_nn_python_script.py`) for production use

To run any implementation:
```bash
cd [implementation_directory]
python general_nn_python_script.py
```

## Model Performance

The model achieves competitive performance metrics:
- AUC Score: ~0.85-0.90
- Training Accuracy: ~80-85%
- Validation Accuracy: ~75-80%

## Future Work

- Implement early stopping
- Add more sophisticated text processing
- Explore additional negative sampling strategies
- Add model interpretation tools
- Create inference pipeline for production deployment

## License

This project is licensed under the MIT License - see the LICENSE file for details.