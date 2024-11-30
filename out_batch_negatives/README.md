# Out-batch Negative Sampling Implementation

This directory contains an implementation of the video game recommendation system using out-batch negative sampling.

## Approach

The out-batch negative sampling strategy:
1. Uses negative samples from outside the current batch
2. Oversamples positive interactions (20x) to handle class imbalance
3. Maintains a balanced ratio of positive to negative samples

## Model Architecture

### Input Features
- User ID (embedded)
- Item ID (embedded)
- Brand ID (embedded)
- Price (normalized)
- Product Description (text embedded)

### Neural Network Structure
```
User/Item Features -> Embedding Layers -> Concatenate -> Dense(128) -> Dense(64) -> Dense(1)
```

- Embedding dimension: 64
- Hidden layers: [128, 64]
- Activation: ReLU
- Dropout: 0.5
- Final activation: Sigmoid

## Training Details

- Batch size: 512
- Learning rate: 0.01
- Optimizer: Adam
- Loss function: Binary Cross Entropy
- Epochs: 50
- Device: GPU (if available)

## Performance Metrics

The model is evaluated using:
- Training/Validation Loss
- Training/Validation Accuracy
- AUC Score

## Files

- `general_nn.ipynb`: Jupyter notebook with detailed implementation and analysis
- `general_nn_python_script.py`: Production-ready Python script
- `README.md`: This documentation file

## Usage

```bash
# Run the Python script
python general_nn_python_script.py

# Or use the Jupyter notebook for interactive development
jupyter notebook general_nn.ipynb
```

## Data Requirements

Input data (`combined_samples.csv`) should contain:
- reviewerID
- asin (item ID)
- brand
- price
- description
- label (1 for positive interaction, 0 for negative)
