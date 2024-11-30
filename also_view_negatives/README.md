# Also-View Negative Sampling Implementation

This directory contains an implementation of the video game recommendation system using also-view items for negative sampling.

## Approach

The also-view negative sampling strategy:
1. Uses "also viewed" items as hard negative samples
2. Combines with random negative sampling for diversity
3. Creates challenging training examples to improve model robustness
4. Helps model learn fine-grained item differences

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

## Advantages

1. Hard Negative Mining:
   - Uses semantically similar items as negatives
   - Forces model to learn subtle differences

2. Improved Generalization:
   - More realistic negative examples
   - Better handling of similar items

3. Enhanced User Experience:
   - More diverse recommendations
   - Better distinction between similar items

## Files

- `general_nn.ipynb`: Jupyter notebook with implementation details
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
- also_view (list of related item IDs)
- label (1 for positive interaction, 0 for negative)

## Implementation Notes

- The model benefits from the challenging nature of also-view negatives
- Balancing also-view and random negatives is important
- The approach helps in learning nuanced item relationships
