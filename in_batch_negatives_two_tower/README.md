# In-batch Negative Sampling Two-Tower Implementation

This directory contains an implementation of the video game recommendation system using in-batch negative sampling with a two-tower architecture.

## Approach

The in-batch negative sampling strategy:
1. Uses other items within the same batch as negative samples
2. Leverages the two-tower architecture for efficient similarity computation
3. Reduces memory usage by reusing batch items
4. Enables faster training by avoiding explicit negative sampling

## Model Architecture

### Two-Tower Design
1. User Tower:
   - User embedding layer
   - Dense layers: [128, 64]
   - Output: 64-dimensional user vector

2. Item Tower:
   - Item embedding layer
   - Brand embedding layer
   - Price feature integration
   - Description text embedding
   - Dense layers: [128, 64]
   - Output: 64-dimensional item vector

### Interaction Layer
- Element-wise multiplication of user and item vectors
- Final dense layer for prediction

## Training Details

- Batch size: 512 (important for negative sampling)
- Learning rate: 0.01
- Optimizer: Adam
- Loss function: Binary Cross Entropy
- Epochs: 50
- Device: GPU (if available)

## Advantages

1. Memory Efficiency:
   - No need to store negative samples
   - Reuses batch items as negatives

2. Training Speed:
   - Reduced data loading overhead
   - Efficient similarity computation

3. Hard Negative Mining:
   - Naturally includes challenging negative examples
   - Improves model robustness

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
- label (1 for positive interaction, 0 for negative)

## Implementation Notes

- The batch size significantly impacts the number of negative samples
- The two-tower architecture enables efficient similarity computation
- The model learns to distinguish between similar items within the same batch
