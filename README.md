# Digit-Recognition
# SVHN Digit Classifier

This project implements a Convolutional Neural Network (CNN) to classify digits from the Street View House Numbers (SVHN) dataset using TensorFlow and Keras.

## Overview
- **Dataset**: SVHN Cropped (32x32 RGB images of digits 0-9)
- **Model**: CNN with Conv2D, BatchNormalization, MaxPooling, and Dropout layers
- **Performance**: Achieves ~96.5% validation accuracy after 50 epochs
- **Features**: Data augmentation, learning rate scheduling, model checkpointing

## Files
- `svhn.ipynb`: Loads and preprocesses the SVHN dataset
- `cnn.ipynb`: Defines, trains, and evaluates the CNN model; includes custom image prediction
- `best_model.h5`: Trained model weights (optional, if included)
- `images/`: Sample images for testing (e.g., `images.png`)

## Requirements
```bash
pip install tensorflow tensorflow_datasets numpy matplotlib pillow
```
## Usage
Run svhn.ipynb to load the SVHN dataset.
Run cnn.ipynb to train the model and test on custom images.
Update the image path in cnn.ipynb for custom predictions.
## Results
Training accuracy: ~94%
Validation accuracy: ~96.5%
Sample prediction: Correctly identifies digit 3 in images.png
## Future Improvements
Add early stopping to prevent overfitting
Experiment with deeper architectures or transfer learning
Support batch predictions for multiple images
