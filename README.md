# 🐕 AI Programming with Python - Dog Breed Classifier

A deep learning image classification project that uses pre-trained Convolutional Neural Networks (CNNs) to identify dog breeds and distinguish between dogs and non-dogs. This project is part of Udacity's AI Programming with Python Nanodegree.

## 📋 Project Overview

This application leverages state-of-the-art CNN architectures (VGG, AlexNet, ResNet) to automatically classify pet images. The program extracts labels from image filenames, runs them through a CNN classifier, and evaluates the model's performance in:

- **Dog Detection**: Identifying whether an image contains a dog
- **Breed Classification**: Correctly identifying the specific dog breed
- **Non-Dog Recognition**: Correctly identifying non-dog images

## 🎯 Key Features

- **Multiple CNN Architectures**: Compare performance across VGG, AlexNet, and ResNet models
- **Comprehensive Statistics**: Detailed accuracy metrics and classification results
- **Batch Processing**: Process entire directories of images efficiently
- **Misclassification Reports**: Optional detailed reports of incorrect predictions
- **Command-Line Interface**: Easy-to-use CLI with customizable arguments

## 🏗️ Project Architecture
```
udacity-city-dog-show-classification-project/
├── data/
│ ├── check_images.py # Main program orchestrator
│ ├── get_input_args.py # Command-line argument parser
│ ├── get_pet_labels.py # Extract labels from filenames
│ ├── classify_images.py # Image classification using CNN
│ ├── adjust_results4_isadog.py # Dog vs non-dog classification
│ ├── calculates_results_stats.py # Statistical analysis
│ ├── print_results.py # Results visualization
│ ├── classifier.py # CNN model wrapper (VGG/AlexNet/ResNet)
│ ├── dognames.txt # Reference list of dog breeds
│ ├── pet_images/ # Sample pet images dataset
│ └── uploaded_images/ # User-provided images
├── main.py
├── pyproject.toml
└── README.md
```


## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- PyTorch
- torchvision
- Pillow (PIL)
- NumPy

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd udacity-city-dog-show-classification-project
   ```

### Install dependencies:
```bash
pip install torch torchvision pillow numpy
```
Navigate to the project directory:
```bash
cd data
```

## 💻 Usage
Basic Usage
Run the classifier with default settings (VGG model, pet_images directory):
``` python
python check_images.py
```

Advanced Usage
Customize the classification with command-line arguments:
``` python
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
```

Command-Line Arguments
| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--dir` | Path to image directory | `pet_images/` | Any valid directory path |
| `--arch` | CNN model architecture | `vgg` | `vgg`, `alexnet`, `resnet` |
| `--dogfile` | Dog breed reference file | `dognames.txt` | Any valid text file |

Example Commands
```bash
# Use ResNet architecture
python check_images.py --arch resnet

# Classify uploaded images with AlexNet
python check_images.py --dir uploaded_images/ --arch alexnet

# Full custom configuration
python check_images.py --dir uploaded_images/ --arch vgg --dogfile dognames.txt
```

## 📊 Output and Results
The program generates comprehensive statistics including:
Performance Metrics
* Total Images Processed: Count of all images analyzed
* Dog Images: Number of images containing dogs
* Non-Dog Images: Number of images without dogs
* Classification Accuracy:
    * Percentage of correctly classified dogs
    * Percentage of correctly classified non-dogs
    * Percentage of correctly identified dog breeds
Sample Output
```
*** Results Summary for CNN Model Architecture VGG ***
Number of Images    :  40
Number of Dog Images:  30
Number of Not Dog Images: 10
Percentage of Correctly Classified Dogs:  100.0
Percentage of Correctly Classified Not Dogs:  100.0
Percentage of Correctly Classified Dog Breeds:  93.3
```

Misclassification Reports

Enable detailed reports to see:

* Incorrectly classified dogs/non-dogs
* Misidentified dog breeds
* Real vs. predicted labels comparison
## 🧪 Testing
Test the classifier with sample images:
``` bash
python test_classifier.py
```

Batch test multiple models:
``` bash
# Test all models on pet images
bash run_models_batch.sh

# Test all models on uploaded images
bash run_models_batch_uploaded.sh
```

## 📈 Model Performance Comparison
Based on the project requirements, different CNN architectures excel at different tasks:

| Model | Dog Detection | Breed Classification | Speed |
|-------|---------------|---------------------|-------|
| VGG | ⭐⭐⭐ Excellent | ⭐⭐⭐ Excellent | ⭐⭐ Moderate |
| AlexNet | ⭐⭐⭐ Excellent | ⭐⭐ Good | ⭐⭐⭐ Fast |
| ResNet | ⭐⭐⭐ Excellent | ⭐⭐ Good | ⭐⭐ Moderate |

Recommendation: VGG provides the best balance of accuracy for both dog detection and breed classification.

## 🔍 How It Works
1. Label Extraction: Parses image filenames to extract ground truth pet labels
2. Image Preprocessing: Resizes, crops, and normalizes images for CNN input
3. Classification: Runs images through selected CNN architecture
4. Label Comparison: Compares predicted labels against ground truth
5. Dog Validation: Checks if predictions match known dog breeds
6. Statistical Analysis: Calculates accuracy metrics and percentages
7. Results Reporting: Displays summary and optional detailed reports
## 🎓 Learning Outcomes
This project demonstrates:

✅ Command-line argument parsing with argparse

✅ Working with pre-trained CNN models (transfer learning)

✅ Image preprocessing and augmentation with PyTorch

✅ Dictionary data structures for result management

✅ Statistical analysis and performance evaluation

✅ File I/O operations and batch processing

✅ Modular programming and function design

✅ Code documentation and best practices

## 👤 Author
Stuart Kozola

Date Created: 2026-01-02

Project: Udacity AI Programming with Python Nanodegree
## 📝 License
This project is part of Udacity's educational curriculum.

## 🙏 Acknowledgments
Udacity: For providing the project framework and learning materials

PyTorch Team: For the pre-trained CNN models

ImageNet: For the comprehensive image classification dataset


##
Note: This classifier uses ImageNet pre-trained models and is designed for educational purposes. For production use, consider fine-tuning models on domain-specific datasets.
