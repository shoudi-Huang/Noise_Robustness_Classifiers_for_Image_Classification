# Noise Robustness Classifiers for Image Classification

## Project Overview
This project focuses on designing and implementing **noise robustness classifiers** for image classification tasks. The goal is to build classifiers that can perform well on clean test data, even when trained and validated on datasets with **class-conditional random label noise**. The project involves working with three image datasets, two of which have known label noise transition matrices, while the third requires the estimation of the transition matrix. The classifiers are evaluated based on **top-1 accuracy**, **precision**, **recall**, and **F1-score**.

## Key Features
- **Noise Robustness Classifiers**: Implement two classifiers using **Importance Reweighting** and **\(T\)-Revision** techniques to handle label noise in image classification tasks.
- **Transition Matrix Estimation**: For the dataset with an unknown transition matrix, implement an estimator to estimate the label noise transition matrix.
- **Performance Evaluation**: Evaluate classifiers using **10-fold cross-validation** and report the mean and standard deviation of performance metrics (accuracy, precision, recall, F1-score).
- **Datasets**: Work with three datasets:
  - **FashionMNIST0.5.npz**: Contains 18,000 training/validation samples and 3,000 test samples with a known transition matrix.
  - **FashionMNIST0.6.npz**: Similar to FashionMNIST0.5 but with a different transition matrix.
  - **CIFAR.npz**: Contains 30,000 training/validation samples and 3,000 test samples with an unknown transition matrix.

## Technical Details
- **Programming Language**: Python 3
- **Evaluation Metrics**: Top-1 accuracy, precision, recall, and F1-score.
- **Cross-Validation**: 10-fold cross-validation is used to ensure robust performance evaluation.
- **Transition Matrix**: For the first two datasets, the transition matrix is provided. For the third dataset, the transition matrix must be estimated.

## Project Tasks
1. **Image Classification with Known Flip Rates**:
   - Use the provided transition matrices for the first two datasets to build noise robustness classifiers.
   - Report the mean and standard deviation of the test accuracy for each classifier.

2. **Image Classification with Unknown Flip Rates**:
   - Estimate the transition matrix for the third dataset (CIFAR.npz) and use it for classification.
   - Validate the effectiveness of the transition matrix estimator using the first two datasets.
   - Report the estimated transition matrix and the mean/standard deviation of the test accuracy.

## How It Works
1. **Data Loading**: Load the datasets using NumPy and preprocess the data if necessary.
2. **Classifier Implementation**: Implement two noise robustness classifiers using **Importance Reweighting** and **\(T\)-Revision** techniques.
3. **Transition Matrix Estimation**: For the CIFAR dataset, estimate the transition matrix and use it for classification.
4. **Performance Evaluation**: Evaluate the classifiers using 10-fold cross-validation and report performance metrics.
5. **Report Submission**: Submit a detailed report and the code in a compressed folder, following the submission guidelines.

## Results and Discussion
- **Transition Matrix Estimation**: The transition matrix estimator demonstrated high accuracy, with low mean absolute error (MAE) between the estimated and actual transition matrices for the FashionMNIST datasets.
- **Classifier Performance**: The **\(T\)-Revision** classifier outperformed the **Importance Reweighting** classifier on the CIFAR dataset, achieving higher accuracy and lower variance. Both classifiers consistently outperformed the baseline across all datasets.
- **Noise Robustness**: The classifiers showed significant resilience to label noise, especially in datasets with higher noise rates.

## Acknowledgments
This project was developed as part of the **COMP5328 - Advanced Machine Learning** course, focusing on the challenge of learning with label noise. The goal is to design robust classifiers that can handle noisy labels and perform well on clean test data, demonstrating the practical application of noise robustness techniques in machine learning.

## Future Work
- Explore alternative loss correction or reweighting techniques such as **Gold Loss Correction** and **Active Bias**.
- Investigate **Meta-Learning** strategies like **Bootstrapping** or **Automatic Reweighting** for further improving noise robustness.
- Study **Robust Regularization** methods to enhance the resilience of classifiers to label noise.
