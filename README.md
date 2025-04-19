# Cross-Validation in AI: A Comprehensive Guide with Python Examples

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository provides a comprehensive guide and practical Python implementations for various Cross-Validation (CV) techniques used in Artificial Intelligence (AI) and Machine Learning (ML). It places a special emphasis on the critical role and correct application of CV in the context of **medical data analysis**, where robustness, reliability, and avoiding information leakage are paramount.

The goal is to serve as an educational resource and practical reference for both **beginners** learning about model evaluation and **advanced practitioners** looking to ensure rigorous validation, particularly in high-stakes domains like healthcare.

We cover the fundamental "why" behind CV (overfitting, generalization), explore different CV strategies, and provide runnable code examples using `scikit-learn`. Special attention is given to **patient-wise (group-based) cross-validation** and the distinction between using CV for **hyperparameter tuning** versus **final model evaluation**, including the gold-standard **Nested CV** approach.

## Key Concepts Covered

*   The problem of Overfitting and the need for Generalization estimates.
*   The fundamental principles of Cross-Validation.
*   The amplified importance of CV in Medical AI (patient safety, data scarcity, heterogeneity).
*   **Information Leakage**: What it is and how to prevent it.
*   **Patient-Wise (Group) Cross-Validation**: Correctly validating models when multiple samples exist per patient/subject.
*   **Pipelines**: Ensuring preprocessing steps are correctly handled within CV folds.
*   **Common CV Techniques**:
    *   Hold-Out (Train/Test Split)
    *   K-Fold
    *   Stratified K-Fold (for imbalanced data)
    *   Leave-One-Out (LOOCV)
    *   Group K-Fold, Leave-One-Group-Out (LOGO) - for Patient-Wise splits
    *   Time Series Split
*   **CV Use Cases**:
    *   Hyperparameter Tuning (Model Selection)
    *   Final Model Performance Estimation (Model Evaluation)
*   **Nested Cross-Validation**: Robustly performing hyperparameter tuning AND estimating generalization performance with minimal bias.

## Repository Structure

*   `README.md`: This file.
*   `LICENSE`: MIT License file.
*   `.gitignore`: Standard Python gitignore file.
*   `requirements.txt`: List of Python packages required to run the notebooks.
*   `notebooks/`: Contains Jupyter notebooks demonstrating the concepts:
    *   `00_Introduction_and_Setup.ipynb`: Basic concepts (overfitting, generalization), data generation setup, and the initial Train/Test split.
    *   `01_Basic_CV_Techniques.ipynb`: Implements and explains Hold-Out, K-Fold, Stratified K-Fold, and Leave-One-Out CV.
    *   `02_Group_CV_Patient_Wise.ipynb`: Focuses entirely on `GroupKFold` and `LeaveOneGroupOut`, demonstrating why it's essential for medical data with patient IDs.
    *   `03_TimeSeries_CV.ipynb`: Shows how to use `TimeSeriesSplit` for temporally ordered data.
    *   `04_Pipelines_and_Leakage.ipynb`: Illustrates data leakage from preprocessing and how `scikit-learn` Pipelines prevent it within CV loops.
    *   `05_CV_for_Tuning_vs_Evaluation.ipynb`: Explains the difference and shows how `GridSearchCV` uses internal CV for tuning. Highlights the bias if this score is reported as final performance.
    *   `06_Nested_CV.ipynb`: Implements Nested Cross-Validation for less biased hyperparameter tuning and performance estimation.

## How to Use

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/<YourUsername>/<YourRepoName>.git
    cd <YourRepoName>
    ```

2.  **Set up a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
    Navigate through the `notebooks/` directory in the Jupyter interface that opens in your browser. Run the cells in each notebook sequentially to see the demonstrations.

## Citation

If you use the code or concepts from this repository in your research or article, please cite it as follows (example format - consider using Zenodo for a DOI):

```bibtex
@misc{Delfan_CV_Repo_2025,
  author = {Niloufar Delfan},
  title = {Cross-Validation in AI: A Comprehensive Guide with Python Examples},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/niloufardelfan/Cross-Validation-in-AI-for-Medical-Data}}
}
