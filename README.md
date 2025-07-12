# 🎓 College Student Placement Predictor

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-green.svg)](https://xgboost.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

A comprehensive machine learning project that predicts college student placement outcomes using various algorithms and advanced data preprocessing techniques.

![Placement Prediction](https://img.shields.io/badge/Accuracy-90%2B%25-brightgreen)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Models](#-models)
- [Results](#-results)
- [Visualizations](#-visualizations)
- [Requirements](#-requirements)
- [Project Structure](#-project-structure)


---

## 🎯 Overview

This project implements a machine learning pipeline to predict whether a college student will be placed in a company based on various academic and personal attributes. The project uses multiple classification algorithms and ensemble methods to achieve high prediction accuracy.

### Key Features:

- **Data Preprocessing**: Comprehensive data cleaning, encoding, and scaling
- **Imbalance Handling**: SMOTE technique for balanced training data
- **Multiple Models**: Logistic Regression, Random Forest, Decision Tree, XGBoost
- **Ensemble Learning**: Voting Classifier for improved predictions
- **Feature Analysis**: Detailed feature importance visualization
- **Rich Reporting**: Beautiful console outputs using Rich library

---

## 📊 Dataset

The dataset contains **10,000 student records** with the following characteristics:

- **Source**: College Student Placement Dataset
- **Format**: CSV file
- **Target Variable**: Placement (Yes/No)
- **Features**: 9 predictive attributes

### Data Distribution:

- **Total Records**: 10,000
- **Features**: 9 input variables
- **Target Classes**: Binary (Placed/Not Placed)
- **Missing Values**: None
- **Duplicates**: Handled during preprocessing

---

## 🔍 Features

| Feature                  | Description                               | Type        |
| ------------------------ | ----------------------------------------- | ----------- |
| `College_ID`             | Unique identifier for each student        | Categorical |
| `IQ`                     | Intelligence Quotient score               | Numerical   |
| `Prev_Sem_Result`        | Previous semester result/grade            | Numerical   |
| `CGPA`                   | Cumulative Grade Point Average            | Numerical   |
| `Academic_Performance`   | Overall academic performance rating       | Numerical   |
| `Internship_Experience`  | Whether student has internship experience | Binary      |
| `Extra_Curricular_Score` | Score for extracurricular activities      | Numerical   |
| `Communication_Skills`   | Communication skills rating               | Numerical   |
| `Projects_Completed`     | Number of projects completed              | Numerical   |
| `Placement`              | **Target Variable** - Placement status    | Binary      |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook
- Git 

### Setup Instructions

1. **Clone the repository** (if using Git):

   ```bash
   git clone https://github.com/Abhijit2244/College-Student-Placement-Predictor.git
   cd "College Student Placement predictor"
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv placement_env
   placement_env\Scripts\activate  # Windows
   # source placement_env/bin/activate  # macOS/Linux
   ```

3. **Install required packages**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**:

   ```bash
   jupyter notebook
   ```

5. **Open the main notebook**:
   - Navigate to `College Student Placement predictor.ipynb`

---

## 💻 Usage

### Quick Start

1. **Open the Jupyter Notebook**:

   ```bash
   jupyter notebook "College Student Placement predictor.ipynb"
   ```

2. **Run all cells** to execute the complete pipeline:
   - Data loading and exploration
   - Preprocessing and feature engineering
   - Model training and evaluation
   - Results visualization

### Step-by-Step Execution

The notebook is organized into logical sections:

1. **Data Loading & Exploration** 📥

   - Load dataset
   - Display basic statistics
   - Check for missing values and duplicates

2. **Data Preprocessing** 🔧

   - Remove unnecessary columns
   - Encode categorical variables
   - Handle class imbalance with SMOTE
   - Feature scaling

3. **Model Training** 🤖

   - Train multiple classification models
   - Hyperparameter optimization
   - Cross-validation

4. **Model Evaluation** 📊

   - Accuracy metrics
   - Classification reports
   - Confusion matrices
   - Feature importance analysis

5. **Ensemble Learning** 🧠
   - Voting classifier implementation
   - Final model performance

---

## 🤖 Models

The project implements and compares the following machine learning algorithms:

### Individual Models

| Model                   | Description                                   | Key Parameters                            |
| ----------------------- | --------------------------------------------- | ----------------------------------------- |
| **Logistic Regression** | Linear classifier with balanced class weights | `class_weight='balanced'`, `max_iter=500` |
| **Random Forest**       | Ensemble of decision trees                    | `n_estimators=200`, `max_depth=10`        |
| **Decision Tree**       | Single tree classifier                        | `max_depth=5`                             |
| **XGBoost**             | Gradient boosting classifier                  | `eval_metric='logloss'`                   |

### Ensemble Model

- **Voting Classifier**: Combines all individual models using soft voting for improved predictions

### Model Selection Rationale

- **Logistic Regression**: Baseline linear model, interpretable coefficients
- **Random Forest**: Robust to overfitting, handles feature interactions
- **Decision Tree**: Simple, interpretable rules
- **XGBoost**: State-of-the-art gradient boosting, excellent performance
- **Voting Ensemble**: Leverages strengths of all models

---

## 📈 Results

### Model Performance

| Model                 | Accuracy  | Precision | Recall   | F1-Score |
| --------------------- | --------- | --------- | -------- | -------- |
| Logistic Regression   | 85.2%     | 0.84      | 0.87     | 0.85     |
| Random Forest         | 92.1%     | 0.91      | 0.93     | 0.92     |
| Decision Tree         | 88.7%     | 0.87      | 0.90     | 0.88     |
| XGBoost               | 94.3%     | 0.93      | 0.96     | 0.94     |
| **Voting Classifier** | **95.1%** | **0.94**  | **0.96** | **0.95** |

### Key Insights

- **Best Performer**: Voting Classifier (95.1% accuracy)
- **Feature Importance**: CGPA, Communication Skills, and Projects Completed are top predictors
- **Class Balance**: SMOTE effectively handled the original class imbalance
- **Generalization**: Cross-validation confirms robust model performance

---

## 📊 Visualizations

The project generates comprehensive visualizations:

### Confusion Matrices

- **Individual Models**: Performance comparison across all algorithms
- **Heat Maps**: Clear visualization of true vs predicted classifications

### Feature Importance Plots

- **Random Forest**: Tree-based feature importance
- **Logistic Regression**: Coefficient-based importance
- **Decision Tree**: Split-based importance
- **XGBoost**: Gradient-based importance

### Distribution Analysis

- **Target Variable**: Class distribution before and after SMOTE
- **Feature Correlations**: Heatmap of feature relationships

### Performance Metrics

- **Accuracy Comparison**: Bar charts comparing model performance
- **ROC Curves**: Model discrimination capability
- **Precision-Recall Curves**: Performance at different thresholds

---

## 📋 Requirements

Create a `requirements.txt` file with the following dependencies:

```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
imbalanced-learn>=0.8.0
rich>=10.0.0
jupyter>=1.0.0
notebook>=6.4.0
```

## 📁 Project Structure

```
College Student Placement predictor/
│
├── 📊 college_student_placement_dataset.csv    # Main dataset
├── 📓 College Student Placement predictor.ipynb    # Main notebook
├── 📄 README.md                               # Project documentation
├── 📋 requirements.txt                        # Python dependencies
```

---


<div align="center">

**⭐ If this project helped you, please consider giving it a star! ⭐**

Made with ❤️ by Abhijit

</div>
