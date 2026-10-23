# Experiment 6 - End-to-End ML Pipeline

## Project Description
An end-to-end machine learning pipeline for heart disease
classification using Python and Scikit-learn.

## Features
- Automated dataset loading
- Data validation and cleaning
- Feature engineering
- Data preprocessing
- Random Forest model training
- Model evaluation
- Model saving and loading
- Prediction generation
- Reproducibility verification

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib

## Installation
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Execution
```powershell
python download_data.py
python main.py
python run_prediction.py
python visualize.py
```

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Project Structure
- data/ - Dataset
- src/ - ML pipeline modules
- config/ - Configuration
- models/ - Saved model
- outputs/ - Evaluation and predictions
- main.py - Main pipeline
