# 🎓 Smart Student Performance & Placement Predictor

An intermediate-level end-to-end Data Science project built with Python, Scikit-learn and Streamlit.

## Features
- Exploratory dataset and synthetic student records
- Random Forest placement classification
- Random Forest package regression
- GridSearchCV hyperparameter tuning
- Accuracy, Precision, Recall, F1 and ROC-AUC
- MAE and R² for package estimation
- Feature importance visualization
- Personalized improvement suggestions
- Interactive Streamlit dashboard
- CSV download

## Dataset
The included `data/student_data.csv` is **synthetic data generated for educational/demo purposes**. It is not real student or hiring data.

## Run locally

```bash
pip install -r requirements.txt
python src/train_model.py
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload all project files.
3. Open Streamlit Community Cloud and connect the GitHub repository.
4. Select `app.py` as the main file.
5. Deploy.

The repository already contains the trained model files, so deployment does not need to train the model again.

## Project structure

```text
student-placement-predictor/
├── app.py
├── data/student_data.csv
├── models/
│   ├── placement_model.pkl
│   ├── package_model.pkl
│   ├── features.pkl
│   └── metrics.pkl
├── src/train_model.py
├── requirements.txt
└── README.md
```

## Disclaimer
This project is intended for learning and portfolio demonstration. ML predictions can contain bias and uncertainty and should not be used as the sole basis for real hiring or career decisions.
