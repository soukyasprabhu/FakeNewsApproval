import os
if not os.path.exists('ml'):
    os.makedirs('ml')
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

def train():
    # 1. Download Dataset
    print("Downloading dataset...")
    url = "https://raw.githubusercontent.com/dsrscientist/DSData/master/loan_prediction.csv"
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"Failed to download from primary URL: {e}")
        # fallback to a synthesized dataset if download fails
        df = pd.DataFrame({
            'Loan_ID': [f'LP00{i}' for i in range(1000, 1100)],
            'Gender': ['Male', 'Female'] * 50,
            'Married': ['Yes', 'No'] * 50,
            'Dependents': ['0', '1', '2', '3+'] * 25,
            'Education': ['Graduate', 'Not Graduate'] * 50,
            'Self_Employed': ['No', 'Yes'] * 50,
            'ApplicantIncome': np.random.randint(1500, 10000, 100),
            'CoapplicantIncome': np.random.randint(0, 5000, 100),
            'LoanAmount': np.random.randint(50, 300, 100),
            'Loan_Amount_Term': [360, 180] * 50,
            'Credit_History': [1.0, 0.0] * 50,
            'Property_Area': ['Urban', 'Rural', 'Semiurban'] * 33 + ['Urban'],
            'Loan_Status': ['Y', 'N'] * 50
        })

    print("Dataset loaded successfully. Shape:", df.shape)

    # 2. Preprocess Data
    # Drop Loan_ID
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)

    # Handle Missing Values
    # Numerical: Mean
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    imputer_num = SimpleImputer(strategy='mean')
    df[num_cols] = imputer_num.fit_transform(df[num_cols])

    # Categorical: Mode
    cat_cols = df.select_dtypes(include=['object']).columns
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

    # Encode Categorical Features
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # 3. Split Dataset
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train Model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # 6. Save Model and Encoders
    os.makedirs('ml', exist_ok=True)
    with open('ml/loan_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'imputer_num': imputer_num,
            'imputer_cat': imputer_cat,
            'label_encoders': label_encoders,
            'num_cols': num_cols.tolist(),
            'cat_cols': [c for c in cat_cols if c != 'Loan_Status'],
            'expected_features': X.columns.tolist()
        }, f)
    
    print("Model saved to ml/loan_model.pkl")

if __name__ == "__main__":
    train()
