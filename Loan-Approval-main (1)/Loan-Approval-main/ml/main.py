
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import os

app = FastAPI(title="Loan Approval Prediction API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model and Encoders
model_path = os.path.join(os.path.dirname(__file__), '../ml/loan_model.pkl')
try:
    with open(model_path, 'rb') as f:
        artifacts = pickle.load(f)
        model = artifacts['model']
        imputer_num = artifacts['imputer_num']
        imputer_cat = artifacts['imputer_cat']
        label_encoders = artifacts['label_encoders']
        num_cols = artifacts['num_cols']
        cat_cols = artifacts['cat_cols']
        expected_features = artifacts['expected_features']
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


class LoanApplication(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str
    CibilScore: int


@app.post("/predict")
def predict_loan(application: LoanApplication):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Convert input to DataFrame
    input_data = {
        'Gender': [application.Gender],
        'Married': [application.Married],
        'Dependents': [application.Dependents],
        'Education': [application.Education],
        'Self_Employed': [application.Self_Employed],
        'ApplicantIncome': [application.ApplicantIncome],
        'CoapplicantIncome': [application.CoapplicantIncome],
        'LoanAmount': [application.LoanAmount],
        'Loan_Amount_Term': [application.Loan_Amount_Term],
        'Credit_History': [application.Credit_History],
        'Property_Area': [application.Property_Area]
    }
    df = pd.DataFrame(input_data)

    # Impute missing values (just in case, though pydantic should enforce presence)
    # Using the imputers from training
    try:
        # Pydantic already enforces that all fields are present, so there shouldn't be NaNs,
        # but we must ensure column order matches expected features.

        # Encode categorical variables
        for col in cat_cols:
            if col in label_encoders:
                # Handle unseen labels by assigning a default (e.g., 0) or using try-except
                le = label_encoders[col]
                # If a value wasn't in training data, this will fail, so we fallback safely
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])

        # Ensure correct order
        df = df[expected_features]

        # Predict
        prediction = model.predict(df)
        approved = bool(prediction[0] == 1 or str(prediction[0]).upper() == 'Y')

        # Override with CIBIL score logic
        if application.CibilScore >= 750 and not approved:
            approved = True
            message = "Congratulations! Although some criteria were not met, your excellent CIBIL score has qualified you for loan approval. Our team will be in touch."
        elif application.CibilScore < 600 and approved:
            approved = False
            message = "We're sorry, but despite your other strong qualifications, your CIBIL score is too low for us to approve the loan at this time. Please work on building your credit."
        elif approved:
            message = "Congratulations! Your loan application has been conditionally approved. Our team will be in touch with the next steps."
        else:
            message = "We're sorry, but we cannot approve your loan at this time based on the provided details. Don't be discouraged! You can improve your chances by building your credit history, reducing existing debts, and applying again in the future. We're rooting for your financial success!"

        return {"approved": approved, "message": message}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    if _name_== "__main__":
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000)

    # for the backend to run open the browser amd run http://127.0.0.1:8000/docs