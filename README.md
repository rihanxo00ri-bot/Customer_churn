# Customer Churn Prediction API

## Model Performance

The project uses a **Tuned Random Forest Classifier** with a custom classification threshold of `0.38`.

The model was evaluated on a test dataset containing 1,409 customer records.

### Evaluation Metrics

| Metric | Score |
|---|---:|
| Accuracy | 78.85% |
| Precision (Churn) | 59% |
| Recall (Churn) | 67% |
| F1-score (Churn) | 63% |
| Macro F1-score | 74% |
| Weighted F1-score | 79% |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No Churn (0) | 0.87 | 0.83 | 0.85 | 1,035 |
| Churn (1) | 0.59 | 0.67 | 0.63 | 374 |

### Why a 0.38 Threshold?

The default classification threshold is commonly `0.50`. This project uses `0.38` to make the model more sensitive to potential churn customers.

A lower threshold can increase recall, helping identify more customers who may leave, although it can also increase false-positive predictions.

The threshold can be adjusted depending on business requirements.

### Model Artifacts

The trained model and its associated feature-column information are serialized using Joblib:

- `churn_model.joblib`
- `model_columns.joblib`

These artifacts are loaded by the FastAPI application to generate predictions.


A Machine Learning-powered REST API that predicts whether a customer is likely to leave a service based on their account information, service usage, and billing details.

Built with **Python, Scikit-learn, Pandas, Joblib, and FastAPI**.

---

## Overview

Customer churn occurs when a customer stops using a company's products or services.

For businesses such as telecom providers, understanding which customers are likely to leave is important. Early predictions can help businesses identify at-risk customers and take action to improve retention.

This project uses Machine Learning to estimate the probability of customer churn and exposes the trained model through a REST API.

**The main idea:** Developers can integrate customer churn predictions into their own applications using HTTP requests, without needing to download or run the Machine Learning model themselves.

## Features

- Machine Learning-based churn prediction.
- Probability score for each customer.
- Customizable classification threshold.
- REST API built with FastAPI.
- Input validation using Pydantic.
- Automatic feature engineering and preprocessing.
- JSON-based request and response.
- Interactive API documentation with Swagger UI.
- Model inference through a reusable API endpoint.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| Scikit-learn | Model training and prediction |
| Joblib | Model serialization |
| FastAPI | REST API development |
| Pydantic | Request validation |
| Uvicorn | ASGI server |

## Dataset

This project uses the **IBM Telco Customer Churn Dataset**.

- Total records: 7,043 customers
- Features: 20 customer attributes, plus the target column
- Target variable: `Churn`
- Target values: `Yes` and `No`

The dataset contains customer information such as:

- Demographics
- Contract type
- Internet and phone services
- Payment method
- Tenure
- Monthly charges
- Total charges

Dataset source:

[IBM Telco Customer Churn Dataset](https://github.com/IBM/telco-customer-churn-on-icp4d)

## Machine Learning Workflow

The project follows a standard supervised Machine Learning workflow:

1. **Data collection:** Load the Telco Customer Churn dataset.
2. **Data preprocessing:** Handle data types, missing values, and categorical features.
3. **Exploratory Data Analysis:** Understand customer patterns and relationships between features.
4. **Feature engineering:** Create additional features such as total services and price increase ratio.
5. **Model training:** Train and evaluate classification models.
6. **Model evaluation:** Compare model performance using classification metrics.
7. **Model serialization:** Save the trained model using Joblib.
8. **API integration:** Load the model into FastAPI and expose predictions through an HTTP endpoint.

## API Architecture

The trained model runs on the API server. External applications communicate with it through HTTP requests.

```text
        Developer Applications
      ┌─────────┬──────────┬─────────┐
      │ React   │ Mobile   │ Backend │
      │ Frontend│ App      │ Service │
      └────┬────┴────┬─────┴────┬────┘
           │        │          │
           └────────┴──────────┘
                    │
                HTTP POST
                    │
          ┌─────────▼─────────┐
          │   FastAPI Server  │
          │                   │
          │  Input Validation │
          │  Feature Creation │
          │  Preprocessing    │
          │  Model Prediction │
          └─────────┬─────────┘
                    │
             ┌──────▼──────┐
             │ Trained ML  │
             │    Model    │
             └──────┬──────┘
                    │
             Prediction JSON
                    │
          ┌─────────▼─────────┐
          │ Client Application│
          └───────────────────┘
```

## Project Structure

```text
customer-churn-api/
│
├── main.py
│
├── model/
│   ├── churn_model.joblib
│   └── model_columns.joblib
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-api.git
cd customer-churn-api
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the API server

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### 5. Explore API documentation

Open the following URL in your browser:

```text
http://127.0.0.1:8000/docs
```

FastAPI provides interactive Swagger documentation where developers can test the endpoints directly.

---

## API Endpoints

### 1. Health Check

**GET** `/`

Checks whether the API server is running.

Example response:

```json
{
  "message": "Customer Churn Prediction Api is running"
}
```

### 2. Predict Customer Churn

**POST** `/predict`

Accepts customer information and returns a churn prediction with its probability.

#### Example Request

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.5,
  "TotalCharges": 1026.0
}
```

#### Example Response

```json
{
  "churn_probability": 0.7234,
  "churn_prediction": "Yes",
  "threshold_used": 0.38
}
```

*The response above is illustrative. Actual predictions depend on the trained model and input data.*

### Response Fields

| Field | Description |
|---|---|
| `churn_probability` | Estimated probability of customer churn |
| `churn_prediction` | Predicted class: Yes or No |
| `threshold_used` | Classification threshold used for the decision |

### Understanding the Prediction

The model returns a probability between 0 and 1.

For example:

- `0.80` means an estimated 80% churn probability.
- `0.25` means an estimated 25% churn probability.

The current classification threshold is `0.38`.

- Probability >= 0.38 → Churn: Yes
- Probability < 0.38 → Churn: No

The threshold determines the classification decision; it does not change the probability generated by the model.

---

## Using the API in Your Own Application

One of the main goals of this project is reusability.

A developer does not need to install Scikit-learn, download the model, or understand the internal ML implementation.

They only need to send a valid HTTP request.

### JavaScript Example

```javascript
async function predictChurn(customerData) {
    const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(customerData)
        }
    );

    if (!response.ok) {
        throw new Error("Prediction request failed");
    }

    return await response.json();
}
```

Once deployed, replace the local URL with the public API URL.

This makes the API usable from React, Vue, mobile applications, or other backend services.

---

## Future Improvements

- Deploy the API to a cloud hosting platform.
- Add API key authentication.
- Implement rate limiting.
- Add API versioning (`/api/v1/predict`).
- Improve preprocessing using a reusable Scikit-learn Pipeline.
- Add automated tests for prediction and validation.
- Monitor model performance and data drift.
- Add batch prediction for multiple customers.

## Important Notes

- The model files must be available in the `model/` directory.
- The inference preprocessing must remain consistent with the preprocessing used during model training.
- Do not expose model files or secret API keys in a public frontend.
- The prediction is an estimate, not a guarantee that a customer will leave.

## License

This project is open-source. Add a license file if you intend to distribute it under a specific license.


