# 🌍 GeoHealth-Disease-Prediction
## GeoHealth Risk Forecasting and Preventive Healthcare System

GeoHealth is a Machine Learning-based healthcare prediction system that forecasts climate-sensitive disease risks using environmental and geographical factors.

The system analyzes parameters such as:

- Temperature
- Humidity
- Rainfall
- Location (State)
- Month/Season
- Population Density
- Environmental factors

and predicts possible disease risks along with preventive healthcare suggestions.

---

# 🚀 Features

✅ Climate-based disease prediction  
✅ Machine Learning powered prediction system  
✅ State and month-based risk analysis  
✅ Disease prevention recommendations  
✅ Medical attention guidelines  
✅ Environmental feature engineering  
✅ Flask REST API backend  
✅ Interactive web interface  
✅ Saved trained ML model for instant predictions  

---

# 🦠 Diseases Covered

The current system supports:

- Typhoid
- Cholera
- Malaria
- Dengue
- Heatstroke

---

# 📂 Project Structure

```
GeoHealth-AI/
│
├── app.py                       # Flask backend application
├── train_model.py               # ML model training script
├── suggestions.py               # Disease suggestions module
│
├── templates/
│   └── index.html               # Frontend HTML
│
├── model.pkl                    # Trained ML model
├── scaler.pkl                   # Feature scaler
├── features.pkl                 # Feature list
├── state_encoder.pkl            # State label encoder
├── month_encoder.pkl            # Month encoder
├── season_encoder.pkl           # Season encoder
├── disease_encoder.pkl          # Disease encoder
│
├── india_all_states.csv         # Dataset
├── requirements.txt             # Python dependencies
└── README.md
```

---

# 🛠️ Technologies Used

## Programming Language
- Python

## Machine Learning
- Scikit-learn
- Pandas
- NumPy
- Imbalanced-learn

## Data Visualization
- Matplotlib
- Seaborn

## Backend
- Flask
- Flask-CORS

## Frontend
- HTML
- CSS
- JavaScript

---

# ⚙️ Installation Guide (VS Code)

## 1. Clone the Repository

Open VS Code terminal:

```bash
https://github.com/Varbz-arch/APPLIED-MACHINE-LEARNING-MINI-PROJECT.git
```

Navigate into the project:

```bash
cd GeoHealth-Disease-Prediction
```

---

# 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# 3. Install Dependencies

Run:

```bash
pip install -r requirements.txt
```

---

# 4. Verify Installation

Check Python version:

```bash
python --version
```

Recommended:

```
Python 3.10+
```

---

# 🧠 Machine Learning Model Training

The ML pipeline is implemented in:

```
train_model.py
```

The training process includes:

## Data Processing

- Dataset loading
- Disease filtering
- Label encoding
- Feature scaling

## Feature Engineering

Generated features:

- Temperature-Humidity Ratio
- Rainfall × Temperature
- LAI × Rainfall
- Population Density × Temperature
- Temperature Squared
- Humidity Squared
- Log Rainfall
- Seasonal indicators

---

# 🤖 Model Training

Two models are trained:

## Random Forest Classifier

Configuration:

- 500 estimators
- Maximum depth = 15
- Balanced class weights


## Gradient Boosting Classifier

Configuration:

- 200 estimators
- Maximum depth = 5


The model with the highest accuracy is automatically selected.

---

# 📊 Model Performance

Training Dataset:

```
5000 records
```

Final trained diseases:

```
Typhoid
Heatstroke
Cholera
```

Performance:

| Metric | Score |
|---|---|
| Final Accuracy | 71.58% |
| Cross Validation Accuracy | 72.48% |
| Selected Model | Gradient Boosting |

Evaluation methods:

- Accuracy Score
- Classification Report
- Confusion Matrix
- Cross Validation
- Feature Importance Analysis

---

# 💾 Generated Model Files

After training, the following files are generated:

```
model.pkl
scaler.pkl
features.pkl
state_encoder.pkl
month_encoder.pkl
season_encoder.pkl
disease_encoder.pkl
```

These files store:

- Trained machine learning model
- Encoders
- Scaling parameters
- Required feature information

They allow the Flask application to directly make predictions without retraining.

---

# 🌐 Running the Application

After installation:

## Step 1: Train Model (optional)

If `.pkl` files already exist:

Skip this step.

Otherwise:

```bash
python train_model.py
```

---

## Step 2: Start Flask Server

Run:

```bash
python app.py
```

The application will start at:

```
http://127.0.0.1:5000
```

Open this URL in your browser.

---

# 🔌 API Documentation

## Disease Prediction API

### Endpoint

```
POST /predict
```

Example Request:

```json
{
    "state":"Assam",
    "month":"July",
    "temp":32,
    "humidity":85,
    "rainfall":200
}
```

Example Response:

```json
{
    "success":true,
    "predictions":[
        {
            "disease":"Malaria",
            "probability":0.82
        }
    ]
}
```

---

# 🩺 Health Suggestion API

### Endpoint

```
POST /get_suggestions
```

Request:

```json
{
    "disease":"Malaria"
}
```

Returns:

- What to do
- Prevention methods
- Risk factors
- Medical attention signs
- General health tips

---

# 🔄 System Workflow

```
Environmental Data
        |
        ↓
Data Processing
        |
        ↓
Feature Engineering
        |
        ↓
Machine Learning Model
        |
        ↓
Disease Prediction
        |
        ↓
Healthcare Suggestions
```

---

# 📌 Important Notes

- The model provides predictions based on environmental patterns.
- It is intended for educational and research purposes.
- Predictions should not replace professional medical diagnosis.

---

# 🔮 Future Enhancements

- Real-time weather API integration
- Live disease outbreak monitoring
- GIS-based disease risk maps
- More disease classes
- Deep learning models
- Mobile application
- Cloud deployment
- User authentication

---

# 👩‍💻 Contributors

Developed as part of the **Applied Machine Learning Project**.

**Team Members:**

- ARCHISHA BORAH
- SHREYA GUPTA
- TANISHQ TYAGI
- ABHI GUPTA

---

# 📄 License

This project is developed for educational purposes.
