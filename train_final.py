# train_optimized_binary.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("OPTIMIZED BINARY CLASSIFIER ")
print("="*60)

# Load data
df = pd.read_csv('india_all_states_balanced.csv')
print(f"\n✅ Loaded {len(df)} rows")

# Create binary target
df['Is_Typhoid'] = (df['Disease'] == 'Typhoid').astype(int)
print(f"\n📊 Target distribution:")
print(f"   Typhoid: {df['Is_Typhoid'].sum()} ({df['Is_Typhoid'].mean()*100:.1f}%)")
print(f"   Others: {(~df['Is_Typhoid'].astype(bool)).sum()} ({(1-df['Is_Typhoid'].mean())*100:.1f}%)")

# Encode categorical variables
state_encoder = LabelEncoder()
month_encoder = LabelEncoder()

df['State_encoded'] = state_encoder.fit_transform(df['State'])
df['Month_encoded'] = month_encoder.fit_transform(df['Month'])

# ADD MORE FEATURES (but carefully selected)
month_to_num = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}
df['Month_Num'] = df['Month'].map(month_to_num)

# Simple but effective features
df['Temp_Humidity'] = df['Temperature'] / (df['Humidity'] + 0.1)
df['Rainfall_Intensity'] = df['Rainfall'] / (df['Humidity'] + 0.1)
df['Is_High_Rain'] = (df['Rainfall'] > 150).astype(int)
df['Is_High_Temp'] = (df['Temperature'] > 35).astype(int)

# Selected features (11 features - optimal balance)
features = [
    'Temperature', 'Humidity', 'Rainfall', 'LAI', 'Population_Density',
    'State_encoded', 'Month_encoded', 'Month_Num',
    'Temp_Humidity', 'Rainfall_Intensity', 'Is_High_Rain'
]

X = df[features]
y = df['Is_Typhoid']

print(f"\n✅ Features: {len(features)} features")

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n📊 Training: {len(X_train)} rows")
print(f"📊 Testing: {len(X_test)} rows")

# Apply SMOTE with different k_neighbors
print("\n🔄 Applying SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"   Before SMOTE: {len(X_train)}")
print(f"   After SMOTE: {len(X_train_bal)}")

# Try Gradient Boosting (often better for binary classification)
print("\n🤖 Training Gradient Boosting...")
model = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
model.fit(X_train_bal, y_train_bal)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*60)
print(f"🎯 ACCURACY: {accuracy:.2%}")
print("="*60)

# Cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print(f"📊 Cross-validation accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")

print("\n📋 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Other Diseases', 'Typhoid']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n📊 CONFUSION MATRIX:")
print(f"                 Predicted")
print(f"                 Others  Typhoid")
print(f"  Actual Others    {cm[0][0]:3}      {cm[0][1]:3}")
print(f"         Typhoid    {cm[1][0]:3}      {cm[1][1]:3}")

# Calculate and display key metrics
specificity = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
sensitivity = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0

print(f"\n📈 KEY METRICS:")
print(f"   Sensitivity (True Typhoid detection): {sensitivity:.2%}")
print(f"   Specificity (True Other detection): {specificity:.2%}")

# Save all files
print("\n💾 Saving model files...")
joblib.dump(model, 'model.pkl')
joblib.dump(state_encoder, 'state_encoder.pkl')
joblib.dump(month_encoder, 'month_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(features, 'features.pkl')

print("\n✅ Files saved successfully!")
print("="*60)
print("🎉 Training complete! Now run: python app.py")
print("="*60)