import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("MODELTRAINING STARTED")
print("="*60)

# 1. LOAD DATA

df = pd.read_csv("india_all_states.csv")

# 2. USE TOP 3 DISEASES

df.loc[df['Temperature'] > 40, 'Disease'] = 'Heatstroke'
top_n = 3
top_diseases = df['Disease'].value_counts().nlargest(top_n).index
df = df[df['Disease'].isin(top_diseases)]

print(f"\nUsing diseases: {list(top_diseases)}")
print(df['Disease'].value_counts())

# 3. FEATURE ENGINEERING

month_to_num = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

df['Month_Num'] = df['Month'].map(month_to_num)

def get_season(month):
    if month in [12,1,2]: return 'Winter'
    elif month in [3,4,5]: return 'Summer'
    elif month in [6,7,8,9]: return 'Monsoon'
    else: return 'Post-Monsoon'

df['Season'] = df['Month_Num'].apply(get_season)

# Feature engineering
df['Temp_Humidity_Ratio'] = df['Temperature'] / (df['Humidity'] + 0.1)
df['Rainfall_Temp'] = df['Rainfall'] * df['Temperature']
df['LAI_Rainfall'] = df['LAI'] * df['Rainfall']
df['Population_Temp'] = df['Population_Density'] * df['Temperature']
df['Temp_Squared'] = df['Temperature'] ** 2
df['Rainfall_Log'] = np.log1p(df['Rainfall'])
df['Humidity_Squared'] = df['Humidity'] ** 2

df['Is_Summer'] = (df['Season'] == 'Summer').astype(int)
df['Is_Monsoon'] = (df['Season'] == 'Monsoon').astype(int)
df['Is_Winter'] = (df['Season'] == 'Winter').astype(int)

# 4. ENCODING

state_encoder = LabelEncoder()
month_encoder = LabelEncoder()
season_encoder = LabelEncoder()
disease_encoder = LabelEncoder()

df['State_encoded'] = state_encoder.fit_transform(df['State'])
df['Month_encoded'] = month_encoder.fit_transform(df['Month'])
df['Season_encoded'] = season_encoder.fit_transform(df['Season'])
df['Disease_encoded'] = disease_encoder.fit_transform(df['Disease'])

# 5. FEATURES

features = [
    'Temperature', 'Humidity', 'Rainfall', 'LAI', 'Population_Density',
    'State_encoded', 'Month_encoded',
    'Temp_Humidity_Ratio', 'Rainfall_Temp', 'LAI_Rainfall', 'Population_Temp',
    'Temp_Squared', 'Rainfall_Log', 'Humidity_Squared',
    'Is_Summer', 'Is_Monsoon', 'Is_Winter', 'Season_encoded'
]

X = df[features]
y = df['Disease_encoded']

# 6. SCALING

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 8. MODEL TRAINING (FINAL)

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42
)

rf_model.fit(X_train, y_train)

# Optional: try Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=200, max_depth=5)
gb_model.fit(X_train, y_train)

# Compare models
rf_acc = rf_model.score(X_test, y_test)
gb_acc = gb_model.score(X_test, y_test)

print(f"\nRandom Forest Accuracy: {rf_acc:.2%}")
print(f"Gradient Boosting Accuracy: {gb_acc:.2%}")

# Choose best
model = rf_model if rf_acc >= gb_acc else gb_model
print(f" Using: {'Random Forest' if model == rf_model else 'Gradient Boosting'}")

# 9. EVALUATION

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n Final Accuracy: {accuracy:.2%}")

cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print(f" Cross-validation accuracy: {cv_scores.mean():.2%}")

print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=disease_encoder.classes_))

# 9. CONFUSION MATRIX 

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=disease_encoder.classes_,
            yticklabels=disease_encoder.classes_)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

importances = model.feature_importances_
feature_names = features

plt.figure(figsize=(8,5))
plt.barh(feature_names, importances)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# 10. SAVE MODEL

joblib.dump(model, 'model.pkl')
joblib.dump(state_encoder, 'state_encoder.pkl')
joblib.dump(month_encoder, 'month_encoder.pkl')
joblib.dump(season_encoder, 'season_encoder.pkl')
joblib.dump(disease_encoder, 'disease_encoder.pkl')
joblib.dump(features, 'features.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\n Model saved successfully!")
print("="*60)