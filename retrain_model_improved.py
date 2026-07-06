# retrain_model_improved.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("IMPROVED MODEL TRAINING")
print("="*60)

# -----------------------------
# 1. LOAD DATA
# -----------------------------
print("\n1. Loading dataset...")
df = pd.read_csv('india_all_states_big.csv')
print(f"   ✅ Loaded {len(df)} rows")
print(f"   Columns: {df.columns.tolist()}")

# -----------------------------
# 2. FEATURE ENGINEERING (NEW!)
# -----------------------------
print("\n2. Creating new features...")

# Create month numbers (1-12)
month_to_num = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}
df['Month_Num'] = df['Month'].map(month_to_num)

# Create season feature
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8, 9]:
        return 'Monsoon'
    else:
        return 'Post-Monsoon'

df['Season'] = df['Month_Num'].apply(get_season)

# Interaction features (combine related variables)
df['Temp_Humidity_Ratio'] = df['Temperature'] / (df['Humidity'] + 0.1)
df['Rainfall_Temp'] = df['Rainfall'] * df['Temperature']
df['LAI_Rainfall'] = df['LAI'] * df['Rainfall']
df['Population_Temp'] = df['Population_Density'] * df['Temperature']
df['Temp_Squared'] = df['Temperature'] ** 2
df['Rainfall_Log'] = np.log1p(df['Rainfall'])
df['Humidity_Squared'] = df['Humidity'] ** 2

# Seasonal flags
df['Is_Summer'] = (df['Season'] == 'Summer').astype(int)
df['Is_Monsoon'] = (df['Season'] == 'Monsoon').astype(int)
df['Is_Winter'] = (df['Season'] == 'Winter').astype(int)

print(f"   ✅ Created {len(df.columns) - 8} new features")

# -----------------------------
# 3. ENCODE CATEGORICAL VARIABLES
# -----------------------------
print("\n3. Encoding categorical variables...")
state_encoder = LabelEncoder()
month_encoder = LabelEncoder()
season_encoder = LabelEncoder()
disease_encoder = LabelEncoder()

df['State_encoded'] = state_encoder.fit_transform(df['State'])
df['Month_encoded'] = month_encoder.fit_transform(df['Month'])
df['Season_encoded'] = season_encoder.fit_transform(df['Season'])
df['Disease_encoded'] = disease_encoder.fit_transform(df['Disease'])

# -----------------------------
# 4. SELECT FEATURES (EXPANDED!)
# -----------------------------
print("\n4. Selecting features...")

features = [
    # Original features
    'Temperature', 'Humidity', 'Rainfall', 'LAI', 'Population_Density',
    'State_encoded', 'Month_encoded',
    
    # New features
    'Temp_Humidity_Ratio', 'Rainfall_Temp', 'LAI_Rainfall', 'Population_Temp',
    'Temp_Squared', 'Rainfall_Log', 'Humidity_Squared',
    'Is_Summer', 'Is_Monsoon', 'Is_Winter', 'Season_encoded'
]

X = df[features]
y = df['Disease_encoded']

print(f"   Total features: {len(features)}")
print(f"   Features: {features[:5]}... + {len(features)-5} more")

# -----------------------------
# 5. SCALE FEATURES (IMPORTANT FOR SOME MODELS)
# -----------------------------
print("\n5. Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 6. TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training: {len(X_train)} rows")
print(f"   Testing: {len(X_test)} rows")

# -----------------------------
# 7. TRY MULTIPLE MODELS
# -----------------------------
print("\n6. Training multiple models...")
print("-"*40)

models = {
    'Random Forest (Current)': RandomForestClassifier(n_estimators=100, random_state=42),
    'Random Forest (Tuned)': RandomForestClassifier(
        n_estimators=300, 
        max_depth=15, 
        min_samples_split=5,
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=150, 
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
}

results = {}
best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"   {name:25} | Accuracy: {accuracy:.2%}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name

print("-"*40)
print(f"🏆 BEST MODEL: {best_name} with {best_accuracy:.2%} accuracy")

# -----------------------------
# 8. HYPERPARAMETER TUNING (OPTIONAL - UNCOMMENT TO USE)
# -----------------------------
print("\n7. Hyperparameter tuning for best model...")

param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,  # 3-fold cross validation (faster)
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"   Best parameters: {grid_search.best_params_}")
print(f"   Best CV accuracy: {grid_search.best_score_:.2%}")

tuned_model = grid_search.best_estimator_
tuned_accuracy = accuracy_score(y_test, tuned_model.predict(X_test))
print(f"   Tuned model test accuracy: {tuned_accuracy:.2%}")

# Use tuned model if better
if tuned_accuracy > best_accuracy:
    best_model = tuned_model
    best_accuracy = tuned_accuracy
    print(f"   ✅ Using tuned model (better!)")

# -----------------------------
# 9. SAVE EVERYTHING
# -----------------------------
print("\n8. Saving model and encoders...")
joblib.dump(best_model, 'model.pkl')
joblib.dump(state_encoder, 'state_encoder.pkl')
joblib.dump(month_encoder, 'month_encoder.pkl')
joblib.dump(season_encoder, 'season_encoder.pkl')
joblib.dump(disease_encoder, 'disease_encoder.pkl')
joblib.dump(features, 'features.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Save scaler for predictions

# -----------------------------
# 10. DETAILED RESULTS
# -----------------------------
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"✅ Model saved as 'model.pkl'")
print(f"✅ Accuracy improved from 60.20% to {best_accuracy:.2%}")
print(f"✅ Improvement: +{(best_accuracy - 0.6020)*100:.1f}%")

# Show per-class performance
y_pred = best_model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=disease_encoder.classes_))

print("\n" + "="*60)
print("🎉 Training complete! Now run: python app.py")
print("="*60)