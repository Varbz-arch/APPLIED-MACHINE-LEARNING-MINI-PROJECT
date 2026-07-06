# retrain_model.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

print("Loading dataset...")
df = pd.read_csv('india_all_states_big.csv')
print(f"✅ Loaded {len(df)} rows")

# Encode categorical variables
print("Encoding categorical variables...")
state_encoder = LabelEncoder()
month_encoder = LabelEncoder()
disease_encoder = LabelEncoder()

df['State_encoded'] = state_encoder.fit_transform(df['State'])
df['Month_encoded'] = month_encoder.fit_transform(df['Month'])
df['Disease_encoded'] = disease_encoder.fit_transform(df['Disease'])

# Features (NO latitude/longitude)
features = ['Temperature', 'Humidity', 'Rainfall', 'LAI', 'Population_Density', 'State_encoded', 'Month_encoded']
X = df[features]
y = df['Disease_encoded']

print(f"Features: {features}")
print(f"Target classes: {list(disease_encoder.classes_)}")

# Train model
print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save everything
print("Saving model and encoders...")
joblib.dump(model, 'model.pkl')
joblib.dump(state_encoder, 'state_encoder.pkl')
joblib.dump(month_encoder, 'month_encoder.pkl')
joblib.dump(disease_encoder, 'disease_encoder.pkl')
joblib.dump(features, 'features.pkl')

accuracy = model.score(X_test, y_test)
print(f"✅ Done! Model accuracy: {accuracy:.2%}")