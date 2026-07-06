import pandas as pd
import numpy as np

# Load original dataset
df = pd.read_csv("india_all_states.csv")

# Create synthetic dataset
synthetic_data = pd.DataFrame()

for col in df.columns:
    if df[col].dtype == 'object':
        # categorical column
        synthetic_data[col] = np.random.choice(df[col], 5000)
    else:
        # numeric column
        synthetic_data[col] = np.random.uniform(df[col].min(), df[col].max(), 5000)

# Save new dataset
synthetic_data.to_csv("india_all_states_big.csv", index=False)

print("✅ New dataset created with shape:", synthetic_data.shape)