# balance_dataset.py
import pandas as pd

df = pd.read_csv('india_all_states_big.csv')

print("="*50)
print("BALANCING DATASET ")
print("="*50)

print("\nOriginal distribution:")
print(df['Disease'].value_counts())

# Take 500 samples from each disease
balanced_dfs = []
for disease in df['Disease'].unique():
    disease_data = df[df['Disease'] == disease]
    n_samples = min(500, len(disease_data))
    sampled = disease_data.sample(n=n_samples, random_state=42)
    balanced_dfs.append(sampled)
    print(f"{disease}: {len(disease_data)} → {len(sampled)} (kept {n_samples})")

balanced_df = pd.concat(balanced_dfs)

print("\n✅ Balanced distribution:")
print(balanced_df['Disease'].value_counts())

# Save
balanced_df.to_csv('india_all_states_balanced.csv', index=False)
print("\n✅ Saved as 'india_all_states_balanced.csv'")
print(f"\nTotal rows in balanced dataset: {len(balanced_df)}")