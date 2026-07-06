# # import pandas as pd

# # # Load dataset
# # data = pd.read_csv("C:/Users/shrey/OneDrive/Desktop/Shreya/GeoHealth-AI/final_data(Sheet1).csv")

# # # Show first 5 rows
# # print(data.head())

# # # Show column names
# # print("Columns in dataset:")
# # print(data.columns)
# # # Handle missing values
# # data = data.dropna()


# import pandas as pd

# # Load dataset
# data = pd.read_csv("C:/Users/shrey/OneDrive/Desktop/Shreya/GeoHealth-AI/final_data(Sheet1).csv")

# # Clean column names (remove spaces/newlines)
# data.columns = data.columns.str.strip()

# # Drop unwanted column
# data = data.drop(columns=["Unnamed: 14"], errors='ignore')

# # Remove missing values
# data = data.dropna()

# # Show cleaned columns
# print("Clean Columns:", data.columns)

# # Example: create risk level from Cases
# def risk_level(Cases):
#     if Cases < 50:
#         return "Low"
#     elif Cases < 200:
#         return "Medium"
#     else:
#         return "High"

# data['Risk'] = data['Cases'].apply(risk_level)
# print(data.head())



import pandas as pd

# -----------------------------
# STEP 1: Load dataset
# -----------------------------
data = pd.read_csv("india_all_states_big.csv")

# Store original row count
original_rows = len(data)

# -----------------------------
# STEP 2: Clean column names
# -----------------------------
data.columns = data.columns.str.strip()

print("\nColumns in dataset:")
print(data.columns)

# -----------------------------
# STEP 3: Check missing values
# -----------------------------
print("\nMissing values before cleaning:")
print(data.isnull().sum())

# Remove rows with missing values
data = data.dropna()

# -----------------------------
# STEP 4: Row information
# -----------------------------
cleaned_rows = len(data)
removed_rows = original_rows - cleaned_rows

print("\nOriginal rows:", original_rows)
print("Cleaned rows:", cleaned_rows)
print("Removed rows:", removed_rows)

# -----------------------------
# STEP 5: Create synthetic risk level
# -----------------------------
def risk_level(temp, humidity, rainfall):
    score = temp + humidity + rainfall

    if score < 150:
        return "Low"
    elif score < 250:
        return "Medium"
    else:
        return "High"

# -----------------------------
# STEP 6: Apply risk logic
# -----------------------------
data["Risk"] = data.apply(
    lambda row: risk_level(
        row["Temperature"],
        row["Humidity"],
        row["Rainfall"]
    ),
    axis=1
)

# -----------------------------
# STEP 7: Show sample output
# -----------------------------
print("\nSample Data:")
print(data[["State", "Disease", "Risk"]].head())

# -----------------------------
# STEP 8: Disease analysis
# -----------------------------
print("\nUnique diseases:")
print(data["Disease"].unique())

print("\nNumber of diseases:", data["Disease"].nunique())