import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ===== 1. Load dataset =====
df = pd.read_csv("uploads/mock_kaggle.csv")

# ===== 2. Rename columns =====
df = df.rename(columns={
    "data": "Date",
    "venda": "Sales",
    "estoque": "Stock_Units",
    "preco": "Price"
})

# ===== 3. Convert Date column and create features =====
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['day_of_week'] = df['Date'].dt.dayofweek
df['month'] = df['Date'].dt.month
df['day_of_month'] = df['Date'].dt.day
df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)

# ===== 4. Create lag & rolling features =====
df = df.sort_values(by='Date').reset_index(drop=True)
df['lag_1'] = df['Stock_Units'].shift(1)
df['lag_7'] = df['Stock_Units'].shift(7)
df['rolling_7'] = df['Stock_Units'].rolling(7).mean()
df['rolling_std_7'] = df['Stock_Units'].rolling(7).std()
df['sales_stock_ratio'] = df['Sales'] / (df['Stock_Units'] + 1)

# ===== 🧹 Handle Missing & Zero Values =====
numeric_cols = ['Sales', 'Stock_Units', 'Price', 'lag_1', 'lag_7', 'rolling_7']
for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)
    df[col] = df[col].replace(0, df[col].mean())

# Fill object (categorical) NaNs with 'Unknown'
for col in df.select_dtypes(include='object').columns:
    df[col].fillna("Unknown", inplace=True)

# ===== 5. Data Visualization =====
# 5a. Trend plot
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Stock_Units'], color='blue')
plt.title("Stock Units Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Stock Units")
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5b. Histogram
plt.figure(figsize=(8,5))
plt.hist(df['Stock_Units'], bins=20, color='orange', edgecolor='black')
plt.title("Distribution of Stock Units")
plt.xlabel("Stock Units")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle="--", alpha=0.5)
plt.show()

# 5c. Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# ===== 6. Prepare features and target =====
X = df.drop(["Stock_Units", "Date"], axis=1)
y = df["Stock_Units"]

# ===== 7. Define preprocessing pipeline =====
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ===== 8. Full pipeline with model =====
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=2000,
        max_depth=35,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ))
])

# ===== 9. Split data =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== 10. Train pipeline =====
model_pipeline.fit(X_train, y_train)

# ===== 11. Predictions =====
y_pred = model_pipeline.predict(X_test)

# ===== 12. Evaluation =====
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
r2_percent = r2 * 100

print("Random Forest Evaluation (Pipeline):")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2 Score: {r2:.4f}")
print(f"Approx. Accuracy (R2 %): {r2_percent:.2f}%")

# ===== 13. Plot Actual vs Predicted =====
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color="blue")
plt.xlabel("Actual Stock Units")
plt.ylabel("Predicted Stock Units")
plt.title("Random Forest - Actual vs Predicted (Pipeline)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("pipeline_results.png")
plt.show()

# ===== 14. Save model =====
with open("model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)

print("✅ Pipeline-based model.pkl created successfully!")
