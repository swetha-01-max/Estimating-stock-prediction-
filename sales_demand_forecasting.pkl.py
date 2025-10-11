import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load your CSV with historical data
df = pd.read_csv("your_training_data.csv")

# Preprocess & feature engineering (same as in Flask app)
df = df.rename(columns={"data": "Date", "venda": "Sales", "estoque": "Stock_Units", "preco": "Price"})
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date').reset_index(drop=True)
df['day_of_week'] = df['Date'].dt.dayofweek
df['month'] = df['Date'].dt.month
df['day_of_month'] = df['Date'].dt.day
df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
df['lag_1'] = df['Stock_Units'].shift(1).fillna(0)
df['lag_7'] = df['Stock_Units'].shift(7).fillna(0)
df['rolling_7'] = df['Stock_Units'].rolling(7).mean().fillna(0)

# Features & target
model_features = ['Price', 'day_of_week', 'month', 'day_of_month', 'week_of_year', 'lag_1', 'lag_7', 'rolling_7']
X = df[model_features]
y = df['Stock_Units']

# Train model
model = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_split=5, min_samples_leaf=3, random_state=42, n_jobs=-1)
model.fit(X, y)

# Save model
with open("models/sales_demand_forecasting.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved successfully!")
