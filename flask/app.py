from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Pre-defined model features
model_features = [
    'Price', 'Sales', 'day_of_week', 'month', 'day_of_month',
    'week_of_year', 'lag_1', 'lag_7', 'rolling_7', 'rolling_std_7', 'sales_stock_ratio'
]

# Column mapping options
COLUMN_MAPPING = {
    'Date': ['data', 'date', 'day'],
    'Sales': ['venda', 'sales', 'sold', 'vendas', 'sold_units'],
    'Stock_Units': ['estoque', 'stock_units', 'quantity', 'qty'],
    'Price': ['preco', 'price', 'cost', 'preços']
}

# =================== Column Mapping ===================
def map_columns_fallback(df):
    mapped = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for standard_name, possible_names in COLUMN_MAPPING.items():
        found = False
        for name in possible_names:
            if name.lower() in lower_cols:
                mapped[lower_cols[name.lower()]] = standard_name
                found = True
                break
        if not found:
            for col in df.columns:
                if col in mapped:
                    continue
                if standard_name == 'Date':
                    try:
                        pd.to_datetime(df[col])
                        mapped[col] = standard_name
                        break
                    except:
                        continue
                else:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        mapped[col] = standard_name
                        break
    df = df.rename(columns={k: v for k, v in mapped.items()})
    return df, mapped


# =================== Feature Engineering ===================
def generate_features(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)

    # Time features
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['day_of_month'] = df['Date'].dt.day
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)

    # Lag features
    df['lag_1'] = df['Stock_Units'].shift(1)
    df['lag_7'] = df['Stock_Units'].shift(7)

    # Rolling stats
    df['rolling_7'] = df['Stock_Units'].rolling(7).mean()
    df['rolling_std_7'] = df['Stock_Units'].rolling(7).std()

    # Ratio feature
    df['sales_stock_ratio'] = df['Sales'] / (df['Stock_Units'] + 1e-5)

    # Handle missing values
    for col in ['lag_1', 'lag_7', 'rolling_7', 'rolling_std_7']:
        df[col] = df[col].fillna(df[col].mean())

    df['Stock_Units'] = df['Stock_Units'].fillna(method='ffill').fillna(df['Stock_Units'].median())
    df['Sales'] = df['Sales'].fillna(method='ffill').fillna(df['Sales'].median())
    df['Price'] = df['Price'].fillna(method='ffill').fillna(df['Price'].median())

    return df


# =================== Future Predictions ===================
def predict_future(df, model, days=7):
    last_known = df['Stock_Units'].tolist()[-7:]
    last_sales = df['Sales'].iloc[-1]
    future_preds = []
    last_date = df['Date'].max()

    for i in range(days):
        next_date = last_date + pd.Timedelta(days=1)
        features = {
            'Price': df['Price'].iloc[-1],
            'Sales': last_sales,
            'lag_1': last_known[-1],
            'lag_7': last_known[-7] if len(last_known) >= 7 else last_known[-1],
            'rolling_7': np.mean(last_known[-7:]),
            'rolling_std_7': np.std(last_known[-7:]),
            'sales_stock_ratio': last_sales / (last_known[-1] + 1e-5),
            'day_of_week': next_date.dayofweek,
            'month': next_date.month,
            'day_of_month': next_date.day,
            'week_of_year': next_date.isocalendar()[1]
        }
        X_future = pd.DataFrame([features])[model_features]
        pred = model.predict(X_future)[0]
        future_preds.append({'Date': next_date, 'Predicted_Venda': pred})
        last_known.append(pred)
        last_known.pop(0)
        last_date = next_date

    return future_preds


# =================== Flask Routes ===================
@app.route('/', methods=['GET', 'POST'])
def index():
    table_data = None
    error_msg = None
    plot_html = None
    metrics = None
    mapped_columns = None
    future_predictions = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            try:
                df = pd.read_csv(filepath)

                # Column mapping
                df, mapped = map_columns_fallback(df)
                mapped_columns = {v: k for k, v in mapped.items()}

                # Check required columns
                for col in ['Date', 'Sales', 'Stock_Units', 'Price']:
                    if col not in df.columns:
                        raise KeyError(f"❌ Could not detect required column '{col}' even with fallback.")

                # Feature engineering
                df = generate_features(df)

                # Ensure all model features exist
                missing_features = [f for f in model_features if f not in df.columns]
                if missing_features:
                    raise KeyError(f"Missing features required by the model: {missing_features}")

                # ===== Prepare data =====
                X = df[model_features]
                y = df['Stock_Units']

                # ===== Define preprocessing pipeline =====
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

                # ===== Build pipeline model =====
                model_pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('model', RandomForestRegressor(
                        n_estimators=1000,
                        max_depth=22,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        max_features='sqrt',
                        bootstrap=True,
                        random_state=42,
                        n_jobs=-1
                    ))
                ])

                # Train model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model_pipeline.fit(X_train, y_train)

                # Predictions
                df['Predicted_Venda'] = model_pipeline.predict(X)

                # Save model
                model_file = os.path.join(app.config['UPLOAD_FOLDER'], 'model.pkl')
                with open(model_file, 'wb') as f:
                    pickle.dump(model_pipeline, f)

                # Evaluate
                y_test_pred = model_pipeline.predict(X_test)
                mse = mean_squared_error(y_test, y_test_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_test_pred)
                metrics = {'MSE': mse, 'RMSE': rmse, 'R2': r2}

                # Future prediction
                future_predictions = predict_future(df, model_pipeline, days=7)

                # Save CSV for download
                download_file = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions.csv')
                df.to_csv(download_file, index=False)

                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Stock_Units'], mode='lines+markers',
                                         name='Actual', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Predicted_Venda'], mode='lines+markers',
                                         name='Predicted', line=dict(color='red')))
                fig.update_layout(title='Actual vs Predicted Stock Units',
                                  xaxis_title='Date', yaxis_title='Stock Units')
                plot_html = pio.to_html(fig, full_html=False)

                table_data = df.to_dict(orient='records')

        # ✅ Return results.html after successful processing
                return render_template('results.html',
                                       table_data=table_data,
                                       error_msg=None,
                                       plot_html=plot_html,
                                       metrics=metrics,
                                       mapped_columns=mapped_columns,
                                       future_predictions=future_predictions)
            except Exception as e:
                error_msg = f"❌ Error: {e}"
        else:
            error_msg = "❌ Please upload a valid CSV file."

    return render_template('index.html',
                           table_data=table_data,
                           error_msg=error_msg,
                           plot_html=plot_html,
                           metrics=metrics,
                           mapped_columns=mapped_columns,
                           future_predictions=future_predictions)


@app.route('/download')
def download():
    download_file = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions.csv')
    if os.path.exists(download_file):
        return send_file(download_file, as_attachment=True)
    return "❌ No file available for download."


if __name__ == '__main__':
    app.run(debug=True)
