Stock Demand Prediction Web App

A Flask-based web application for forecasting stock units using historical sales data. Users can upload CSV files, get predictions, view plots of actual vs predicted values, and download the results.


🛠 Features

➢ Upload CSV with sales, stock, price, and date information.

➢ Automatic column mapping to standardize input.

➢ Feature engineering: lag features, rolling statistics, day/week/month features, sales-to-stock ratio.

➢ Train a Random Forest model on the uploaded dataset.

➢ Predict stock units and future 7-day stock demand.

➢ Interactive plot of actual vs predicted stock units using Plotly.

• Download predicted CSV for further analysis.

📂 Project Structure

project/
│
├─ app.py                  # Main Flask app
├─ templates/
│   ├─ index.html          # Upload page
│   └─ results.html        # Results page
├─ uploads/                # Folder for uploaded CSV & model
├─ training/
│   └─ mock_kaggle.csv     # Sample dataset
├─ model.pkl               # Trained Random Forest model
├─ requirements.txt        # Project dependencies
└─ README.md

* Requirements

• Python 3.10+ recommended

• Install dependencies:

• pip install -r requirements.txt


Example requirements.txt:

Flask==2.3.4
pandas==2.1.1
numpy==1.26.4
scikit-learn==1.3.3
plotly==5.16.0
matplotlib==3.8.0
seaborn==0.12.3


⚠️ Versions may vary depending on your Python version.

🚀 How to Run

Clone/download the project folder.

Place your CSV in the uploads/ folder or upload via the web interface.

Run the Flask app:

python app.py


Open your browser at: http://127.0.0.1:5000

Upload CSV → View predictions → Download results.

📈 Screenshots





📝 Notes

CSV columns are automatically mapped using standard names (Date, Sales, Stock_Units, Price).

Future predictions generate 7-day forecast by default.

The app saves trained models in uploads/model.pkl to avoid retraining every time.

💡 Author

Mokarala Swetha – Final Year Project
