# DemandCast AI

## Run/deploy
1. Create a GitHub repository named DemandCast-AI.
2. Upload `app.py` and `requirements.txt`.
3. Visit https://share.streamlit.io/ and sign in with GitHub.
4. Select **Create app**, choose your repository, branch, and `app.py`, then deploy.
5. Open the app URL and use **Load sample data** or upload your own CSV/Excel.

## Dataset columns
Required: Date and Sales (or Quantity / Units Sold / Demand).
Optional: Product (or Product ID / Item / SKU).
The app aggregates repeated date/product records and creates features for CatBoost.

## Modules
1. Data Processing and Feature Engineering
2. Model Development and Integration (CatBoost; optional TFT)
3. Demand Prediction and Performance Evaluation (MAE, RMSE, MAPE and forecast CSV)
