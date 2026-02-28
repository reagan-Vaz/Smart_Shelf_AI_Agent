🚀 SmartShelf AI  
Dynamic Pricing Intelligence Platform

SmartShelf AI is an AI-powered dynamic pricing and demand forecasting system designed to optimize inventory movement and maximize profitability using Machine Learning.

📌 Overview

SmartShelf AI predicts product demand based on multiple business factors such as:

- Inventory Level
- Discount Percentage
- Product Category
- Region
- Weather Condition
- Seasonality
- Promotion Status

It then provides:

- 📈 Demand Forecast
- 💰 Revenue Estimation
- 📊 Profit Calculation
- 🎯 Strategic Discount Recommendation
- ⚠ Inventory Risk Assessment.

🧠 Machine Learning Model

The system uses:

- **Random Forest Regressor**
- Feature Importance Analysis
- Train-Test Split Validation
- Performance Metrics:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² Score

The trained model is saved using **Joblib**.

🖥 Tech Stack

- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Plotly
- Joblib

📂 Project Structure

Smart_Shelf_AI_Agent/
│
├── app.py
├── src/
│ ├── train_model.py
│ ├── preprocess.py
│ ├── pricing_agent.py
│ └── init.py
│
├── data/
│ └── sales_data.csv
│
└── models/

▶ How to Run Locally

1️⃣ Install Dependencies - pip install -r requirements.txt

2️⃣ Train Model - python src/train_model.py

3️⃣ Run Streamlit App - streamlit run app.py


🎯 Business Impact

SmartShelf AI enables retailers to:

Reduce overstock risk

Optimize discount strategies

Improve sell-through rate

Increase profitability

Make data-driven pricing decisions

👨‍💻 Author

Reagan Vaz
AI & Data Science Enthusiast

⭐ If you found this project useful, consider giving it a star!
