import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="SmartShelf AI",
    page_icon="🚀",
    layout="wide"
)

# --------------------------------------------------
# Custom Styling (SaaS Look)
# --------------------------------------------------

st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: 700;
}
.subtext {
    font-size: 16px;
    color: #9ca3af;
}
.kpi-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
.kpi-value {
    font-size: 28px;
    font-weight: bold;
}
.section-header {
    font-size: 22px;
    font-weight: 600;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------

model_data = joblib.load("models/demand_model.pkl")
model = model_data["model"]
feature_names = model_data["features"]

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------

st.sidebar.title("Pricing Inputs")

inventory = st.sidebar.number_input("Inventory Level", min_value=0, value=100)
manual_discount = st.sidebar.number_input("Manual Discount (%)", min_value=0, max_value=100, value=20)
price = st.sidebar.number_input("Product Base Price (₹)", min_value=1.0, value=100.0)
promotion = st.sidebar.selectbox("Promotion Running?", [0, 1])
category = st.sidebar.selectbox("Category", ["Electronics", "Clothing", "Groceries"])
region = st.sidebar.selectbox("Region", ["North", "South", "East", "West"])
weather = st.sidebar.selectbox("Weather Condition", ["Sunny", "Rainy", "Cloudy"])
season = st.sidebar.selectbox("Seasonality", ["Summer", "Winter", "Monsoon"])

run = st.sidebar.button("Run AI Prediction")

# --------------------------------------------------
# Header Section
# --------------------------------------------------

st.markdown('<div class="big-title">SmartShelf AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Dynamic Pricing Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">AI-powered demand forecasting and intelligent discount optimization engine.</div>', unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------

if run:

    # Create input dataframe
    input_data = {
        "Inventory_Level": inventory,
        "Discount_Percent": manual_discount,
        "Price": price,
        "Promotion": promotion,
        "Category": category,
        "Region": region,
        "Weather": weather,
        "Season": season
    }

    input_df = pd.DataFrame([input_data])

    # Ensure correct feature order
    input_df = input_df.reindex(columns=feature_names)

    # Prediction
    predicted_demand = model.predict(input_df)[0]

    # Business calculations
    recommended_discount = min(50, manual_discount + 15)
    final_price = price * (1 - recommended_discount / 100)
    revenue = predicted_demand * final_price
    profit = revenue * 0.2
    sell_through = (predicted_demand / inventory) * 100 if inventory > 0 else 0

    # --------------------------------------------------
    # KPI Section
    # --------------------------------------------------

    st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.markdown(f'<div class="kpi-card"><div>Predicted Demand</div><div class="kpi-value">{predicted_demand:.2f}</div></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="kpi-card"><div>Recommended Discount</div><div class="kpi-value">{recommended_discount}%</div></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="kpi-card"><div>Sell-Through</div><div class="kpi-value">{sell_through:.2f}%</div></div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="kpi-card"><div>Revenue</div><div class="kpi-value">₹ {revenue:.2f}</div></div>', unsafe_allow_html=True)
    col5.markdown(f'<div class="kpi-card"><div>Profit</div><div class="kpi-value">₹ {profit:.2f}</div></div>', unsafe_allow_html=True)

    st.divider()

    # --------------------------------------------------
    # Demand Visualization
    # --------------------------------------------------

    st.markdown('<div class="section-header">Demand Forecast Visualization</div>', unsafe_allow_html=True)

    fig = px.bar(
        x=["Forecasted Demand"],
        y=[predicted_demand],
        text=[f"{predicted_demand:.2f} Units"],
        template="plotly_dark"
    )

    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="",
        yaxis_title="Units Sold"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --------------------------------------------------
    # Risk & Strategy
    # --------------------------------------------------

    colA, colB = st.columns(2)

    with colA:
        st.markdown('<div class="section-header">Inventory Risk</div>', unsafe_allow_html=True)

        if sell_through < 20:
            st.error("High leftover stock risk. Immediate action required.")
        elif sell_through < 50:
            st.warning("Moderate stock risk. Consider discount optimization.")
        else:
            st.success("Healthy stock movement expected.")

    with colB:
        st.markdown('<div class="section-header">Strategic Insight</div>', unsafe_allow_html=True)

        if sell_through < 20:
            st.info("Heavy discount strategy recommended to improve stock rotation.")
        elif sell_through < 50:
            st.info("Balanced discounting can improve revenue while maintaining margins.")
        else:
            st.info("Maintain pricing strategy to maximize profitability.")