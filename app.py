import streamlit as st
import numpy as np
import joblib
import random
import time

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Customer Insight Tool",
    page_icon="🛍️",
    layout="centered"
)

# ---------------- Global Styling ----------------
st.markdown(
    """
    <style>
        body {
            font-family: Inter, sans-serif;
        }
        div[data-testid="stProgress"] > div > div {
            background-color: #22c55e;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Load Model ----------------
kmeans = joblib.load("Kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- Header ----------------
st.markdown(
    """
    <h2 style="text-align:center;">Customer Insight Tool</h2>
    <p style="text-align:center; color: #9ca3af;">
        Smart segmentation without the clutter
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- Session State ----------------
if "income" not in st.session_state:
    st.session_state.income = 50
    st.session_state.spending = 50

# ---------------- Input Section ----------------
st.markdown("## Customer Details")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input(
        "Annual Income (k$)",
        min_value=5,
        max_value=200,
        value=st.session_state.income,
        step=1
    )

with col2:
    spending = st.number_input(
        "Spending Score (1–100)",
        min_value=1,
        max_value=100,
        value=st.session_state.spending,
        step=1
    )

colA, colB = st.columns(2)

with colA:
    if st.button("Generate Sample Customer"):
        st.session_state.income = random.randint(15, 150)
        st.session_state.spending = random.randint(10, 95)
        st.rerun()

with colB:
    analyze = st.button("Analyze Customer", type="primary")

# ---------------- Personas ----------------
personas = {
    0: ("Budget Watcher", "Careful spender who prioritizes essentials and discounts."),
    1: ("Premium Shopper", "High income and high spending. Values quality and exclusivity."),
    2: ("Regular Customer", "Balanced income and spending with predictable behavior."),
    3: ("Impulse Buyer", "Emotion-driven spender who reacts strongly to promotions."),
    4: ("Careful Wealth", "High income but rational and value-focused in spending.")
}

# ---------------- Analysis Result ----------------
if analyze:
    input_data = np.array([[income, spending]])
    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]

    title, desc = personas[cluster]

    # -------- Value Score --------
    income_score = min(income / 150 * 100, 100)
    value_score = int((income_score + spending) / 2)

    glow_strength = min(0.25 + (value_score / 100) * 0.65, 0.9)

    value_label = (
        "High" if value_score >= 75
        else "Moderate" if value_score >= 40
        else "Low"
    )

    st.divider()

    # ---------------- Result Card ----------------
    st.markdown(
        f"""
        <div style="
            background: #020617;
            padding: 30px;
            border-radius: 20px;
            border-left: 6px solid rgba(34,197,94,{glow_strength});
            box-shadow: 0 0 30px rgba(34,197,94,{glow_strength});
        ">
            <h3 style="margin-bottom: 10px;">{title}</h3>
            <p style="font-size:16px; color:#d1d5db; margin-bottom: 18px;">
                {desc}
            </p>
            <p style="font-weight:600; color: rgba(34,197,94,{glow_strength});">
                Customer Value: {value_label}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------- Animated Value Bar (ALWAYS SLIDES) ----------------
    st.markdown("### Customer Value")

    bar_placeholder = st.empty()
    progress = bar_placeholder.progress(0)

    for i in range(value_score + 1):
        progress.progress(i)
        time.sleep(0.01)

    st.caption(value_label)

    # ---------------- Suggested Action ----------------
    st.markdown("## Suggested Action")
    st.write(
        "Align communication, pricing, and offers with this customer’s "
        "spending behavior to improve engagement and long-term value."
    )

# ---------------- Footer ----------------
st.divider()
st.caption("Minimal • Intuitive • ML-powered")