import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =============================================================================
# STREAMLIT CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="ShipmentSure Predictor",
    page_icon="📦",
    layout="wide"
)


# =============================================================================
# LOAD MODEL
# =============================================================================

@st.cache_resource
def load_assets():
    try:
        pipeline = joblib.load("shipment_xgboost_pipeline.pkl")
        threshold = joblib.load("decision_threshold.pkl")

        return pipeline, threshold

    except FileNotFoundError:
        st.error(
            """
            ❌ Required model files are missing.

            Make sure these files are present in the project folder:

            • shipment_xgboost_pipeline.pkl
            • decision_threshold.pkl
            """
        )
        st.stop()

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()


pipeline, threshold = load_assets()


# =============================================================================
# PREDICTION FUNCTION
# =============================================================================

def predict_shipment(data):
    try:
        # Feature engineering
        data = data.copy()

        data["Cost_to_Weight_Ratio"] = (
            data["Cost_of_the_Product"]
            / data["Weight_in_gms"].replace(0, np.nan)
        )

        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Complete pipeline handles:
        # Imputation → Encoding → Scaling → XGBoost
        probabilities = pipeline.predict_proba(data)[0]

        probability_not_on_time = probabilities[0]
        probability_on_time = probabilities[1]

        # Class 1 = On Time
        prediction = (
            1 if probability_on_time >= threshold else 0
        )

        return (
            prediction,
            probability_on_time,
            probability_not_on_time
        )

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None, None, None


# =============================================================================
# HEADER
# =============================================================================

st.title("📦 ShipmentSure: On-Time Delivery Predictor")

st.markdown(
    """
    ### 🚚 AI-Powered Shipment Delivery Prediction

    Predict whether an e-commerce shipment is likely to arrive
    **On Time** or **Not On Time** using an XGBoost machine learning model.
    """
)

st.write("---")


# =============================================================================
# TABS
# =============================================================================

tab1, tab2 = st.tabs(
    ["🚀 Predictor", "🧠 Model Information"]
)


# =============================================================================
# TAB 1 — PREDICTOR
# =============================================================================

with tab1:

    st.subheader(
        f"Decision Threshold: **{threshold:.2f}**"
    )

    st.info(
        "The threshold was selected using precision-recall analysis "
        "on the validation dataset."
    )

    st.write("---")

    col1, col2 = st.columns(2)


    # -------------------------------------------------------------------------
    # SHIPMENT DETAILS
    # -------------------------------------------------------------------------

    with col1:

        st.header("📦 Shipment Details")

        warehouse_block = st.selectbox(
            "Warehouse Block",
            ["A", "B", "C", "D", "E", "F"]
        )

        mode_of_shipment = st.selectbox(
            "Mode of Shipment",
            ["Flight", "Road", "Ship"]
        )

        product_importance = st.selectbox(
            "Product Importance",
            ["low", "medium", "high"]
        )

        gender = st.selectbox(
            "Customer Gender",
            ["F", "M"]
        )


    # -------------------------------------------------------------------------
    # CUSTOMER & PRODUCT DETAILS
    # -------------------------------------------------------------------------

    with col2:

        st.header("📊 Customer & Product Metrics")

        cost_of_the_product = st.number_input(
            "Cost of the Product ($)",
            min_value=10,
            max_value=500,
            value=200
        )

        weight_in_gms = st.number_input(
            "Weight (grams)",
            min_value=100,
            max_value=8000,
            value=4000,
            step=100
        )

        customer_care_calls = st.slider(
            "Customer Care Calls",
            min_value=1,
            max_value=7,
            value=3
        )

        customer_rating = st.slider(
            "Customer Rating",
            min_value=1,
            max_value=5,
            value=3
        )

        prior_purchases = st.slider(
            "Prior Purchases",
            min_value=1,
            max_value=10,
            value=3
        )

        discount_offered = st.slider(
            "Discount Offered (%)",
            min_value=0,
            max_value=65,
            value=10
        )


    # -------------------------------------------------------------------------
    # PREDICTION BUTTON
    # -------------------------------------------------------------------------

    if st.button(
        "🔮 Predict Delivery Status",
        use_container_width=True
    ):

        input_data = pd.DataFrame({
            "Warehouse_block": [warehouse_block],
            "Mode_of_Shipment": [mode_of_shipment],
            "Customer_care_calls": [customer_care_calls],
            "Customer_rating": [customer_rating],
            "Cost_of_the_Product": [cost_of_the_product],
            "Prior_purchases": [prior_purchases],
            "Product_importance": [product_importance],
            "Gender": [gender],
            "Discount_offered": [discount_offered],
            "Weight_in_gms": [weight_in_gms]
        })


        prediction, prob_on_time, prob_not_on_time = predict_shipment(
            input_data
        )


        # ---------------------------------------------------------------------
        # RESULT
        # ---------------------------------------------------------------------

        st.write("---")
        st.header("🎯 Prediction Result")

        if prediction is not None:

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    "Probability of On-Time Delivery",
                    f"{prob_on_time * 100:.2f}%"
                )

            with result_col2:
                st.metric(
                    "Probability of Delay",
                    f"{prob_not_on_time * 100:.2f}%"
                )


            if prediction == 1:

                st.success(
                    "✅ Shipment Predicted: **ON TIME**"
                )

                st.info(
                    "The model predicts that this shipment is likely "
                    "to reach its destination on schedule."
                )

            else:

                st.warning(
                    "⚠️ Shipment Predicted: **NOT ON TIME**"
                )

                st.error(
                    "The model predicts a potential delivery delay. "
                    "Review the shipment logistics."
                )


# =============================================================================
# TAB 2 — MODEL INFORMATION
# =============================================================================

with tab2:

    st.header("🧠 Model Information")

    st.markdown(
        """
        ShipmentSure uses an **XGBoost Classifier** for binary
        classification.

        **Target Classes:**

        - `0` → Not On Time
        - `1` → On Time
        """
    )


    # -------------------------------------------------------------------------
    # PIPELINE
    # -------------------------------------------------------------------------

    st.subheader("🔄 Machine Learning Pipeline")

    st.code(
        """
Raw Input
   ↓
Feature Engineering
   ↓
Missing Value Handling
   ↓
Categorical Encoding
   ↓
Standard Scaling
   ↓
XGBoost
   ↓
Probability
   ↓
Optimized Threshold
   ↓
On Time / Not On Time
        """,
        language="text"
    )


    # -------------------------------------------------------------------------
    # FEATURES
    # -------------------------------------------------------------------------

    st.subheader("📋 Input Features")

    st.markdown(
        """
        - Warehouse Block
        - Mode of Shipment
        - Customer Care Calls
        - Customer Rating
        - Cost of the Product
        - Prior Purchases
        - Product Importance
        - Gender
        - Discount Offered
        - Weight in Grams
        - Cost-to-Weight Ratio
        """
    )


    # -------------------------------------------------------------------------
    # TECHNOLOGIES
    # -------------------------------------------------------------------------

    st.subheader("🛠️ Technologies")

    st.markdown(
        """
        **Python** • **Pandas** • **NumPy** • **Scikit-learn**  
        **XGBoost** • **SMOTE** • **Joblib** • **Streamlit**
        """
    )


    # -------------------------------------------------------------------------
    # DEPLOYMENT
    # -------------------------------------------------------------------------

    st.subheader("💾 Saved Model")

    st.code(
        """
shipment_xgboost_pipeline.pkl
decision_threshold.pkl
        """,
        language="text"
    )

    st.success(
        "✅ The same preprocessing pipeline used during training "
        "is used during prediction."
    )