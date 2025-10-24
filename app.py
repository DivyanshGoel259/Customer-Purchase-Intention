import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Set page configuration
st.set_page_config(
    page_title="Customer Purchase Intention Predictor",
    page_icon="🛒",
    layout="wide"
)

# Load the trained model and preprocessing objects
@st.cache_resource
def load_models():
    try:
        model = load_model('model.h5')
        with open('label_encoder.pkl', 'rb') as file:
            label_encoder = pickle.load(file)
        with open('onehot_encoder.pkl', 'rb') as file:
            onehot_encoder = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model, label_encoder, onehot_encoder, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

model, label_encoder, onehot_encoder, scaler = load_models()

# Title and description
st.title("🛒 Customer Purchase Intention Predictor")
st.markdown("""
This application predicts whether a customer will make a purchase based on their browsing behavior.
Fill in the customer's session details below to get a prediction.
""")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Page Visit Information")
    
    administrative = st.number_input(
        "Administrative Pages Visited",
        min_value=0,
        max_value=50,
        value=0,
        help="Number of administrative pages visited"
    )
    
    administrative_duration = st.number_input(
        "Administrative Duration (seconds)",
        min_value=0.0,
        max_value=5000.0,
        value=0.0,
        help="Time spent on administrative pages"
    )
    
    informational = st.number_input(
        "Informational Pages Visited",
        min_value=0,
        max_value=50,
        value=0,
        help="Number of informational pages visited"
    )
    
    informational_duration = st.number_input(
        "Informational Duration (seconds)",
        min_value=0.0,
        max_value=5000.0,
        value=0.0,
        help="Time spent on informational pages"
    )
    
    product_related = st.number_input(
        "Product Related Pages Visited",
        min_value=0,
        max_value=200,
        value=1,
        help="Number of product-related pages visited"
    )
    
    product_related_duration = st.number_input(
        "Product Related Duration (seconds)",
        min_value=0.0,
        max_value=10000.0,
        value=0.0,
        help="Time spent on product-related pages"
    )

with col2:
    st.subheader("📈 Engagement Metrics")
    
    bounce_rates = st.slider(
        "Bounce Rate",
        min_value=0.0,
        max_value=0.5,
        value=0.0,
        step=0.01,
        help="Percentage of visitors who leave immediately"
    )
    
    exit_rates = st.slider(
        "Exit Rate",
        min_value=0.0,
        max_value=0.5,
        value=0.0,
        step=0.01,
        help="Percentage of page views that were the last in the session"
    )
    
    page_values = st.number_input(
        "Page Values",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        help="Average value of pages visited by the user"
    )
    
    special_day = st.slider(
        "Special Day Proximity",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Closeness to special days (0 = not close, 1 = very close)"
    )
    
    region = st.number_input(
        "Region",
        min_value=1,
        max_value=9,
        value=1,
        help="Geographic region (1-9)"
    )
    
    traffic_type = st.number_input(
        "Traffic Type",
        min_value=1,
        max_value=20,
        value=1,
        help="Type of traffic source"
    )

# Additional features in a third section
st.subheader("🗓️ Visit Context")
col3, col4, col5 = st.columns(3)

with col3:
    month = st.selectbox(
        "Month",
        options=['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        help="Month of the visit"
    )

with col4:
    visitor_type = st.selectbox(
        "Visitor Type",
        options=['Returning_Visitor', 'New_Visitor', 'Other'],
        help="Type of visitor"
    )

with col5:
    weekend = st.selectbox(
        "Weekend Visit",
        options=['False', 'True'],
        help="Was the visit on a weekend?"
    )

# Prediction button
st.markdown("---")
if st.button("🔮 Predict Purchase Intention", type="primary", use_container_width=True):
    if model is None:
        st.error("Model not loaded. Please ensure all model files are present.")
    else:
        try:
            # Create input dataframe
            input_data = pd.DataFrame({
                'Administrative': [administrative],
                'Administrative_Duration': [administrative_duration],
                'Informational': [informational],
                'Informational_Duration': [informational_duration],
                'ProductRelated': [product_related],
                'ProductRelated_Duration': [product_related_duration],
                'BounceRates': [bounce_rates],
                'ExitRates': [exit_rates],
                'PageValues': [page_values],
                'SpecialDay': [special_day],
                'Region': [region],
                'TrafficType': [traffic_type],
                'VisitorType': [visitor_type],
                'Weekend': [weekend],
                'Month': [month]
            })
            
            # Encode VisitorType and Weekend
            input_data['VisitorType'] = label_encoder.transform(input_data['VisitorType'])
            input_data['Weekend'] = label_encoder.transform(input_data['Weekend'])
            
            # One-hot encode Month
            month_encoded = onehot_encoder.transform(input_data[['Month']])
            month_encoded_df = pd.DataFrame(
                month_encoded.toarray(),
                columns=onehot_encoder.get_feature_names_out(['Month'])
            )
            
            # Combine features
            input_data = pd.concat([
                input_data.drop(['Month'], axis=1),
                month_encoded_df
            ], axis=1)
            
            # Scale the features
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            prediction_proba = prediction[0][0]
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                if prediction_proba > 0.5:
                    st.success("✅ **HIGH Purchase Likelihood**")
                    st.metric("Purchase Probability", f"{prediction_proba*100:.2f}%")
                else:
                    st.warning("❌ **LOW Purchase Likelihood**")
                    st.metric("Purchase Probability", f"{prediction_proba*100:.2f}%")
            
            with col_result2:
                st.info("**Recommendation**")
                if prediction_proba > 0.7:
                    st.write("🎯 Strong buyer intent! Consider offering a special deal.")
                elif prediction_proba > 0.5:
                    st.write("🤔 Moderate interest. Send targeted email or retargeting ads.")
                elif prediction_proba > 0.3:
                    st.write("💡 Low intent. Consider offering free shipping or discounts.")
                else:
                    st.write("📧 Very low intent. Add to remarketing list for future campaigns.")
            
            # Progress bar for visualization
            st.progress(float(prediction_proba))
            
            # Additional insights
            with st.expander("📈 View Input Summary"):
                st.dataframe(input_data)
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.write("Please check your input values and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit | Customer Purchase Intention Prediction Model</p>
</div>
""", unsafe_allow_html=True)

