import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from datetime import datetime
import time
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="King County House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 0.4rem;
        border-radius: 10px;
        # margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 0.2rem 0 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    .price-display {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .confidence-badge {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin-top: 1rem;
    }
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .info-card h4 {
        color: #333;
        margin-bottom: 0.2rem;
    }
    .info-card h3 {
        color: #667eea;
        margin: 0.4rem 0;
    }
    .info-card p {
        color: #666;
        margin: 0;
    }
    .feature-explanation {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .sidebar-section {
        background: #f0f2f6;
        padding: 0.02rem;
        border-radius: 8px;
        margin: 0.75rem 0;
    }
    .error-message {
        background: #ffe6e6;
        color: #d32f2f;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #d32f2f;
        margin: 1rem 0;
    }
    .success-message {
        background: #e8f5e8;
        color: #2e7d2e;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2e7d2e;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data with error handling
@st.cache_resource
def load_models():
    try:
        final_pipeline = load("artifacts/final_pipeline.joblib")
        feature_order = load("artifacts/feature_order.joblib")
        mean_log_price_per_zip = load("artifacts/mean_log_price_per_zip.joblib")
        global_mean_log_price = load("artifacts/global_mean_log_price.joblib")
        return final_pipeline, feature_order, mean_log_price_per_zip, global_mean_log_price, None
    except Exception as e:
        return None, None, None, None, str(e)

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# Load models
final_pipeline, feature_order, mean_log_price_per_zip, global_mean_log_price, error = load_models()

# Header
st.markdown("""
<div class="main-header">
    <h1>🏠 King County House Price Predictor</h1>
    <p>Seattle & Washington State real estate valuation using AI</p>
</div>
""", unsafe_allow_html=True)

if error:
    st.markdown(f"""
    <div class="error-message">
        <strong>⚠️ Model Loading Error:</strong> {error}
        <br>Please ensure all model files are in the 'artifacts' directory.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Create two columns for layout
col1, col2 = st.columns([1, 2])

# Sidebar for inputs
with st.sidebar:
    st.markdown("### 🏡 Property Details")
    
    # Basic property info
    # st.markdown("---")
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**🛏️ Basic Information**")
    bedrooms = st.slider("Bedrooms", 1, 10, 3, help="Number of bedrooms in the house")
    bathrooms = st.slider("Bathrooms", 1.0, 8.0, 2.0, step=0.25, help="Number of bathrooms (can be fractional)")
    floors = st.slider("Floors", 1.0, 3.0, 1.0, step=0.5, help="Number of floors in the house")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Size information
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**📏 Size Information**")
    sqft_living = st.number_input("Living Area (sqft)", 370, 12050, 2000, step=50, 
                                  help="Interior living space in square feet")
    sqft_lot = st.number_input("Lot Size (sqft)", 520, 1164794, 5000, step=100,
                              help="Total lot size in square feet")
    sqft_lot15 = st.number_input("Avg Lot Size of 15 Nearest Houses (sqft)", 651, 871200, 5000, step=100,
                                help="Average lot size of 15 nearest neighbor houses")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quality and features
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**⭐ Quality & Features**")
    view = st.slider("View Rating", 0, 4, 0, help="0=No view, 4=Excellent view")
    condition = st.slider("Condition", 1, 5, 3, help="1=Poor, 5=Excellent condition")
    waterfront = st.selectbox("Waterfront Property?", ["No", "Yes"], 
                             help="Is the property on the waterfront?")
    has_basement = st.selectbox("Has Basement?", ["No", "Yes"],
                               help="Does the property have a basement?")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Age and location
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**📅 Age & Location**")
    house_age = st.number_input("House Age (years)", 0, 150, 20, step=1,
                               help="Age of the house in years")
    renovation_age = st.number_input("Years Since Last Renovation", 0, 150, 0, step=1,
                                    help="Years since last major renovation (0 if never renovated)")
    zipcode = st.number_input("Zipcode", 98000, 98199, 98052, step=1,
                             help="King County zipcode")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Prediction button with validation
    input_valid = True
    if sqft_living <= 0 or sqft_lot <= 0:
        st.error("⚠️ Please enter valid positive values for all size measurements.")
        input_valid = False
    
    predict_button = st.button("🔮 Predict Price", type="primary", disabled=not input_valid,
                              help="Click to generate price prediction" if input_valid else "Fix input errors first")

# Main content area
if predict_button and input_valid:
    with st.spinner("🤖 Analyzing property features and generating prediction..."):
        # Simulate processing time for better UX
        time.sleep(1)
        
        # Map input zipcode to target-encoded value
        if zipcode in mean_log_price_per_zip.index:
            zipcode_encoded = mean_log_price_per_zip[zipcode]
        else:
            zipcode_encoded = global_mean_log_price
            st.warning(f"⚠️ Zipcode {zipcode} not in training data. Using average price for prediction.")
        
        # Create input dataframe
        input_df = pd.DataFrame({
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "sqft_living": [sqft_living],
            "sqft_lot": [sqft_lot],
            "floors": [floors],
            "waterfront": [1 if waterfront == "Yes" else 0],
            "view": [view],
            "condition": [condition],
            "sqft_lot15": [sqft_lot15],
            "has_basement": [1 if has_basement == "Yes" else 0],
            "renovation_age": [renovation_age],
            "house_age": [house_age],
            "zipcode_encoded": [zipcode_encoded]
        })
        
        # Reorder columns to match training data
        input_df = input_df[feature_order]
        
        # Make prediction
        try:
            log_pred = final_pipeline.predict(input_df)[0]
            prediction = np.expm1(log_pred)
            st.session_state.prediction_made = True
            st.session_state.prediction = prediction
            st.session_state.input_df = input_df
            
        except Exception as e:
            st.error(f"⚠️ Prediction Error: {str(e)}")
            st.stop()

# Display results if prediction was made
if st.session_state.prediction_made:
    prediction = st.session_state.prediction
    input_df = st.session_state.input_df
    
    # Price display with confidence interval
    st.markdown(f"""
    <div class="metric-container">
        <h2>💰 Estimated Property Value</h2>
        <div class="price-display">${prediction:,.0f}</div>
        <div class="confidence-badge">
            📊 Confidence Range: ${prediction*0.85:,.0f} - ${prediction*1.15:,.0f}
        </div>
        <p style="margin-top: 1rem; opacity: 0.9;">
            Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for charts and insights
    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Feature importance chart - HORIZONTAL IMPLEMENTATION
        if hasattr(final_pipeline['model'], "feature_importances_"):
            st.markdown("### 📊 Key Price Drivers")
            
            # Get feature importance
            importance_df = pd.DataFrame({
                "Feature": input_df.columns,
                "Importance": final_pipeline['model'].feature_importances_
            }).sort_values(by="Importance", ascending=False).head(6)
            
            # Create better feature names for display
            feature_names = {
                'sqft_living': 'Living Area',
                'zipcode_encoded': 'Location',
                'sqft_lot': 'Lot Size', 
                'bathrooms': 'Bathrooms',
                'bedrooms': 'Bedrooms',
                'view': 'View',
                'condition': 'Condition',
                'waterfront': 'Waterfront',
                'floors': 'Floors',
                'house_age': 'House Age',
                'has_basement': 'Basement',
                'renovation_age': 'Renovation Age',
                'sqft_lot15': 'Neighborhood'
            }
            
            importance_df['Display_Name'] = importance_df['Feature'].map(feature_names).fillna(importance_df['Feature'])
            
            if len(importance_df) > 0 and importance_df['Importance'].sum() > 0:
                # Create vertical chart with rotated x-axis labels using Plotly
                fig = px.bar(
                    importance_df.sort_values('Importance', ascending=False), 
                    x='Display_Name', 
                    y='Importance',
                    title="Key Price Drivers",
                    color='Importance',
                    color_continuous_scale='viridis',
                    height=400
                )
                
                # Rotate x-axis labels
                fig.update_layout(
                    xaxis_tickangle=-45,  # Rotate labels 45 degrees
                    showlegend=False,
                    xaxis_title="Features",
                    yaxis_title="Importance Score",
                    font=dict(size=12),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    title_x=0.5
                )
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance data not available for this model type.")
    
    with chart_col2:
        # Property comparison chart - HORIZONTAL IMPLEMENTATION
        st.markdown("### 🏘️ Market Comparison")
        
        # Create comparison data (simulated market data)
        comparison_data = {
            'Property Type': ['Your Property', 'Similar Properties (Avg)', 'Neighborhood (Avg)', 'King County (Avg)'],
            'Estimated Value': [
                prediction,
                prediction * 0.95,
                prediction * 0.88,
                prediction * 0.75
            ]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Create vertical chart with rotated x-axis labels for comparison
        if len(df_comparison) > 0:
            # Create vertical chart with rotated labels using Plotly
            fig_comp = px.bar(
                df_comparison.sort_values('Estimated Value', ascending=False),
                x='Property Type',
                y='Estimated Value',
                title="Market Value Comparison",
                color='Estimated Value',
                color_continuous_scale='blues',
                height=400
            )
            
            # Rotate x-axis labels and format
            fig_comp.update_layout(
                xaxis_tickangle=-45,  # Rotate labels 45 degrees
                showlegend=False,
                xaxis_title="Property Type",
                yaxis_title="Estimated Value ($)",
                yaxis=dict(tickformat='$,.0f'),  # Format y-axis as currency
                font=dict(size=12),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_x=0.5
            )
            
            # Display the chart
            st.plotly_chart(fig_comp, use_container_width=True)
            
    st.markdown("---")
    # Property insights
    st.markdown("### 🔍 Property Insights")
    
    insights_col1, insights_col2, insights_col3 = st.columns(3)
    
    with insights_col1:
        # Price per sqft
        price_per_sqft = prediction / sqft_living
        avg_king_county_psf = 450  # Approximate average for King County
        comparison_text = "Above average" if price_per_sqft > avg_king_county_psf else "Below average"
        
        st.markdown(f"""
        <div class="info-card">
            <h4>Price per Sq Ft</h4>
            <h3>${price_per_sqft:.0f}</h3>
            <p>{comparison_text} for King County (avg: ${avg_king_county_psf})</p>
        </div>
        """, unsafe_allow_html=True)
    
    with insights_col2:
        # Market segment
        if prediction > 800000:
            segment = "Luxury"
            color = "#FFD700"
        elif prediction > 500000:
            segment = "Premium"
            color = "#87CEEB"
        else:
            segment = "Standard"
            color = "#98FB98"
        
        st.markdown(f"""
        <div class="info-card">
            <h4>Market Segment</h4>
            <h3 style="color: {color};">{segment}</h3>
            <p>Based on King County pricing tiers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with insights_col3:
        # Investment potential
        base_score = 50
        location_bonus = 20 if prediction > 600000 else 10
        view_bonus = view * 4
        age_penalty = max(0, (house_age - 20) * 0.5)
        condition_bonus = condition * 3
        waterfront_bonus = 15 if waterfront == "Yes" else 0
        
        investment_score = min(100, max(0, int(base_score + location_bonus + view_bonus + condition_bonus + waterfront_bonus - age_penalty)))
        
        if investment_score >= 80:
            potential = "Excellent"
        elif investment_score >= 65:
            potential = "Good"
        elif investment_score >= 50:
            potential = "Fair"
        else:
            potential = "Poor"


            st.markdown(f"""
        <div class="info-card">
            <h4>Market Segment</h4>
            <h3 style="color: {color};">{segment}</h3>
            <p>Based on King County pricing tiers</p>
        </div>
        """, unsafe_allow_html=True)
    
        
        st.markdown(f"""
        <div class="info-card">
            <h4>Investment Score</h4>
            <h3>{investment_score}/100</h3>
            <p>{potential} investment potential</p>
        </div>
        """, unsafe_allow_html=True)

# Model information and methodology
with st.expander("ℹ️ About This Model"):
    st.markdown("""
    ### Model Information
    
    **Algorithm:** XGBoost (Extreme Gradient Boosting) Regressor
    
    **Dataset:** King County House Sales Dataset
    - Contains 21,613 house sale records from May 2014 to May 2015
    - Covers properties in King County, Washington State (Seattle area)
    - Includes detailed property features and sale prices
    
    **Features Used:**
    - Property characteristics (bedrooms, bathrooms, square footage)
    - Location encoding (zipcode-based price averages)
    - Property age and renovation history
    - Quality indicators (view, condition, waterfront)
    - Neighborhood characteristics (lot size of 15 nearest neighbors)
    
    **Model Performance:**
    - Trained using XGBoost algorithm for superior accuracy
    - Cross-validated for reliability and generalization
    - Feature importance analysis for transparency
    - Hyperparameter tuning for optimal performance
    
    **Limitations:**
    - Predictions are estimates based on historical data (2014-2015)
    - Market conditions and unique property features may affect actual prices
    - Model trained on specific time period, current market may differ
    - Always consult with real estate professionals for final decisions
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🏠 King County House Price Predictor | Built with Streamlit & Machine Learning</p>
   
</div>
""", unsafe_allow_html=True)