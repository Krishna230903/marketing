import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="SpiceJet Passenger Analysis",
    page_icon="✈️",
    layout="wide"
)

# --- Main Function to Run the Analysis ---
def run_analysis():
    """
    This function loads data directly, performs analysis, and displays results in Streamlit.
    """
    st.header("✈️ SpiceJet Passenger Traffic Analysis")
    st.write("""
    This web app performs a regression analysis to understand the factors influencing the number of passengers carried by SpiceJet.
    The analysis includes data cleaning, descriptive statistics, correlation analysis, and a predictive linear regression model.
    """)

    try:
        # --- 1. Load the Data Automatically ---
        # The app now loads the data file directly from the repository.
        # Ensure 'spicejet24 (1).xlsx - Monthly_Template.csv' is in your GitHub repo.
        df = pd.read_csv('spicejet24 (1).xlsx - Monthly_Template.csv', skiprows=2)
        st.success("Dataset loaded successfully!")

        # --- 2. Data Cleaning and Preprocessing ---
        df.columns = df.columns.str.strip()
        df.columns = [
            'MONTH', 'DEPARTURES', 'HOURS', 'KILOMETRE', 'PASSENGERS_CARRIED',
            'PASSENGER_KMS_PERFORMED', 'AVAILABLE_SEAT_KILOMETRE', 'PAX_LOAD_FACTOR',
            'FREIGHT_TONNE', 'MAIL_TONNE', 'TOTAL_TONNE', 'PASSENGER_TKM',
            'FREIGHT_TKM', 'MAIL_TKM', 'TOTAL_TKM', 'AVAILABLE_TONNE_KILOMETRE',
            'WEIGHT_LOAD_FACTOR'
        ]
        df_cleaned = df.drop(columns=['PAX_LOAD_FACTOR', 'WEIGHT_LOAD_FACTOR', 'MONTH'])
        for col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        df_cleaned.dropna(inplace=True)

        if st.checkbox("Show Cleaned Data"):
            st.subheader("Cleaned & Processed Data")
            st.dataframe(df_cleaned)

        # --- 3. Descriptive Analysis ---
        st.header("📊 Descriptive Analysis")
        st.write("Here is a statistical summary of the operational data:")
        st.dataframe(df_cleaned.describe())

        # --- 4. Correlation Analysis ---
        st.header("🔗 Correlation Analysis")
        st.write("The heatmap below shows the correlation between different variables. This helps us understand the relationships between them.")
        
        correlation_matrix = df_cleaned.corr()
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title('Correlation Matrix of Airline Operational Data')
        st.pyplot(fig)

        # --- 5. Regression Analysis ---
        st.header("📈 Regression Analysis")
        st.write("We build a linear regression model to predict the number of **Passengers Carried** based on other operational factors.")

        y = df_cleaned['PASSENGERS_CARRIED']
        X = df_cleaned.drop(columns=['PASSENGERS_CARRIED'])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # --- 6. Model Evaluation ---
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("R-squared", f"{metrics.r2_score(y_test, y_pred):.4f}")
        col2.metric("RMSE", f"{np.sqrt(metrics.mean_squared_error(y_test, y_pred)):.2f}")
        
        # --- 7. Display Results & Conclusion ---
        st.subheader("Model Coefficients")
        st.write("The coefficients represent the change in the number of passengers for a one-unit change in the corresponding feature.")
        coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
        st.dataframe(coeff_df)
        
        st.info(f"**Model Intercept:** {model.intercept_:.2f}")

        st.header("📝 Conclusion & Next Steps")
        st.write("""
        This analysis provides a foundational regression model. For a more comprehensive model, as outlined in your project document, 
        additional data like fleet size, fuel prices, GDP, and competitor data would be required. The next phase could involve collecting this 
        data to build an even more powerful predictive model, potentially using Neural Networks.
        """)

    except FileNotFoundError:
        st.error("Error: The data file 'spicejet24 (1).xlsx - Monthly_Template.csv' was not found.")
        st.info("Please make sure the CSV file is in the same GitHub repository as the app.py file.")
    except Exception as e:
        st.error(f"An error occurred during the analysis: {e}")

# --- Run the main function ---
if __name__ == "__main__":
    run_analysis()
