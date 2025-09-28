import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import io

# Page configuration
st.set_page_config(
    page_title="Multiple Linear Regression Analysis",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Multiple Linear Regression Analysis")
st.markdown("""
**Welcome to the Multiple Linear Regression Analysis Tool!**

This application helps you build predictive models using Multiple Linear Regression, a statistical technique that:
- Predicts a target variable based on multiple input features
- Finds the best-fitting linear relationship between variables
- Helps understand which features most influence your target

**Perfect for:** Sales forecasting, price predictions, risk assessment, and any scenario where you need to understand relationships between variables.
""")

# Initialize session state for data persistence
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'df_after_correlation' not in st.session_state:
    st.session_state.df_after_correlation = None
if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0

# Sidebar for file upload and configuration
st.sidebar.header("📁 Data Configuration")

# File uploader with dynamic key for reset functionality
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload a CSV file with your dataset. The file should contain both your input features (independent variables) and target variable (what you want to predict).",
    key=f"file_uploader_{st.session_state.file_uploader_key}"
)

# Reset button
if st.sidebar.button("🔄 Reset Application", type="secondary",
                     help="Clear all data and reset to initial state. This will remove your uploaded file and all analysis results."):
    # Clear all session state except the file_uploader_key
    file_key = st.session_state.file_uploader_key
    for key in list(st.session_state.keys()):
        if key != 'file_uploader_key':
            del st.session_state[key]
    # Increment the key to reset the file uploader
    st.session_state.file_uploader_key = file_key + 1
    st.rerun()

# Load data
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

        # Display data info
        st.sidebar.success(f"✅ Data loaded successfully!")
        st.sidebar.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    except Exception as e:
        st.sidebar.error(f"Error loading file: {str(e)}")
        st.stop()
else:
    # Display tab functionalities when no data is loaded
    st.info("👈 Please upload a CSV file to begin")

    st.markdown("---")
    st.header("📚 Application Workflow")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 📋 Data Overview
        - **View your dataset** with basic statistics
        - **Inspect data types** and column information
        - **Check missing values** and data quality
        - **Explore statistical summary** of numeric columns

        *Why it matters:* Understanding your data is the first step to building a good model.
        """)

        st.markdown("""
        ### 🔧 Preprocessing
        - **Select features** to keep or drop
        - **Remove categorical columns** automatically
        - **Analyze correlations** between features
        - **View scatter matrix** for feature relationships
        - **Perform additional feature selection** based on correlations

        *Why it matters:* Clean, relevant data leads to better predictions.
        """)

    with col2:
        st.markdown("""
        ### 🎯 Model Training
        - **Choose target variable** (dependent variable)
        - **Select feature variables** (independent variables)
        - **Configure training parameters** (test size, random state)
        - **Apply standardization** if needed
        - **Train the model** and view coefficients
        - **Evaluate performance** with multiple metrics

        *Why it matters:* This creates your predictive model.
        """)

    with col3:
        st.markdown("""
        ### 📈 Visualization
        - **Actual vs Predicted** scatter plots
        - **Residual analysis** for model validation
        - **Feature importance** visualization
        - **3D plots** for two-feature models
        - **Individual feature relationships** with target

        *Why it matters:* Visualizations help you understand and trust your model.
        """)

        st.markdown("""
        ### 🔍 Predictions
        - **Make single predictions** with custom inputs
        - **Batch predictions** from CSV files
        - **Prediction intervals** with confidence bounds
        - **Export results** to CSV format

        *Why it matters:* Use your model for real-world predictions.
        """)

    st.markdown("---")
    st.markdown("*Start by uploading your CSV dataset in the sidebar to access all features.*")
    st.stop()

# Main content area with tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data Overview",
    "🔧 Preprocessing",
    "🎯 Model Training",
    "📈 Visualization",
    "🔍 Predictions"
])

# Tab 1: Data Overview
with tab1:
    st.header("Data Overview")
    st.markdown("""
    This section helps you understand your dataset before building a model. Look for:
    - **Data quality issues** (missing values, incorrect types)
    - **Statistical patterns** (means, ranges, distributions)
    - **Potential problems** (outliers, imbalanced data)
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("First 10 Rows")
        with st.expander("ℹ️ Why review sample data?"):
            st.write(
                "Checking sample rows helps you verify the data loaded correctly and understand what each column contains.")
        st.dataframe(st.session_state.df.head(10))

        st.subheader("Data Types")
        with st.expander("ℹ️ Understanding data types"):
            st.write("""
            - **Numeric columns** (int64, float64): Can be used for regression
            - **Text columns** (object): Usually need to be removed or encoded
            - **Non-Null Count**: How many valid values in each column
            - **Null Count**: Missing values that may need handling
            """)
        st.dataframe(pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Type': st.session_state.df.dtypes.astype(str).values,
            # Convert to string to avoid Arrow serialization issues
            'Non-Null Count': st.session_state.df.count().values,
            'Null Count': st.session_state.df.isnull().sum().values
        }))

    with col2:
        st.subheader("Statistical Summary")
        with st.expander("ℹ️ How to interpret statistics"):
            st.write("""
            - **count**: Number of non-missing values
            - **mean**: Average value (check if reasonable)
            - **std**: Standard deviation (higher = more spread)
            - **min/max**: Range of values (check for outliers)
            - **25%, 50%, 75%**: Quartiles showing data distribution
            """)
        st.dataframe(st.session_state.df.describe().round(2))

        st.subheader("Missing Values")
        missing = st.session_state.df.isnull().sum()
        if missing.sum() > 0:
            st.warning("⚠️ Missing values detected! Consider handling them before training.")
            fig, ax = plt.subplots(figsize=(8, 4))
            missing[missing > 0].plot(kind='bar', ax=ax)
            ax.set_ylabel("Count")
            ax.set_title("Missing Values by Column")
            st.pyplot(fig)
        else:
            st.success("✅ No missing values found!")

# Tab 2: Preprocessing
with tab2:
    st.header("Data Preprocessing")
    st.markdown("""
    Prepare your data for modeling by removing irrelevant columns and understanding feature relationships.
    Good preprocessing is crucial for model performance.
    """)

    # Column selection
    st.subheader("1. Initial Feature Selection")
    with st.expander("ℹ️ Which columns to remove?"):
        st.write("""
        **Remove these types of columns:**
        - Text columns (names, descriptions, categories)
        - ID columns (customer ID, order ID, etc.)
        - Dates (unless converted to numeric)
        - Any column that won't help predict your target

        **Keep numeric columns that might influence your target variable.**
        """)

    col1, col2 = st.columns(2)

    with col1:
        # Get numeric columns
        numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = st.session_state.df.select_dtypes(include=['object']).columns.tolist()

        st.write("**Numeric Columns:**", numeric_cols)
        st.write("**Categorical Columns:**", categorical_cols)

    with col2:
        # Drop columns
        columns_to_drop = st.multiselect(
            "Select columns to drop (categorical or unwanted):",
            options=st.session_state.df.columns.tolist(),
            default=categorical_cols,
            help="Select all columns you don't want to use in your model. Categorical columns are pre-selected."
        )

    # Apply preprocessing
    if st.button("Apply Preprocessing"):
        df_processed = st.session_state.df.drop(columns=columns_to_drop, errors='ignore')
        st.session_state.df_processed = df_processed
        st.session_state.df_after_correlation = df_processed.copy()  # Initialize for later use
        st.success("Preprocessing applied!")

    # Show correlation matrix and scatter plots if processed data exists
    if st.session_state.df_processed is not None:
        st.subheader("2. Correlation Analysis")

        # Correlation Matrix
        st.markdown("##### Correlation Matrix")
        with st.expander("ℹ️ How to read the correlation matrix"):
            st.write("""
            **Values range from -1 to +1:**
            - **+1**: Perfect positive correlation (as one increases, other increases)
            - **0**: No linear relationship
            - **-1**: Perfect negative correlation (as one increases, other decreases)

            **Look for:**
            - Features highly correlated with your target (good predictors)
            - Features highly correlated with each other (consider dropping one)
            - Values above 0.8 or below -0.8 indicate strong relationships
            """)

        fig, ax = plt.subplots(figsize=(10, 8))
        correlation = st.session_state.df_processed.corr()
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        ax.set_title("Feature Correlation Matrix")
        st.pyplot(fig)

        # Additional feature selection based on correlation
        st.subheader("3. Additional Feature Selection")
        with st.expander("ℹ️ Why remove highly correlated features?"):
            st.write("""
            **Multicollinearity** occurs when features are highly correlated with each other.
            This can cause:
            - Unstable coefficient estimates
            - Difficulty interpreting feature importance
            - Reduced model reliability

            Consider removing one feature from pairs with correlation > 0.8
            """)

        available_cols = st.session_state.df_processed.columns.tolist()

        # Use form to prevent rerun on selection
        with st.form("additional_feature_selection_form"):
            additional_drop = st.multiselect(
                "Select additional columns to drop based on correlation analysis:",
                options=available_cols,
                help="Consider dropping features that are highly correlated with other features (multicollinearity)"
            )

            # Apply additional dropping with form submit button
            apply_button = st.form_submit_button("Apply Additional Dropping")

        if apply_button:
            if additional_drop:
                st.session_state.df_after_correlation = st.session_state.df_processed.drop(columns=additional_drop,
                                                                                           errors='ignore')
                st.success(f"Dropped {len(additional_drop)} additional columns")

                # Show updated correlation matrix
                st.markdown("##### Updated Correlation Matrix (After Additional Dropping)")
                fig_updated, ax_updated = plt.subplots(figsize=(10, 8))
                correlation_updated = st.session_state.df_after_correlation.corr()
                sns.heatmap(correlation_updated, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax_updated)
                ax_updated.set_title("Updated Feature Correlation Matrix")
                st.pyplot(fig_updated)

                # Show updated scatter matrix
                st.markdown("##### Updated Scatter Matrix (After Additional Dropping)")
                with st.expander("ℹ️ Understanding the scatter matrix"):
                    st.write("""
                    The scatter matrix shows pairwise relationships between all features:
                    - **Diagonal**: Distribution of each feature (histogram)
                    - **Off-diagonal**: Scatter plots between feature pairs
                    - **Look for**: Linear patterns, outliers, and relationships
                    """)

                if len(st.session_state.df_after_correlation.columns) <= 10:
                    with st.spinner("Generating updated scatter matrix plot..."):
                        fig_scatter, axes_scatter = plt.subplots(figsize=(14, 14))
                        axes_scatter = pd.plotting.scatter_matrix(st.session_state.df_after_correlation,
                                                                  alpha=0.2,
                                                                  figsize=(14, 14),
                                                                  diagonal='hist')

                        # Rotate axis labels for readability
                        for ax in axes_scatter.flatten():
                            ax.xaxis.label.set_rotation(90)
                            ax.yaxis.label.set_rotation(0)
                            ax.yaxis.label.set_ha('right')

                        plt.tight_layout()
                        plt.gcf().subplots_adjust(wspace=0, hspace=0)
                        st.pyplot(plt.gcf())
                        plt.close()
                else:
                    st.warning(
                        f"Updated scatter matrix is not displayed for datasets with more than 10 columns (current: {len(st.session_state.df_after_correlation.columns)})")
            else:
                st.warning("No columns selected for dropping")

# Tab 3: Model Training
with tab3:
    st.header("Model Training")
    st.markdown("""
    Train your Multiple Linear Regression model. The model will learn the relationship between your features and target variable,
    creating an equation to make predictions.
    """)

    # Determine which dataframe to use
    if st.session_state.df_after_correlation is not None:
        working_df = st.session_state.df_after_correlation
    elif st.session_state.df_processed is not None:
        working_df = st.session_state.df_processed
    else:
        st.warning("⚠️ Please preprocess the data first in the Preprocessing tab")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        # Feature and target selection
        st.subheader("Select Features and Target")

        with st.expander("ℹ️ What are target and feature variables?"):
            st.write("""
            **Target Variable (Dependent Variable):**
            - What you want to predict
            - Examples: Price, Sales, Temperature, Risk Score

            **Feature Variables (Independent Variables):**
            - Information used to make predictions
            - Examples: Size, Age, Location, Previous Sales

            The model will learn how features influence the target.
            """)

        available_cols = working_df.columns.tolist()

        target_variable = st.selectbox(
            "Select target variable (y) - what you want to predict:",
            options=available_cols,
            index=len(available_cols) - 1 if available_cols else 0,
            help="This is what your model will learn to predict"
        )

        feature_variables = st.multiselect(
            "Select feature variables (X) - information for making predictions:",
            options=[col for col in available_cols if col != target_variable],
            default=[col for col in available_cols if col != target_variable],
            help="These are the inputs your model will use to make predictions"
        )

    with col2:
        st.subheader("Training Parameters")

        with st.expander("ℹ️ Understanding training parameters"):
            st.write("""
            **Test Set Size:**
            - Percentage of data reserved for testing (not used in training)
            - Typically 20-30%
            - Larger test set = better validation but less training data

            **Random State:**
            - Seeds the random split for reproducibility
            - Same number = same split every time
            - Change to get different train/test splits

            **Standardization:**
            - Scales features to similar ranges (mean=0, std=1)
            - Important when features have different units
            - Makes coefficients comparable
            """)

        test_size = st.slider("Test set size:", 0.1, 0.4, 0.2, 0.05,
                              help="Percentage of data to reserve for testing. 20% is a common choice.")
        random_state = st.number_input("Random state:", 0, 100, 42,
                                       help="Set this to any number for reproducible results. Common choice: 42")
        standardize = st.checkbox("Standardize features", value=True,
                                  help="Recommended when features have different scales (e.g., age in years vs salary in thousands)")

    # Train model button
    if st.button("🚀 Train Model", type="primary"):
        if len(feature_variables) == 0:
            st.error("Please select at least one feature variable")
        else:
            try:
                # Prepare data
                X = working_df[feature_variables].values
                y = working_df[target_variable].values.reshape(-1, 1)

                # Standardize if selected
                if standardize:
                    scaler = preprocessing.StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    st.session_state.scaler = scaler
                else:
                    X_scaled = X
                    st.session_state.scaler = None

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size, random_state=random_state
                )

                # Train model
                regressor = linear_model.LinearRegression()
                regressor.fit(X_train, y_train)

                # Make predictions
                y_train_pred = regressor.predict(X_train)
                y_test_pred = regressor.predict(X_test)

                # Calculate metrics
                train_mse = mean_squared_error(y_train, y_train_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)

                # Store all results in session state
                st.session_state.model = regressor
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.y_train_pred = y_train_pred
                st.session_state.y_test_pred = y_test_pred
                st.session_state.feature_names = feature_variables
                st.session_state.target_name = target_variable

                # Store training results for persistence
                st.session_state.training_results = {
                    'coefficients': regressor.coef_[0] if regressor.coef_.ndim > 1 else regressor.coef_,
                    'intercept': regressor.intercept_[0] if hasattr(regressor.intercept_,
                                                                    '__len__') else regressor.intercept_,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'standardized': standardize
                }

                st.success("✅ Model trained successfully!")

            except Exception as e:
                st.error(f"Error during model training: {str(e)}")

    # Display training results if available
    if st.session_state.training_results is not None:
        st.markdown("---")
        st.subheader("📊 Training Results")

        # Model coefficients
        st.subheader("Model Parameters")

        with st.expander("ℹ️ Understanding coefficients"):
            st.write("""
            **Coefficients** tell you how much the target changes when a feature increases by 1 unit:
            - **Positive coefficient**: Feature and target move together
            - **Negative coefficient**: Feature increases, target decreases
            - **Larger absolute value**: Stronger influence

            **Intercept** is the predicted value when all features are zero.

            If standardized, compare coefficient magnitudes to see feature importance.
            """)

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Coefficients:**")
            # Ensure coefficients is always an array
            coef_array = st.session_state.training_results['coefficients']
            if not hasattr(coef_array, '__len__'):
                coef_array = [coef_array]

            coef_df = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'Coefficient': coef_array
            })
            st.dataframe(coef_df)
            st.write(f"**Intercept:** {st.session_state.training_results['intercept']:.4f}")

        with col2:
            # If standardized, show original scale coefficients
            if st.session_state.training_results['standardized'] and st.session_state.scaler:
                means_ = st.session_state.scaler.mean_
                std_devs_ = np.sqrt(st.session_state.scaler.var_)

                # Ensure coefficients is an array
                coef_array = st.session_state.training_results['coefficients']
                if not hasattr(coef_array, '__len__'):
                    coef_array = [coef_array]

                coef_original = np.array(coef_array) / std_devs_
                intercept_original = st.session_state.training_results['intercept'] - np.sum(
                    np.array(coef_array) * means_ / std_devs_
                )

                st.write("**Original Scale Coefficients:**")
                st.info("These show the actual change in target per unit change in each feature")
                coef_original_df = pd.DataFrame({
                    'Feature': st.session_state.feature_names,
                    'Coefficient': coef_original
                })
                st.dataframe(coef_original_df)
                st.write(f"**Original Intercept:** {intercept_original:.4f}")

        # Performance metrics
        st.subheader("Model Performance")

        with st.expander("ℹ️ Understanding performance metrics"):
            st.write("""
            **MSE (Mean Squared Error):** Average of squared prediction errors. Lower is better.

            **RMSE (Root MSE):** Square root of MSE, in same units as target. Shows typical prediction error.

            **MAE (Mean Absolute Error):** Average absolute prediction error. Less sensitive to outliers than MSE.

            **R² Score (0 to 1):** Percentage of variance explained by the model:
            - 1.0 = Perfect predictions
            - 0.8-0.9 = Very good model
            - 0.6-0.8 = Good model
            - < 0.5 = Poor model

            **Compare Training vs Testing:** Similar values = good. Large difference = overfitting.
            """)

        metrics_df = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE', 'R² Score'],
            'Training': [
                st.session_state.training_results['train_mse'],
                np.sqrt(st.session_state.training_results['train_mse']),
                st.session_state.training_results['train_mae'],
                st.session_state.training_results['train_r2']
            ],
            'Testing': [
                st.session_state.training_results['test_mse'],
                np.sqrt(st.session_state.training_results['test_mse']),
                st.session_state.training_results['test_mae'],
                st.session_state.training_results['test_r2']
            ]
        })

        st.dataframe(metrics_df.round(4))

        # Interpret R² score
        test_r2 = st.session_state.training_results['test_r2']
        if test_r2 > 0.9:
            st.success(
                f"Excellent model! R² = {test_r2:.3f} means your model explains {test_r2 * 100:.1f}% of the variance.")
        elif test_r2 > 0.7:
            st.success(
                f"Good model! R² = {test_r2:.3f} means your model explains {test_r2 * 100:.1f}% of the variance.")
        elif test_r2 > 0.5:
            st.info(f"Decent model. R² = {test_r2:.3f} means your model explains {test_r2 * 100:.1f}% of the variance.")
        else:
            st.warning(
                f"Model needs improvement. R² = {test_r2:.3f} means your model only explains {test_r2 * 100:.1f}% of the variance.")

        # Performance comparison chart
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Bar chart for metrics
        metrics_df.set_index('Metric')[['Training', 'Testing']].iloc[:3].plot(
            kind='bar', ax=axes[0], color=['blue', 'orange']
        )
        axes[0].set_title('Error Metrics Comparison')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # R² Score comparison
        r2_data = metrics_df.set_index('Metric')[['Training', 'Testing']].iloc[3]
        axes[1].bar(['Training', 'Testing'], r2_data.values, color=['blue', 'orange'])
        axes[1].set_title('R² Score Comparison')
        axes[1].set_ylabel('R² Score')
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3)

        st.pyplot(fig)

# Tab 4: Visualization
with tab4:
    st.header("Model Visualization")
    st.markdown("""
    Visualize your model's performance and understand how it makes predictions.
    Different visualizations help identify strengths and weaknesses.
    """)

    if 'model' not in st.session_state:
        st.warning("⚠️ Please train the model first in the Model Training tab")
        st.stop()

    # Visualization options - dynamically adjust based on number of features
    n_features = len(st.session_state.feature_names)

    if n_features == 2:
        viz_options = ["Actual vs Predicted", "Residual Plots", "Feature Importance",
                       "Individual Feature Relationships", "3D Visualization"]
    else:
        viz_options = ["Actual vs Predicted", "Residual Plots", "Feature Importance",
                       "Individual Feature Relationships"]

    with st.expander("ℹ️ Understanding visualization types"):
        st.write("""
        **Actual vs Predicted:** Shows how close predictions are to real values. Points should cluster around the diagonal line.

        **Residual Plots:** Shows prediction errors. Look for random scatter (good) vs patterns (problems).

        **Feature Importance:** Shows which features most influence predictions based on coefficient magnitudes.

        **Individual Feature Relationships:** Shows how each feature relates to the target variable.

        **3D Visualization (2 features only):** Shows the regression plane in 3D space.
        """)

    viz_type = st.selectbox(
        "Select visualization type:",
        viz_options,
        help="Choose different views to understand your model's behavior"
    )

    if viz_type == "Actual vs Predicted":
        st.markdown("Points closer to the red diagonal line indicate better predictions.")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Training data
        axes[0].scatter(st.session_state.y_train, st.session_state.y_train_pred,
                        alpha=0.5, color='blue', s=20)
        axes[0].plot([st.session_state.y_train.min(), st.session_state.y_train.max()],
                     [st.session_state.y_train.min(), st.session_state.y_train.max()],
                     'r--', lw=2)
        axes[0].set_xlabel(f'Actual {st.session_state.target_name}')
        axes[0].set_ylabel(f'Predicted {st.session_state.target_name}')
        axes[0].set_title('Training Set: Actual vs Predicted')
        axes[0].grid(True, alpha=0.3)

        # Testing data
        axes[1].scatter(st.session_state.y_test, st.session_state.y_test_pred,
                        alpha=0.5, color='orange', s=20)
        axes[1].plot([st.session_state.y_test.min(), st.session_state.y_test.max()],
                     [st.session_state.y_test.min(), st.session_state.y_test.max()],
                     'r--', lw=2)
        axes[1].set_xlabel(f'Actual {st.session_state.target_name}')
        axes[1].set_ylabel(f'Predicted {st.session_state.target_name}')
        axes[1].set_title('Testing Set: Actual vs Predicted')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

    elif viz_type == "Residual Plots":
        st.markdown("""
        Residuals (prediction errors) should be randomly scattered around zero.
        Patterns indicate model problems: curves suggest non-linearity, increasing spread suggests heteroscedasticity.
        """)

        # Calculate residuals
        train_residuals = st.session_state.y_train - st.session_state.y_train_pred
        test_residuals = st.session_state.y_test - st.session_state.y_test_pred

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Training residuals
        axes[0, 0].scatter(st.session_state.y_train_pred, train_residuals,
                           alpha=0.5, color='blue', s=20)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Training Set: Residual Plot')
        axes[0, 0].grid(True, alpha=0.3)

        # Testing residuals
        axes[0, 1].scatter(st.session_state.y_test_pred, test_residuals,
                           alpha=0.5, color='orange', s=20)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Testing Set: Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)

        # Histogram of residuals
        axes[1, 0].hist(train_residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Training Set: Residual Distribution')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].hist(test_residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Testing Set: Residual Distribution')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

    elif viz_type == "Feature Importance":
        st.markdown("""
        Feature importance based on coefficient magnitudes. Larger bars indicate stronger influence on predictions.
        Green = positive effect, Red = negative effect.
        """)

        if not st.session_state.training_results['standardized']:
            st.warning(
                "Note: Features not standardized. Coefficient magnitudes may not be directly comparable if features have different scales.")

        # Get coefficients
        coef = st.session_state.model.coef_[
            0] if st.session_state.model.coef_.ndim > 1 else st.session_state.model.coef_
        features = st.session_state.feature_names

        # Ensure coef is always a 1D array
        if not hasattr(coef, '__len__'):
            coef = [coef]
        elif coef.ndim > 1:
            coef = coef.flatten()

        # Sort by absolute value
        importance = pd.DataFrame({
            'Feature': features,
            'Coefficient': coef,
            'Abs_Coefficient': np.abs(coef)
        }).sort_values('Abs_Coefficient', ascending=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Coefficient values
        axes[0].barh(importance['Feature'], importance['Coefficient'],
                     color=['green' if x > 0 else 'red' for x in importance['Coefficient']])
        axes[0].set_xlabel('Coefficient Value')
        axes[0].set_title('Feature Coefficients')
        axes[0].grid(True, alpha=0.3)

        # Absolute importance
        axes[1].barh(importance['Feature'], importance['Abs_Coefficient'],
                     color='purple', alpha=0.7)
        axes[1].set_xlabel('Absolute Coefficient Value')
        axes[1].set_title('Feature Importance (Absolute Values)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

    elif viz_type == "Individual Feature Relationships":
        st.markdown("""
        Each plot shows how a single feature relates to the target. 
        The red line shows the overall trend in the data.
        """)

        n_features = len(st.session_state.feature_names)

        if n_features == 1:
            # Handle single feature case (Simple Linear Regression)
            fig, ax = plt.subplots(figsize=(12, 6))

            # Flatten arrays if needed for single feature
            X_train_flat = st.session_state.X_train.flatten() if st.session_state.X_train.ndim > 1 else st.session_state.X_train
            X_test_flat = st.session_state.X_test.flatten() if st.session_state.X_test.ndim > 1 else st.session_state.X_test
            y_train_flat = st.session_state.y_train.flatten()
            y_test_flat = st.session_state.y_test.flatten()

            ax.scatter(X_train_flat, y_train_flat, c='blue', alpha=0.3, label='Train', s=20)
            ax.scatter(X_test_flat, y_test_flat, c='orange', alpha=0.3, label='Test', s=20)

            # Add regression line
            X_combined = np.concatenate([X_train_flat, X_test_flat])
            X_line = np.linspace(X_combined.min(), X_combined.max(), 100)
            coef = st.session_state.model.coef_[0, 0] if st.session_state.model.coef_.ndim > 1 else \
                st.session_state.model.coef_[0]
            intercept = st.session_state.model.intercept_[0] if hasattr(st.session_state.model.intercept_,
                                                                        '__len__') else st.session_state.model.intercept_
            y_line = coef * X_line + intercept
            ax.plot(X_line, y_line, 'r-', linewidth=2, label='Regression Line')

            ax.set_xlabel(st.session_state.feature_names[0])
            ax.set_ylabel(st.session_state.target_name)
            ax.set_title(
                f'Simple Linear Regression: {st.session_state.feature_names[0]} vs {st.session_state.target_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add equation to plot
            equation = f'y = {coef:.4f}x + {intercept:.4f}'
            ax.text(0.05, 0.95, equation, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            st.pyplot(fig)
        else:
            # Handle multiple features case
            n_cols = 2
            n_rows = (n_features + 1) // 2

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

            for idx, feature in enumerate(st.session_state.feature_names):
                if idx < len(axes):
                    axes[idx].scatter(st.session_state.X_train[:, idx], st.session_state.y_train,
                                      c='blue', alpha=0.3, label='Train', s=20)
                    axes[idx].scatter(st.session_state.X_test[:, idx], st.session_state.y_test,
                                      c='orange', alpha=0.3, label='Test', s=20)

                    # Add partial regression line
                    X_combined = np.vstack([st.session_state.X_train[:, idx:idx + 1],
                                            st.session_state.X_test[:, idx:idx + 1]])
                    X_line = np.linspace(X_combined.min(), X_combined.max(), 100)

                    # For multiple regression, show the marginal effect
                    # This is simplified - just showing the direction and magnitude
                    coef = st.session_state.model.coef_[0, idx] if st.session_state.model.coef_.ndim > 1 else \
                        st.session_state.model.coef_[idx]

                    # Create a trend line based on the coefficient
                    from scipy import stats

                    slope, intercept_line, _, _, _ = stats.linregress(
                        np.concatenate([st.session_state.X_train[:, idx], st.session_state.X_test[:, idx]]),
                        np.concatenate([st.session_state.y_train.flatten(), st.session_state.y_test.flatten()])
                    )
                    y_trend = slope * X_line + intercept_line
                    axes[idx].plot(X_line, y_trend, 'r-', linewidth=2, alpha=0.7, label='Trend Line')

                    axes[idx].set_xlabel(feature)
                    axes[idx].set_ylabel(st.session_state.target_name)
                    axes[idx].set_title(f'{feature} vs {st.session_state.target_name}')
                    axes[idx].legend()
                    axes[idx].grid(True, alpha=0.3)

                    # Add coefficient info
                    # axes[idx].text(0.05, 0.95, f'Coef: {coef:.4f}', transform=axes[idx].transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Hide unused subplots
            for idx in range(n_features, len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)

    elif viz_type == "3D Visualization":
        st.markdown("""
        3D visualization shows the regression plane fitted through your data points.
        This helps visualize how two features combine to predict the target.
        """)

        # This option only appears when there are exactly 2 features
        fig = plt.figure(figsize=(14, 10))

        # 3D scatter plot
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.scatter(st.session_state.X_test[:, 0],
                    st.session_state.X_test[:, 1],
                    st.session_state.y_test,
                    c='blue', marker='o', alpha=0.5, label='Actual')
        ax1.scatter(st.session_state.X_test[:, 0],
                    st.session_state.X_test[:, 1],
                    st.session_state.y_test_pred,
                    c='red', marker='^', alpha=0.5, label='Predicted')
        ax1.set_xlabel(st.session_state.feature_names[0])
        ax1.set_ylabel(st.session_state.feature_names[1])
        ax1.set_zlabel(st.session_state.target_name)
        ax1.set_title('3D Scatter: Actual vs Predicted')
        ax1.legend()

        # Create regression plane
        ax2 = fig.add_subplot(222, projection='3d')

        # Create mesh grid
        X1_range = np.linspace(st.session_state.X_test[:, 0].min(),
                               st.session_state.X_test[:, 0].max(), 30)
        X2_range = np.linspace(st.session_state.X_test[:, 1].min(),
                               st.session_state.X_test[:, 1].max(), 30)
        X1_mesh, X2_mesh = np.meshgrid(X1_range, X2_range)

        # Predict on mesh
        X_mesh = np.c_[X1_mesh.ravel(), X2_mesh.ravel()]
        y_mesh = st.session_state.model.predict(X_mesh).reshape(X1_mesh.shape)

        # Plot surface
        ax2.plot_surface(X1_mesh, X2_mesh, y_mesh, alpha=0.3, cmap='viridis')
        ax2.scatter(st.session_state.X_test[:, 0],
                    st.session_state.X_test[:, 1],
                    st.session_state.y_test,
                    c='blue', marker='o', alpha=0.5)
        ax2.set_xlabel(st.session_state.feature_names[0])
        ax2.set_ylabel(st.session_state.feature_names[1])
        ax2.set_zlabel(st.session_state.target_name)
        ax2.set_title('Regression Plane with Test Data')

        # Individual feature plots
        ax3 = fig.add_subplot(223)
        ax3.scatter(st.session_state.X_train[:, 0], st.session_state.y_train,
                    c='blue', alpha=0.3, label='Train')
        ax3.scatter(st.session_state.X_test[:, 0], st.session_state.y_test,
                    c='orange', alpha=0.3, label='Test')

        # Add regression line for first feature
        coef = st.session_state.model.coef_[0, 0]
        intercept = st.session_state.model.intercept_[0]
        X_line = np.linspace(st.session_state.X_test[:, 0].min(),
                             st.session_state.X_test[:, 0].max(), 100)
        y_line = coef * X_line + intercept
        ax3.plot(X_line, y_line, 'r-', linewidth=2, label='Partial Effect')

        ax3.set_xlabel(st.session_state.feature_names[0])
        ax3.set_ylabel(st.session_state.target_name)
        ax3.set_title(f'{st.session_state.feature_names[0]} vs {st.session_state.target_name}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(224)
        ax4.scatter(st.session_state.X_train[:, 1], st.session_state.y_train,
                    c='blue', alpha=0.3, label='Train')
        ax4.scatter(st.session_state.X_test[:, 1], st.session_state.y_test,
                    c='orange', alpha=0.3, label='Test')

        # Add regression line for second feature
        coef = st.session_state.model.coef_[0, 1]
        X_line = np.linspace(st.session_state.X_test[:, 1].min(),
                             st.session_state.X_test[:, 1].max(), 100)
        y_line = coef * X_line + intercept
        ax4.plot(X_line, y_line, 'r-', linewidth=2, label='Partial Effect')

        ax4.set_xlabel(st.session_state.feature_names[1])
        ax4.set_ylabel(st.session_state.target_name)
        ax4.set_title(f'{st.session_state.feature_names[1]} vs {st.session_state.target_name}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

# Tab 5: Predictions
with tab5:
    st.header("Make Predictions")
    st.markdown("""
    Use your trained model to make predictions on new data. 
    Enter values for each feature to get a predicted target value.
    """)

    if 'model' not in st.session_state:
        st.warning("⚠️ Please train the model first in the Model Training tab")
        st.stop()

    # Determine which dataframe to use for reference
    if st.session_state.df_after_correlation is not None:
        reference_df = st.session_state.df_after_correlation
    elif st.session_state.df_processed is not None:
        reference_df = st.session_state.df_processed
    else:
        reference_df = st.session_state.df

    st.subheader("Single Prediction")

    st.info(
        "💡 **Note**: You can enter any values, including those outside your training data range (extrapolation). However, predictions are most reliable within the training data range shown in the tooltips.")

    # Create input fields for each feature
    col1, col2 = st.columns(2)
    input_values = {}

    for idx, feature in enumerate(st.session_state.feature_names):
        if idx % 2 == 0:
            with col1:
                # Get statistics from reference data for default value
                if reference_df is not None and feature in reference_df.columns:
                    default_val = float(reference_df[feature].mean())
                    step_val = float(reference_df[feature].std() / 10) if reference_df[feature].std() > 0 else 0.1
                else:
                    default_val = 0.0
                    step_val = 0.1

                input_values[feature] = st.number_input(
                    f"{feature}:",
                    value=default_val,
                    step=step_val,
                    format="%.4f",
                    help=f"Mean: {reference_df[feature].mean():.2f}, Std: {reference_df[feature].std():.2f}, Range: [{reference_df[feature].min():.2f}, {reference_df[feature].max():.2f}]" if reference_df is not None and feature in reference_df.columns else "Enter any value"
                )
        else:
            with col2:
                if reference_df is not None and feature in reference_df.columns:
                    default_val = float(reference_df[feature].mean())
                    step_val = float(reference_df[feature].std() / 10) if reference_df[feature].std() > 0 else 0.1
                else:
                    default_val = 0.0
                    step_val = 0.1

                input_values[feature] = st.number_input(
                    f"{feature}:",
                    value=default_val,
                    step=step_val,
                    format="%.4f",
                    help=f"Mean: {reference_df[feature].mean():.2f}, Std: {reference_df[feature].std():.2f}, Range: [{reference_df[feature].min():.2f}, {reference_df[feature].max():.2f}]" if reference_df is not None and feature in reference_df.columns else "Enter any value"
                )

    # Make prediction button
    if st.button("🎯 Make Prediction", type="primary"):
        # Prepare input data
        X_input = np.array([input_values[feature] for feature in st.session_state.feature_names]).reshape(1, -1)

        # Standardize if necessary
        if st.session_state.scaler is not None:
            X_input_scaled = st.session_state.scaler.transform(X_input)
        else:
            X_input_scaled = X_input

        # Make prediction
        prediction = st.session_state.model.predict(X_input_scaled)[0, 0]

        # Display prediction
        st.success(f"### Predicted {st.session_state.target_name}: **{prediction:.4f}**")

        # Show input summary
        st.subheader("Input Summary")
        input_df = pd.DataFrame([input_values])
        st.dataframe(input_df)

        # Show warning if extrapolating
        if reference_df is not None:
            extrapolating = False
            for feature in st.session_state.feature_names:
                if feature in reference_df.columns:
                    if input_values[feature] < reference_df[feature].min() or input_values[feature] > reference_df[
                        feature].max():
                        extrapolating = True
                        break

            if extrapolating:
                st.warning(
                    "⚠️ **Extrapolation Warning**: One or more input values are outside the training data range. Predictions may be less reliable.")

        # Calculate prediction interval (simplified - assuming normal distribution)
        # This is a rough approximation
        residuals = st.session_state.y_test - st.session_state.y_test_pred
        std_residuals = np.std(residuals)
        confidence_level = 0.95
        z_score = 1.96  # for 95% confidence

        lower_bound = prediction - z_score * std_residuals
        upper_bound = prediction + z_score * std_residuals

        with st.expander("ℹ️ Understanding prediction intervals"):
            st.write("""
            The **95% Prediction Interval** gives a range where we expect the actual value to fall 95% of the time.
            This accounts for the model's typical prediction error.

            Wider intervals indicate more uncertainty in the prediction.
            """)

        st.info(f"**95% Prediction Interval:** [{lower_bound:.4f}, {upper_bound:.4f}]")

    # Batch prediction
    st.subheader("Batch Predictions")
    st.markdown("""
    Upload a CSV file with multiple records to get predictions for all of them at once.
    The file must contain columns with the same names as your training features.
    """)

    uploaded_pred_file = st.file_uploader(
        "Upload CSV for batch predictions",
        type=['csv'],
        key='batch_pred',
        help=f"CSV must contain these columns: {', '.join(st.session_state.feature_names)}"
    )

    if uploaded_pred_file is not None:
        try:
            pred_df = pd.read_csv(uploaded_pred_file)

            # Check if all required features are present
            missing_features = set(st.session_state.feature_names) - set(pred_df.columns)
            if missing_features:
                st.error(f"Missing features in uploaded file: {missing_features}")
                st.info(f"Required columns: {', '.join(st.session_state.feature_names)}")
            else:
                # Prepare data
                X_batch = pred_df[st.session_state.feature_names].values

                # Standardize if necessary
                if st.session_state.scaler is not None:
                    X_batch_scaled = st.session_state.scaler.transform(X_batch)
                else:
                    X_batch_scaled = X_batch

                # Make predictions
                predictions = st.session_state.model.predict(X_batch_scaled)

                # Add predictions to dataframe
                pred_df[f'Predicted_{st.session_state.target_name}'] = predictions.flatten()

                # Display results
                st.success(f"✅ Predictions made for {len(pred_df)} samples")
                st.dataframe(pred_df)

                # Download button for predictions
                csv = pred_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv'
                )

        except Exception as e:
            st.error(f"Error processing batch predictions: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit | Multiple Linear Regression Analysis Tool</p>
    <p style='font-size: 0.9em; color: gray;'>Create predictive models with ease using machine learning</p>
</div>
""", unsafe_allow_html=True)