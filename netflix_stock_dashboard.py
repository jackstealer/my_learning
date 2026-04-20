import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Netflix Stock Prediction Dashboard", layout="wide", page_icon="📈")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'results' not in st.session_state:
    st.session_state.results = None

# Sidebar
st.sidebar.title("1. Data Source")
st.sidebar.subheader("Upload CSV")

data_source = st.sidebar.radio("Choose data source:", ["Upload CSV", "Fetch from Yahoo Finance"])

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Drag and drop file here\nLimit 200MB per file • CSV", type=['csv'])
    
    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.sidebar.success("File Uploaded!")
        st.sidebar.write(f"📄 {uploaded_file.name}")
        st.sidebar.write(f"{len(st.session_state.data)} rows")
else:
    st.sidebar.write("Fetch Netflix stock data")
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365*5))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    
    if st.sidebar.button("Fetch Data"):
        with st.spinner("Fetching Netflix stock data..."):
            ticker = yf.Ticker("NFLX")
            st.session_state.data = ticker.history(start=start_date, end=end_date)
            st.session_state.data.reset_index(inplace=True)
        st.sidebar.success("Data Fetched!")
        st.sidebar.write(f"{len(st.session_state.data)} rows")

# Main content
st.title("📈 Interactive ML Pipeline Dashboard")
st.subheader("Netflix Stock Prediction")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Data & EDA", 
    "Cleaning & Engineering", 
    "Feature Selection", 
    "Model Training", 
    "Performance",
    "Predictions"
])

# Tab 1: Data & EDA
with tab1:
    if st.session_state.data is not None:
        st.header("Exploratory Data Analysis")
        
        df = st.session_state.data.copy()
        
        # Select target variable
        st.subheader("Select Target Variable")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target = st.selectbox("Target Variable", numeric_cols, index=0 if 'Close' in numeric_cols else 0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Summary")
            summary_df = df.describe().T
            st.dataframe(summary_df, use_container_width=True)
        
        with col2:
            st.subheader("Correlation Heatmap")
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                fig = px.imshow(numeric_df.corr(), 
                               text_auto='.2f',
                               aspect="auto",
                               color_continuous_scale='RdBu_r')
                st.plotly_chart(fig, use_container_width=True)
        
        # Time series plot
        st.subheader("Time Series Visualization")
        if 'Date' in df.columns:
            fig = px.line(df, x='Date', y=target, title=f'{target} Over Time')
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution plots
        st.subheader("Feature Distributions")
        selected_cols = st.multiselect("Select features to visualize", numeric_cols, default=numeric_cols[:4])
        if selected_cols:
            fig = go.Figure()
            for col in selected_cols:
                fig.add_trace(go.Box(y=df[col], name=col))
            fig.update_layout(title="Box Plots of Selected Features")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Please upload a CSV file or fetch data from Yahoo Finance to begin")

# Tab 2: Cleaning & Engineering
with tab2:
    if st.session_state.data is not None:
        st.header("Data Engineering")
        
        df = st.session_state.data.copy()
        
        # Handle missing values
        st.subheader("1. Handle Missing Values")
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if missing_cols:
            st.write("Columns with missing values:")
            for col in missing_cols:
                st.write(f"- {col}: {df[col].isnull().sum()} missing values")
            
            action = st.radio("Action:", ["Keep as is", "Delete Rows", "Impute with Mean", "Impute with Median"])
            
            if action == "Delete Rows":
                df = df.dropna()
                st.success(f"Removed rows with missing values. New shape: {df.shape}")
            elif action == "Impute with Mean":
                for col in missing_cols:
                    if df[col].dtype in [np.float64, np.int64]:
                        df[col].fillna(df[col].mean(), inplace=True)
                st.success("Imputed missing values with mean")
            elif action == "Impute with Median":
                for col in missing_cols:
                    if df[col].dtype in [np.float64, np.int64]:
                        df[col].fillna(df[col].median(), inplace=True)
                st.success("Imputed missing values with median")
        else:
            st.success("✅ No missing values found!")
        
        # Outlier removal
        st.subheader("2. Outlier Removal (IQR)")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if st.checkbox("Remove outliers using IQR method"):
            outlier_cols = st.multiselect("Select columns for outlier removal", numeric_cols)
            
            if outlier_cols:
                original_shape = df.shape
                for col in outlier_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
                
                st.success(f"Removed {original_shape[0] - df.shape[0]} outliers")
                st.write(f"Current Data Shape: {df.shape}")
        
        # Feature engineering
        st.subheader("3. Feature Engineering")
        
        if 'Date' in df.columns:
            if st.checkbox("Create time-based features"):
                df['Date'] = pd.to_datetime(df['Date'])
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
                df['Day'] = df['Date'].dt.day
                df['DayOfWeek'] = df['Date'].dt.dayofweek
                st.success("Created: Year, Month, Day, DayOfWeek")
        
        if st.checkbox("Create technical indicators"):
            if 'Close' in df.columns:
                # Moving averages
                df['MA_7'] = df['Close'].rolling(window=7).mean()
                df['MA_21'] = df['Close'].rolling(window=21).mean()
                df['MA_50'] = df['Close'].rolling(window=50).mean()
                
                # Volatility
                df['Volatility'] = df['Close'].rolling(window=21).std()
                
                # Price change
                df['Price_Change'] = df['Close'].pct_change()
                df['Price_Change_7d'] = df['Close'].pct_change(periods=7)
                
                df = df.dropna()
                st.success("Created: MA_7, MA_21, MA_50, Volatility, Price_Change, Price_Change_7d")
        
        st.session_state.processed_data = df
        st.success("✅ Data engineering completed!")
        
        # Show processed data
        with st.expander("View Processed Data"):
            st.dataframe(df.head(20), use_container_width=True)
    else:
        st.info("👈 Please upload data first")

# Tab 3: Feature Selection
with tab3:
    if st.session_state.processed_data is not None:
        st.header("Feature Engineering & Selection")
        
        df = st.session_state.processed_data.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        st.subheader("Select Method:")
        method = st.radio("", ["All Features", "Variance Threshold", "Correlation-based", "Manual Selection"])
        
        if method == "All Features":
            selected_features = [col for col in numeric_cols if col != target]
            st.write("Selected Features:")
            for i, feat in enumerate(selected_features, 1):
                st.write(f"{i}. **{feat}**")
        
        elif method == "Variance Threshold":
            threshold = st.slider("Variance Threshold", 0.0, 1.0, 0.01)
            variances = df[numeric_cols].var()
            selected_features = variances[variances > threshold].index.tolist()
            if target in selected_features:
                selected_features.remove(target)
            
            st.write(f"Selected {len(selected_features)} features with variance > {threshold}")
            for i, feat in enumerate(selected_features, 1):
                st.write(f"{i}. **{feat}**")
        
        elif method == "Correlation-based":
            corr_threshold = st.slider("Correlation Threshold with Target", 0.0, 1.0, 0.1)
            correlations = df[numeric_cols].corrwith(df[target]).abs()
            selected_features = correlations[correlations > corr_threshold].index.tolist()
            if target in selected_features:
                selected_features.remove(target)
            
            st.write(f"Selected {len(selected_features)} features with |correlation| > {corr_threshold}")
            
            # Show correlation bar chart
            fig = px.bar(x=correlations.index, y=correlations.values, 
                        labels={'x': 'Features', 'y': 'Correlation with Target'},
                        title='Feature Correlations with Target')
            st.plotly_chart(fig, use_container_width=True)
        
        else:  # Manual Selection
            selected_features = st.multiselect(
                "Select features manually:",
                [col for col in numeric_cols if col != target],
                default=[col for col in numeric_cols if col != target][:5]
            )
        
        st.session_state.selected_features = selected_features
        
        if selected_features:
            st.success(f"✅ {len(selected_features)} features selected!")
            
            # Feature importance preview
            with st.expander("Feature Statistics"):
                stats_df = df[selected_features].describe().T
                st.dataframe(stats_df, use_container_width=True)
    else:
        st.info("👈 Please complete data engineering first")

# Tab 4: Model Training
with tab4:
    if st.session_state.processed_data is not None and st.session_state.selected_features:
        st.header("Training Configuration")
        
        df = st.session_state.processed_data.copy()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Choose Model")
            model_name = st.selectbox("", [
                "Linear Regression",
                "Ridge Regression",
                "Lasso Regression",
                "Random Forest",
                "Gradient Boosting"
            ])
        
        with col2:
            st.info("K-Fold Cross Validation (k=5) is enabled by default for stability.")
        
        # Test set size
        test_size = st.slider("Test Set Size %", 10, 40, 20) / 100
        
        # Model-specific parameters
        if model_name == "Ridge Regression":
            alpha = st.slider("Alpha (Regularization)", 0.01, 10.0, 1.0)
        elif model_name == "Lasso Regression":
            alpha = st.slider("Alpha (Regularization)", 0.01, 10.0, 1.0)
        elif model_name == "Random Forest":
            n_estimators = st.slider("Number of Trees", 10, 200, 100)
            max_depth = st.slider("Max Depth", 3, 20, 10)
        elif model_name == "Gradient Boosting":
            n_estimators = st.slider("Number of Estimators", 10, 200, 100)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
        
        if st.button("🚀 Start Training Pipeline", type="primary"):
            with st.spinner("Training model..."):
                # Prepare data
                X = df[st.session_state.selected_features].dropna()
                y = df.loc[X.index, target]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Select and train model
                if model_name == "Linear Regression":
                    model = LinearRegression()
                elif model_name == "Ridge Regression":
                    model = Ridge(alpha=alpha)
                elif model_name == "Lasso Regression":
                    model = Lasso(alpha=alpha)
                elif model_name == "Random Forest":
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                elif model_name == "Gradient Boosting":
                    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_train_pred = model.predict(X_train_scaled)
                y_test_pred = model.predict(X_test_scaled)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                           scoring='r2')
                
                # Store results
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.results = {
                    'model_name': model_name,
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'y_train_pred': y_train_pred,
                    'y_test_pred': y_test_pred,
                    'cv_scores': cv_scores,
                    'train_r2': r2_score(y_train, y_train_pred),
                    'test_r2': r2_score(y_test, y_test_pred),
                    'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                    'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                    'train_mae': mean_absolute_error(y_train, y_train_pred),
                    'test_mae': mean_absolute_error(y_test, y_test_pred)
                }
                
                st.success("✅ Model training completed!")
                st.balloons()
    else:
        st.info("👈 Please complete feature selection first")

# Tab 5: Performance
with tab5:
    if st.session_state.results is not None:
        st.header("Model Evaluation Results")
        
        results = st.session_state.results
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Test R² Score", f"{results['test_r2']:.2%}")
        with col2:
            st.metric("Test RMSE", f"{results['test_rmse']:.2f}")
        with col3:
            st.metric("Test MAE", f"{results['test_mae']:.2f}")
        with col4:
            st.metric("CV R² Mean", f"{results['cv_scores'].mean():.2%}")
        
        st.markdown("---")
        
        # Cross-validation stability
        st.subheader("Stability Across K-Folds")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(len(results['cv_scores']))],
            y=results['cv_scores'],
            marker_color='steelblue'
        ))
        fig.update_layout(
            yaxis_title="R² Score",
            yaxis_range=[0, 1],
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction vs Actual
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Set: Predicted vs Actual")
            fig = px.scatter(
                x=results['y_train'], 
                y=results['y_train_pred'],
                labels={'x': 'Actual', 'y': 'Predicted'},
                trendline="ols"
            )
            fig.add_trace(go.Scatter(
                x=[results['y_train'].min(), results['y_train'].max()],
                y=[results['y_train'].min(), results['y_train'].max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Test Set: Predicted vs Actual")
            fig = px.scatter(
                x=results['y_test'], 
                y=results['y_test_pred'],
                labels={'x': 'Actual', 'y': 'Predicted'},
                trendline="ols"
            )
            fig.add_trace(go.Scatter(
                x=[results['y_test'].min(), results['y_test'].max()],
                y=[results['y_test'].min(), results['y_test'].max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        # Residuals
        st.subheader("Residual Analysis")
        residuals = results['y_test'] - results['y_test_pred']
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(x=results['y_test_pred'], y=residuals,
                           labels={'x': 'Predicted Values', 'y': 'Residuals'},
                           title='Residual Plot')
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(residuals, nbins=30, 
                             labels={'value': 'Residuals'},
                             title='Residual Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (for tree-based models)
        if results['model_name'] in ['Random Forest', 'Gradient Boosting']:
            st.subheader("Feature Importance")
            importances = st.session_state.model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': st.session_state.selected_features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(feature_importance_df, x='Importance', y='Feature', 
                        orientation='h', title='Feature Importance')
            st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        st.subheader("Export Results")
        results_df = pd.DataFrame({
            'Actual': results['y_test'],
            'Predicted': results['y_test_pred'],
            'Residual': results['y_test'] - results['y_test_pred']
        })
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv,
            file_name="netflix_predictions.csv",
            mime="text/csv"
        )
    else:
        st.info("👈 Please train a model first")

# Tab 6: Predictions
with tab6:
    if st.session_state.model is not None:
        st.header("Make Predictions")
        
        st.subheader("Input Feature Values")
        
        input_data = {}
        cols = st.columns(3)
        
        for i, feature in enumerate(st.session_state.selected_features):
            with cols[i % 3]:
                df = st.session_state.processed_data
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                mean_val = float(df[feature].mean())
                
                input_data[feature] = st.number_input(
                    feature,
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    key=f"input_{feature}"
                )
        
        if st.button("🔮 Predict", type="primary"):
            # Prepare input
            input_df = pd.DataFrame([input_data])
            input_scaled = st.session_state.scaler.transform(input_df)
            
            # Make prediction
            prediction = st.session_state.model.predict(input_scaled)[0]
            
            st.success(f"### Predicted {target}: **${prediction:.2f}**")
            
            # Show confidence interval (for ensemble models)
            if st.session_state.results['model_name'] in ['Random Forest', 'Gradient Boosting']:
                std_error = st.session_state.results['test_rmse']
                st.info(f"95% Confidence Interval: ${prediction - 1.96*std_error:.2f} - ${prediction + 1.96*std_error:.2f}")
        
        st.markdown("---")
        
        # Batch predictions
        st.subheader("Batch Predictions")
        batch_file = st.file_uploader("Upload CSV for batch predictions", type=['csv'])
        
        if batch_file is not None:
            batch_df = pd.read_csv(batch_file)
            
            if all(feat in batch_df.columns for feat in st.session_state.selected_features):
                batch_scaled = st.session_state.scaler.transform(batch_df[st.session_state.selected_features])
                predictions = st.session_state.model.predict(batch_scaled)
                
                batch_df['Predicted_' + target] = predictions
                
                st.dataframe(batch_df, use_container_width=True)
                
                csv = batch_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Batch Predictions",
                    data=csv,
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )
            else:
                st.error("Uploaded file must contain all selected features!")
    else:
        st.info("👈 Please train a model first")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Netflix Stock Prediction Dashboard")
