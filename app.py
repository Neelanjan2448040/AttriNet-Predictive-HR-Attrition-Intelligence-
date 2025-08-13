# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib 
import os 

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="HR Attrition Dashboard",
                   page_icon="💼",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------
# Load dataset & artifacts (models, encoders, etc.)
# ---------------------------
@st.cache_data
def load_data(path="WA_Fn-UseC_-HR-Employee-Attrition.csv"):
    df = pd.read_csv(path)
    # Basic pre-processing just for visualization purposes
    to_drop = [c for c in ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'] if c in df.columns]
    df_vis = df.drop(columns=to_drop)
    return df, df_vis

@st.cache_resource(show_spinner="Loading models and preprocessing tools...")
def load_artifacts(path="model_artifacts.joblib"):
    """Loads pre-trained models, encoders, and scaler from a joblib file."""
    if not os.path.exists(path):
        st.error(f"Model file not found at '{path}'. Please run `train.py` first to generate the model artifacts.")
        st.stop()
    artifacts = joblib.load(path)
    return artifacts

# --- Main App Execution ---
df_full, df_vis = load_data()
artifacts = load_artifacts()

# Unpack the artifacts into individual variables for use in the app
ann_model = artifacts['ann_model']
tabnet_model = artifacts['tabnet_model']
rf_model = artifacts['rf_model']
label_encoders = artifacts['label_encoders']
scaler = artifacts['scaler']
X_prepared_columns = artifacts['X_prepared_columns']

# Prepare the full dataset for evaluation purposes (this is fast)
df_proc = df_full.copy()
for c, le in label_encoders.items():
    if c in df_proc.columns:
        df_proc[c] = le.transform(df_proc[c])
X = df_proc.drop(columns=['Attrition'] + [c for c in ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'] if c in df_proc.columns])
y = df_proc['Attrition']
X_prepared = pd.DataFrame(scaler.transform(X[scaler.get_feature_names_out()]), columns=scaler.get_feature_names_out(), index=X.index)
for col in X.columns:
    if col not in X_prepared.columns:
        X_prepared[col] = X[col]
X_prepared = X_prepared[X_prepared_columns] # Ensure column order is correct


# ---------------------------
# Sidebar navigation & model selection
# ---------------------------
st.sidebar.title("HR Attrition Dashboard")
page = st.sidebar.radio("Navigate", ["Home", "Visualizations", "Prediction", "Model Evaluation"])

st.sidebar.markdown("---")
model_choice = st.sidebar.selectbox("Choose model for prediction", ["ANN (MLP)", "TabNet"])
st.sidebar.caption("Models are pre-trained and loaded at start.")

# ---------------------------
# Helper: create many input widgets for prediction
# ---------------------------
def build_input_form(df_vis):
    st.subheader("Employee details (select values and press Predict)")
    # pick a subset of most important/used features for compact UI, but many inputs included
    cols = df_vis.columns.tolist()
    cols.remove('Attrition')

    # We'll group numeric vs categorical for better widgets
    numeric_cols = df_vis.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_vis.select_dtypes(include=['object']).columns.tolist()

    # Layout: 3 columns across
    input_vals = {}
    grid_cols = st.columns(3)
    # Categorical: show as selectbox with unique values
    # we show the most relevant ones first (common HR features)
    order_cat = ["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","OverTime"]
    order_cat = [c for c in order_cat if c in categorical_cols] + [c for c in categorical_cols if c not in order_cat]
    # Numeric order
    order_num = ["Age","MonthlyIncome","TotalWorkingYears","YearsAtCompany","YearsInCurrentRole",
                 "YearsSinceLastPromotion","YearsWithCurrManager","NumCompaniesWorked","TrainingTimesLastYear",
                 "PercentSalaryHike","DailyRate","HourlyRate","MonthlyRate"]
    order_num = [c for c in order_num if c in numeric_cols] + [c for c in numeric_cols if c not in order_num]

    # Fill categorical fields
    i = 0
    for c in order_cat:
        col = grid_cols[i % 3]
        val = col.selectbox(c, options=sorted(df_vis[c].unique()), index=0)
        input_vals[c] = val
        i += 1

    # Fill numeric fields (use sliders / number inputs)
    j = 0
    for c in order_num:
        col = grid_cols[j % 3]
        mn = int(df_vis[c].min())
        mx = int(df_vis[c].max())
        md = int(df_vis[c].median())
        # use slider for reasonable ranges; use number_input for large spread
        if mx - mn <= 100:
            val = col.slider(c, min_value=mn, max_value=mx, value=md)
        else:
            val = col.number_input(c, value=md, min_value=mn, max_value=mx)
        input_vals[c] = val
        j += 1

    return input_vals

# ---------------------------
# Home Page
# ---------------------------
if page == "Home":
    c1, c2, c3 = st.columns(3)
    c1.metric("Employees", f"{df_full.shape[0]}")
    c2.metric("Features", f"{df_full.shape[1]}")
    attr_rate = df_full['Attrition'].value_counts(normalize=True).get('Yes',0)*100
    c3.metric("Attrition Rate", f"{attr_rate:.1f}%")
    
    st.markdown("---")
    st.header("📄 Dataset Overview")

    # Shape
    st.write(f"**Shape:** {df_full.shape[0]} rows × {df_full.shape[1]} columns")

    # Missing values
    missing_df = df_full.isnull().sum().reset_index()
    missing_df.columns = ['Column', 'Missing Values']
    st.subheader("Missing Values per Column")
    st.dataframe(missing_df, use_container_width=True)

    # Unique values
    unique_df = df_full.nunique().reset_index()
    unique_df.columns = ['Column', 'Unique Values']
    st.subheader("Unique Values per Column")
    st.dataframe(unique_df, use_container_width=True)

    # Summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(df_full.describe(include='all').transpose(), use_container_width=True)

    st.markdown("---")
    st.header("Quick Charts")
    colA, colB = st.columns(2)
    with colA:
        fig = px.histogram(
            df_full, 
            x="Attrition", 
            color="Attrition", 
            title="Attrition Distribution", 
            height=350,
            color_discrete_map={"Yes": "#FF6B6B", "No": "#4ECDC4"}
        )
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        if "MonthlyIncome" in df_full.columns:
            fig = px.box(
                df_full, 
                x="Attrition", 
                y="MonthlyIncome", 
                color="Attrition", 
                title="Monthly Income by Attrition", 
                height=350,
                color_discrete_map={"Yes": "#FF6B6B", "No": "#4ECDC4"}
            )
            fig.update_layout(yaxis=dict(tickformat='$,.0f'))
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Visualizations page
# ---------------------------
elif page == "Visualizations":
    tabs = st.tabs(["Overview", "Demographics", "Compensation", "Role & Work", "Correlation"])
    with tabs[0]:
        st.subheader("Attrition Overview")
        fig = px.histogram(
            df_full, 
            x="Attrition", 
            color="Attrition", 
            title="Employee Attrition Distribution",
            color_discrete_map={"Yes": "#FF6B6B", "No": "#4ECDC4"}
        )
        fig.update_layout(xaxis_title="Attrition Status", yaxis_title="Number of Employees", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Attrition by Gender")
        fig = px.histogram(
            df_full, 
            x="Gender", 
            color="Attrition", 
            barmode="group", 
            title="Gender vs Attrition",
            color_discrete_map={"Yes": "#FF6B6B", "No": "#4ECDC4"}
        )
        fig.update_layout(xaxis_title="Gender", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.subheader("Age distribution by Attrition")
        fig = px.histogram(
            df_full, 
            x="Age", 
            color="Attrition", 
            nbins=25, 
            title="Age Distribution by Attrition Status",
            color_discrete_map={"Yes": "#FF6B6B", "No": "#4ECDC4"},
            opacity=0.8
        )
        fig.update_layout(xaxis_title="Age", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Education Field vs Attrition")
        if "EducationField" in df_full.columns:
            fig = px.histogram(
                df_full, 
                x="EducationField", 
                color="Attrition", 
                barmode="group",
                title="Education Field vs Attrition",
                color_discrete_map={"Yes": "#FF6B6B", "No": "#4ECDC4"}
            )
            fig.update_layout(xaxis_title="Education Field", yaxis_title="Count", xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader("Monthly Income Distribution")
        if "MonthlyIncome" in df_full.columns:
            fig = px.histogram(
                df_full, 
                x="MonthlyIncome", 
                color="Attrition", 
                nbins=30, 
                title="Monthly Income Distribution by Attrition",
                color_discrete_map={"Yes": "#FF6B6B", "No": "#4ECDC4"},
                opacity=0.7
            )
            fig.update_layout(xaxis_title="Monthly Income ($)", yaxis_title="Count", xaxis_tickformat='$,.0f')
            st.plotly_chart(fig, use_container_width=True)

            # Salary bands
            bins = [0, 3000, 6000, 9000, 12000, 15000, 25000]
            labels = ["Very Low","Low","Medium","High","Very High","Top"]
            df_temp = df_full.copy()
            df_temp['SalaryBand'] = pd.cut(df_temp['MonthlyIncome'], bins=bins, labels=labels)
            
            fig = px.histogram(
                df_temp, 
                x="SalaryBand", 
                color="Attrition", 
                barmode="group", 
                title="Attrition by Salary Band",
                color_discrete_map={"Yes": "#FF6B6B", "No": "#4ECDC4"}
            )
            fig.update_layout(xaxis_title="Salary Band", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.subheader("OverTime vs Attrition")
        if "OverTime" in df_full.columns:
            fig = px.histogram(
                df_full, 
                x="OverTime", 
                color="Attrition", 
                barmode="group",
                title="Overtime Work vs Attrition",
                color_discrete_map={"Yes": "#FF6B6B", "No": "#4ECDC4"}
            )
            fig.update_layout(xaxis_title="Works Overtime", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Job Role vs Attrition (Top 10 Roles)")
        if "JobRole" in df_full.columns:
            top_roles = df_full['JobRole'].value_counts().nlargest(10).index
            df_filtered = df_full[df_full['JobRole'].isin(top_roles)]
            
            fig = px.histogram(
                df_filtered, 
                x='JobRole', 
                color='Attrition', 
                barmode='group',
                title="Top Job Roles vs Attrition",
                color_discrete_map={"Yes": "#FF6B6B", "No": "#4ECDC4"}
            )
            fig.update_layout(height=500, xaxis_title="Job Role", yaxis_title="Count", xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Years at Company vs Attrition")
        if "YearsAtCompany" in df_full.columns:
            fig = px.histogram(
                df_full, 
                x="YearsAtCompany", 
                color="Attrition", 
                nbins=20,
                title="Years at Company vs Attrition",
                color_discrete_map={"Yes": "#FF6B6B", "No": "#4ECDC4"},
                opacity=0.7
            )
            fig.update_layout(xaxis_title="Years at Company", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        st.subheader("Correlation heatmap (numeric features)")
        try:
            numeric = df_full.select_dtypes(include=[np.number]).copy()
            # Add encoded Attrition if it's not numeric
            if 'Attrition' not in numeric.columns and 'Attrition' in df_full.columns:
                numeric['Attrition'] = df_full['Attrition'].map({'Yes': 1, 'No': 0})
            
            if numeric.shape[1] > 1:
                corr = numeric.corr()
                
                fig = px.imshow(
                    corr,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    title='Feature Correlation Matrix',
                    aspect="auto"
                )
                
                fig.update_traces(texttemplate="%{z:.2f}", textfont=dict(size=10))
                fig.update_layout(
                    margin=dict(l=80, r=80, t=80, b=80),
                    height=min(800, max(400, len(corr.columns) * 30)),
                    xaxis=dict(tickangle=45, side='bottom'),
                )
                st.plotly_chart(fig, use_container_width=True)
                
                if 'Attrition' in corr.columns:
                    st.subheader("Top Correlations with Attrition")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Positive Correlations:**")
                        pos_corr = corr['Attrition'][corr['Attrition'] > 0].sort_values(ascending=False)[1:6]
                        for feat, val in pos_corr.items():
                            st.write(f"• {feat}: {val:.3f}")
                    
                    with col2:
                        st.write("**Negative Correlations:**")
                        neg_corr = corr['Attrition'][corr['Attrition'] < 0].sort_values()[0:5]
                        for feat, val in neg_corr.items():
                            st.write(f"• {feat}: {val:.3f}")
            else:
                st.warning("Not enough numeric features for correlation analysis.")
                
        except Exception as e:
            st.error(f"Correlation analysis failed: {str(e)}")

# ---------------------------
# Prediction page
# ---------------------------
elif page == "Prediction":
    input_vals = build_input_form(df_vis)
    input_df = pd.DataFrame([input_vals])

    if st.button("🔍 Predict"):
        try:
            with st.spinner("Making prediction..."):
                # Prepare the single row for prediction
                input_encoded = input_df.copy()
                for c, le in label_encoders.items():
                    if c in input_encoded.columns:
                        try:
                            input_encoded[c] = le.transform(input_encoded[c])
                        except ValueError:
                            st.warning(f"Unknown value for {c}: {input_encoded[c].iloc[0]}. Using default.")
                            input_encoded[c] = 0

                # Ensure consistent column order
                row = pd.DataFrame(np.zeros((1, len(X_prepared_columns))), columns=X_prepared_columns)
                for col in input_encoded.columns:
                    if col in row.columns:
                        row.at[0, col] = input_encoded.at[0, col]

                # Scale numeric columns
                num_cols_to_scale = scaler.get_feature_names_out()
                row[num_cols_to_scale] = scaler.transform(row[num_cols_to_scale])

                if model_choice == "ANN (MLP)":
                    pred = ann_model.predict(row)[0]
                    proba = ann_model.predict_proba(row)[0]
                    
                    # Get feature importance
                    try:
                        with st.spinner("Computing feature importance..."):
                            r = permutation_importance(ann_model, X_prepared.sample(min(1000, len(X_prepared))), 
                                                       y.sample(min(1000, len(y))), n_repeats=5, random_state=42)
                            imp = pd.Series(r.importances_mean, index=X_prepared.columns).sort_values(ascending=False)
                    except Exception as e:
                        st.warning(f"Using RandomForest importance as fallback: {str(e)}")
                        imp = pd.Series(rf_model.feature_importances_, index=X_prepared.columns).sort_values(ascending=False)
                else:
                    # TabNet prediction
                    preds = tabnet_model.predict(row.values)
                    pred = int(preds[0])
                    proba = tabnet_model.predict_proba(row.values)[0]
                    
                    # Get TabNet feature importance
                    try:
                        imp_values = tabnet_model.feature_importances_
                        imp = pd.Series(imp_values, index=X_prepared.columns).sort_values(ascending=False)
                    except Exception as e:
                        st.warning(f"Using RandomForest importance as fallback: {str(e)}")
                        imp = pd.Series(rf_model.feature_importances_, index=X_prepared.columns).sort_values(ascending=False)

                # Extract probability safely
                if len(proba) > 1:
                    prob_attrition = float(proba[1])
                else:
                    prob_attrition = float(proba) if pred == 1 else (1.0 - float(proba))
                    
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.error("Please check your input values and try again.")
            st.stop()

        # Show results
        st.subheader("Prediction Result")
        colp, cold = st.columns([2,1])
        with colp:
            # Gauge chart
            try:
                figg = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob_attrition*100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Attrition Probability (%)"},
                    gauge={
                        'axis': {'range': [0,100]},
                        'bar': {'color': "crimson" if pred==1 else "limegreen"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                figg.update_layout(height=300)
                st.plotly_chart(figg, use_container_width=True)
            except Exception as e:
                st.error(f"Gauge chart error: {str(e)}")
                # Fallback display
                st.metric("Attrition Probability", f"{prob_attrition*100:.1f}%")

            # Plain label
            if pred == 1:
                st.error("⚠️ Model Prediction: Employee likely to **LEAVE**")
            else:
                st.success("✅ Model Prediction: Employee likely to **STAY**")

        with cold:
            st.write("**Model used:**", model_choice)
            st.write("**Confidence:**", f"{max(proba):.3f}")
            st.write("**Probabilities:**")
            st.write(f"• Stay: {proba[0]:.3f}")
            st.write(f"• Leave: {proba[1]:.3f}")

        # Top contributing features (global style)
        st.subheader("Top Contributing Features")
        try:
            top_imp = imp.head(8)
            fig_imp = px.bar(
                x=top_imp.values[::-1], 
                y=top_imp.index[::-1], 
                orientation='h',
                labels={'x':'Importance Score','y':'Feature'}, 
                title="Feature Importance (Higher = More Influential)",
                height=400,
                color=top_imp.values[::-1],
                color_continuous_scale='Viridis'
            )
            fig_imp.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.error(f"Feature importance chart error: {str(e)}")
            st.write("**Top Important Features:**")
            for i, (feat, score) in enumerate(imp.head(8).items()):
                st.write(f"{i+1}. **{feat}**: {score:.4f}")
        
        # Debug display (raw input & encoded/scaled row) — kept for transparency
        with st.expander("🔍 Debug Information (Advanced)"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Raw Input Values:**")
                st.dataframe(pd.DataFrame([input_vals]).T.rename(columns={0:"Value"}))
            with col2:
                st.write("**Processed Features (Model Input):**")
                st.dataframe(row.T.rename(columns={0:'Processed Value'}))

# ---------------------------
# Model evaluation page
# ---------------------------
elif page == "Model Evaluation":
    try:
        # prepare test split
        X_train, X_test_split, y_train, y_test_split = train_test_split(X_prepared, y, test_size=0.3, random_state=42)
        
        # Create tabs for different models
        tab1, tab2 = st.tabs(["ANN (MLP)", "TabNet"])
        
        with tab1:
            st.subheader("ANN (MLP) Performance")
            try:
                y_pred_ann = ann_model.predict(X_test_split)
                y_proba_ann = ann_model.predict_proba(X_test_split)
                
                # Classification report
                st.text("Classification Report:")
                st.text(classification_report(y_test_split, y_pred_ann))
                
                # Confusion Matrix
                cm = confusion_matrix(y_test_split, y_pred_ann)
                fig = px.imshow(
                    cm, 
                    text_auto=True, 
                    labels=dict(x="Predicted", y="Actual"), 
                    x=["Stay", "Leave"], 
                    y=["Stay", "Leave"], 
                    color_continuous_scale="Blues",
                    title="ANN Confusion Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"ANN evaluation failed: {str(e)}")

        with tab2:
            st.subheader("TabNet Performance")
            try:
                y_pred_tab = tabnet_model.predict(X_test_split.values)
                
                # Classification report
                st.text("Classification Report:")
                st.text(classification_report(y_test_split, y_pred_tab))
                
                # Confusion Matrix
                cm2 = confusion_matrix(y_test_split, y_pred_tab)
                fig2 = px.imshow(
                    cm2, 
                    text_auto=True, 
                    labels=dict(x="Predicted", y="Actual"), 
                    x=["Stay", "Leave"], 
                    y=["Stay", "Leave"], 
                    color_continuous_scale="Blues",
                    title="TabNet Confusion Matrix"
                )
                st.plotly_chart(fig2, use_container_width=True)
                
            except Exception as e:
                st.error(f"TabNet evaluation failed: {str(e)}")
                
    except Exception as e:
        st.error(f"Model evaluation setup failed: {str(e)}")
        st.error("Please ensure models are trained properly.")
