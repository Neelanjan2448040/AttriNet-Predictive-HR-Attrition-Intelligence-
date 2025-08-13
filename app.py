import os
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"

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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.inspection import permutation_importance
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    st.warning("TabNet not available. Using RandomForest as alternative.")

# ---------------------------
# Page config & lightweight styling
# ---------------------------
st.set_page_config(page_title="HR Attrition — Dark Dashboard",
                   page_icon="💼",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------
# Load dataset
# ---------------------------
@st.cache_data
def load_data(path="WA_Fn-UseC_-HR-Employee-Attrition.csv"):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"Dataset file '{path}' not found. Please upload the HR dataset.")
        return None

df = load_data()

if df is None:
    st.stop()

# Basic pre-processing helper that we use for both training & single predict
@st.cache_resource
def preprocess_and_prepare(df):
    # drop unused constant cols if present
    to_drop = [c for c in ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'] if c in df.columns]
    df_proc = df.drop(columns=to_drop)
    
    # keep original copy for visualizations
    df_vis = df_proc.copy()
    
    # Convert Attrition to binary if it's string
    if df_proc['Attrition'].dtype == 'object':
        df_proc['Attrition'] = df_proc['Attrition'].map({'Yes': 1, 'No': 0})
    
    # label-encode categorical columns and remember encoders
    label_encoders = {}
    categorical_cols = df_proc.select_dtypes(include=['object']).columns
    
    for c in categorical_cols:
        if c != 'Attrition':  # Skip target variable
            le = LabelEncoder()
            df_proc[c] = le.fit_transform(df_proc[c].astype(str))
            label_encoders[c] = le

    # fill na if any (simple)
    df_proc = df_proc.fillna(0)

    # features & target
    X = df_proc.drop(columns=['Attrition'])
    y = df_proc['Attrition']
    
    # standardize numeric features (for ANN)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    
    X_scaled = X.copy()
    if len(numeric_cols) > 0:
        X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    return df_vis, X, X_scaled, y, label_encoders, scaler

df_vis, X_raw, X_prepared, y, label_encoders, scaler = preprocess_and_prepare(df)

# ---------------------------
# Model training (cached)
# ---------------------------
@st.cache_resource(show_spinner=True)
def train_models(X_prepared, y):
    # Split data for consistent training
    X_train, X_test, y_train, y_test = train_test_split(
        X_prepared, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {}
    
    # Train ANN (MLPClassifier as ANN)
    try:
        ann = MLPClassifier(
            hidden_layer_sizes=(64, 32), 
            max_iter=500, 
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        ann.fit(X_train, y_train)
        models['ann'] = ann
    except Exception as e:
        st.error(f"Error training ANN: {e}")
        models['ann'] = None

    # Train TabNet with proper error handling
    if TABNET_AVAILABLE:
        try:
            # Further split training data for TabNet validation
            X_train_tab, X_val_tab, y_train_tab, y_val_tab = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            tabnet = TabNetClassifier(
                seed=42, 
                verbose=0,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                scheduler_params={"step_size":10, "gamma":0.9},
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                mask_type='entmax'
            )
            
            tabnet.fit(
                X_train_tab.values, y_train_tab.values,
                eval_set=[(X_val_tab.values, y_val_tab.values)],
                eval_name=['val'],
                eval_metric=['accuracy'],
                max_epochs=100,
                patience=15,
                batch_size=256,
                virtual_batch_size=64,
                drop_last=False
            )
            models['tabnet'] = tabnet
        except Exception as e:
            st.warning(f"TabNet training failed: {e}")
            models['tabnet'] = None
    else:
        models['tabnet'] = None

    # Train a RandomForest as reliable fallback
    try:
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        models['rf'] = rf
    except Exception as e:
        st.error(f"Error training RandomForest: {e}")
        models['rf'] = None

    return models, X_test, y_test

with st.spinner("Training models (first run — this may take a while)..."):
    models, X_test, y_test = train_models(X_prepared, y)

# ---------------------------
# Sidebar navigation & model selection
# ---------------------------
st.sidebar.title("HR Attrition Dashboard")
page = st.sidebar.radio("Navigate", ["Home", "Visualizations", "Prediction", "Model Evaluation"])

st.sidebar.markdown("---")

# Dynamic model selection based on what's available
available_models = []
if models['ann'] is not None:
    available_models.append("ANN (MLP)")
if models['tabnet'] is not None:
    available_models.append("TabNet")
if models['rf'] is not None:
    available_models.append("Random Forest")

if available_models:
    model_choice = st.sidebar.selectbox("Choose model for prediction", available_models)
else:
    st.sidebar.error("No models available!")
    model_choice = None

st.sidebar.caption("Models are trained at session start (cached).")

# ---------------------------
# Helper: create many input widgets for prediction
# ---------------------------
def build_input_form(df_vis):
    st.subheader("Employee details (select values and press Predict)")
    
    cols = [c for c in df_vis.columns if c != 'Attrition']
    
    # We'll group numeric vs categorical for better widgets
    numeric_cols = df_vis.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df_vis.select_dtypes(include=['object']).columns.tolist() if c != 'Attrition']
    
    # Layout: 3 columns across
    input_vals = {}
    grid_cols = st.columns(3)
    
    # Categorical: show as selectbox with unique values
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
        unique_vals = sorted(df_vis[c].dropna().unique())
        if len(unique_vals) > 0:
            val = col.selectbox(c, options=unique_vals, index=0)
            input_vals[c] = val
        i += 1

    # Fill numeric fields (use sliders / number inputs)
    j = 0
    for c in order_num:
        col = grid_cols[j % 3]
        col_data = df_vis[c].dropna()
        if len(col_data) > 0:
            mn = float(col_data.min())
            mx = float(col_data.max())
            md = float(col_data.median())
            
            # use slider for reasonable ranges; use number_input for large spread
            if mx - mn <= 100 and mn >= 0:
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
    st.title("💼AttriNet– Predictive HR Attrition Intelligence")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Employees", f"{df.shape[0]}")
    c2.metric("Features", f"{df.shape[1]}")
    
    # Calculate attrition rate safely
    if df['Attrition'].dtype == 'object':
        attr_rate = (df['Attrition'] == 'Yes').mean() * 100
    else:
        attr_rate = df['Attrition'].mean() * 100
    c3.metric("Attrition Rate", f"{attr_rate:.1f}%")
    
    st.markdown("---")
    st.header("📄 Dataset Overview")

    # Shape
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    # Missing values
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ['Column', 'Missing Values']
    st.subheader("Missing Values per Column")
    st.dataframe(missing_df, use_container_width=True)

    # Unique values
    unique_df = df.nunique().reset_index()
    unique_df.columns = ['Column', 'Unique Values']
    st.subheader("Unique Values per Column")
    st.dataframe(unique_df, use_container_width=True)

    # Summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(df.describe(include='all').transpose(), use_container_width=True)

    st.markdown("---")
    st.header("Quick charts")
    colA, colB = st.columns(2)
    with colA:
        fig = px.histogram(df, x="Attrition", color="Attrition", title="Attrition distribution", height=350)
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        if "MonthlyIncome" in df.columns:
            fig = px.box(df, x="Attrition", y="MonthlyIncome", color="Attrition", title="Monthly Income by Attrition", height=350)
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Visualizations page (lots of plots)
# ---------------------------
elif page == "Visualizations":
    st.title("📊 Visualizations")
    tabs = st.tabs(["Overview", "Demographics", "Compensation", "Role & Work", "Correlation"])
    
    with tabs[0]:
        st.subheader("Attrition Overview")
        fig = px.histogram(df, x="Attrition", color="Attrition", title="Attrition counts")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Attrition by Gender & Marital Status")
        if "Gender" in df.columns:
            fig = px.histogram(df, x="Gender", color="Attrition", barmode="group", title="Gender vs Attrition")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.subheader("Age distribution by Attrition")
        if "Age" in df.columns:
            fig = px.histogram(df, x="Age", color="Attrition", nbins=25, title="Age distribution (by Attrition)")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Education Field vs Attrition")
        if "EducationField" in df.columns:
            fig = px.histogram(df, x="EducationField", color="Attrition", barmode="group")
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader("Monthly Income distribution")
        if "MonthlyIncome" in df.columns:
            fig = px.histogram(df, x="MonthlyIncome", color="Attrition", nbins=40, title="Monthly income by Attrition")
            st.plotly_chart(fig, use_container_width=True)

            # Salary bands
            try:
                df_temp = df.copy()
                bins = [0, 3000, 6000, 9000, 12000, 15000, df_temp['MonthlyIncome'].max() + 1]
                labels = ["Very Low","Low","Medium","High","Very High","Top"]
                df_temp['SalaryBand'] = pd.cut(df_temp['MonthlyIncome'], bins=bins, labels=labels)
                fig = px.histogram(df_temp, x="SalaryBand", color="Attrition", barmode="group", title="Attrition by Salary Band")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create salary bands: {e}")

    with tabs[3]:
        st.subheader("OverTime vs Attrition")
        if "OverTime" in df.columns:
            fig = px.histogram(df, x="OverTime", color="Attrition", barmode="group")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("JobRole vs Attrition (top roles)")
        if "JobRole" in df.columns:
            top_roles = df['JobRole'].value_counts().nlargest(12).index
            df_filtered = df[df['JobRole'].isin(top_roles)]
            fig = px.histogram(df_filtered, x='JobRole', color='Attrition', barmode='group')
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Years at Company vs Attrition")
        if "YearsAtCompany" in df.columns:
            fig = px.histogram(df, x="YearsAtCompany", color="Attrition", nbins=20)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        st.subheader("Correlation heatmap (numeric features)")
        numeric = df.select_dtypes(include=[np.number]).copy()
        
        # Convert Attrition to numeric if it's categorical
        if 'Attrition' in df.columns and df['Attrition'].dtype == 'object':
            numeric['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        
        if numeric.shape[1] > 1:
            corr = numeric.corr()
            fig = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                title='Correlation heatmap',
                aspect="auto"
            )
            fig.update_traces(texttemplate="%{z:.2f}", textfont=dict(size=10))
            fig.update_layout(
                coloraxis_colorbar=dict(title="Correlation"),
                margin=dict(l=50, r=50, t=50, b=50),
                font=dict(size=10),
                width=800,
                height=800,
                xaxis=dict(tickangle=45)
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Prediction page
# ---------------------------
elif page == "Prediction":
    st.title("🔮 Predict Attrition")
    st.markdown("Fill employee details below. Click **Predict** to get a model prediction, probability, top factors and HR recommendations.")

    if model_choice is None:
        st.error("No models available for prediction!")
        st.stop()

    # Build many inputs dynamically
    input_vals = build_input_form(df_vis)

    # Convert single dict to DataFrame (one row)
    input_df = pd.DataFrame([input_vals])

    # Button to predict
    if st.button("🔍 Predict"):
        try:
            # encode categorical values using label_encoders
            input_encoded = input_df.copy()
            for c, le in label_encoders.items():
                if c in input_encoded.columns:
                    try:
                        input_encoded[c] = le.transform(input_encoded[c].astype(str))
                    except ValueError:
                        # Handle unseen labels
                        input_encoded[c] = 0

            # Create a row with all features from X_prepared
            row = pd.DataFrame(np.zeros((1, X_prepared.shape[1])), columns=X_prepared.columns)
            
            # Fill in the values we have
            for col in input_encoded.columns:
                if col in row.columns:
                    row.at[0, col] = input_encoded.at[0, col]

            # Scale numeric columns
            numeric_cols = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else X_prepared.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 0:
                try:
                    row_num = row[numeric_cols]
                    row_num_scaled = scaler.transform(row_num)
                    row[numeric_cols] = row_num_scaled
                except Exception as e:
                    st.warning(f"Scaling warning: {e}")

            # Choose model and make prediction
            if model_choice == "ANN (MLP)" and models['ann'] is not None:
                pred = models['ann'].predict(row)[0]
                proba = models['ann'].predict_proba(row)[0]
                
                # Get feature importance
                try:
                    r = permutation_importance(models['ann'], X_prepared, y, n_repeats=5, random_state=42, n_jobs=1)
                    imp = pd.Series(r.importances_mean, index=X_prepared.columns).sort_values(ascending=False)
                except Exception:
                    imp = pd.Series(models['rf'].feature_importances_, index=X_prepared.columns).sort_values(ascending=False) if models['rf'] else None
                    
            elif model_choice == "TabNet" and models['tabnet'] is not None:
                pred = int(models['tabnet'].predict(row.values)[0])
                proba = models['tabnet'].predict_proba(row.values)[0]
                
                try:
                    imp_values = models['tabnet'].feature_importances_
                    imp = pd.Series(imp_values, index=X_prepared.columns).sort_values(ascending=False)
                except Exception:
                    imp = pd.Series(models['rf'].feature_importances_, index=X_prepared.columns).sort_values(ascending=False) if models['rf'] else None
                    
            elif model_choice == "Random Forest" and models['rf'] is not None:
                pred = models['rf'].predict(row)[0]
                proba = models['rf'].predict_proba(row)[0]
                imp = pd.Series(models['rf'].feature_importances_, index=X_prepared.columns).sort_values(ascending=False)
            else:
                st.error("Selected model is not available!")
                st.stop()

            # Get probability of attrition (class 1)
            prob_attrition = float(proba[1]) if len(proba) > 1 else float(proba)

            # Show results
            st.subheader("Prediction Result")
            colp, cold = st.columns([2,1])
            
            with colp:
                # Gauge chart
                figg = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_attrition*100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Attrition probability (%)"},
                    gauge={
                        'axis': {'range': [0,100]},
                        'bar': {'color': "crimson" if pred==1 else "limegreen"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                figg.update_layout(height=300)
                st.plotly_chart(figg, use_container_width=True)

                # Plain label
                if pred == 1:
                    st.error("⚠️ Model Prediction: Employee likely to **leave**")
                else:
                    st.success("✅ Model Prediction: Employee likely to **stay**")

            with cold:
                st.write("**Model used:**", model_choice)
                st.write("**Probability (Stay / Leave)**")
                st.write([f"{p:.3f}" for p in proba])

            # Top contributing features (global style)
            if imp is not None:
                st.subheader("Top contributing features (model-driven)")
                top_imp = imp.head(8)
                fig_imp = px.bar(
                    x=top_imp.values[::-1], 
                    y=top_imp.index[::-1], 
                    orientation='h',
                    labels={'x':'Importance','y':'Feature'}, 
                    height=350,
                    title="Feature Importance"
                )
                st.plotly_chart(fig_imp, use_container_width=True)

            # Debug display (raw input & encoded/scaled row) — kept for transparency
            if st.checkbox("Show raw input & encoded (debug)"):
                st.write("**Raw Input:**")
                st.write(pd.DataFrame([input_vals]).T.rename(columns={0:"value"}))
                st.write("**Encoded/scaled features passed to model:**")
                st.write(row.T.rename(columns={0:'value'}))

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e)

# ---------------------------
# Model evaluation page
# ---------------------------
elif page == "Model Evaluation":
    st.title("🧪 Model Evaluation")

    if X_test is None or y_test is None:
        st.error("No test data available for evaluation!")
        st.stop()

    st.markdown("Classification reports and confusion matrices for available models on hold-out test split.")

    # Evaluate ANN
    if models['ann'] is not None:
        try:
            y_pred_ann = models['ann'].predict(X_test)
            acc_ann = accuracy_score(y_test, y_pred_ann)
            
            st.subheader(f"ANN (MLP) — Accuracy: {acc_ann:.3f}")
            st.text(classification_report(y_test, y_pred_ann))
            
            st.subheader("Confusion Matrix (ANN)")
            cm = confusion_matrix(y_test, y_pred_ann)
            fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"), 
                           x=["Stay","Leave"], y=["Stay","Leave"], color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"ANN evaluation error: {e}")

    # Evaluate TabNet
    if models['tabnet'] is not None:
        try:
            y_pred_tab = models['tabnet'].predict(X_test.values)
            acc_tab = accuracy_score(y_test, y_pred_tab)
            
            st.subheader(f"TabNet — Accuracy: {acc_tab:.3f}")
            st.text(classification_report(y_test, y_pred_tab))
            
            st.subheader("Confusion Matrix (TabNet)")
            cm2 = confusion_matrix(y_test, y_pred_tab)
            fig2 = px.imshow(cm2, text_auto=True, labels=dict(x="Predicted", y="Actual"), 
                            x=["Stay","Leave"], y=["Stay","Leave"], color_continuous_scale="Blues")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"TabNet evaluation error: {e}")

    # Evaluate Random Forest
    if models['rf'] is not None:
        try:
            y_pred_rf = models['rf'].predict(X_test)
            acc_rf = accuracy_score(y_test, y_pred_rf)
            
            st.subheader(f"Random Forest — Accuracy: {acc_rf:.3f}")
            st.text(classification_report(y_test, y_pred_rf))
            
            st.subheader("Confusion Matrix (Random Forest)")
            cm3 = confusion_matrix(y_test, y_pred_rf)
            fig3 = px.imshow(cm3, text_auto=True, labels=dict(x="Predicted", y="Actual"), 
                            x=["Stay","Leave"], y=["Stay","Leave"], color_continuous_scale="Blues")
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.error(f"Random Forest evaluation error: {e}")

    # Model comparison
    st.subheader("Model Comparison")
    comparison_data = []
    if models['ann'] is not None:
        try:
            y_pred_ann = models['ann'].predict(X_test)
            comparison_data.append({"Model": "ANN", "Accuracy": accuracy_score(y_test, y_pred_ann)})
        except:
            pass
    
    if models['tabnet'] is not None:
        try:
            y_pred_tab = models['tabnet'].predict(X_test.values)
            comparison_data.append({"Model": "TabNet", "Accuracy": accuracy_score(y_test, y_pred_tab)})
        except:
            pass
    
    if models['rf'] is not None:
        try:
            y_pred_rf = models['rf'].predict(X_test)
            comparison_data.append({"Model": "Random Forest", "Accuracy": accuracy_score(y_test, y_pred_rf)})
        except:
            pass
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        fig_comp = px.bar(comp_df, x="Model", y="Accuracy", title="Model Accuracy Comparison")
        st.plotly_chart(fig_comp, use_container_width=True)
        st.dataframe(comp_df, use_container_width=True)
