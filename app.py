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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from pytorch_tabnet.tab_model import TabNetClassifier

# ---------------------------
# Page config & lightweight styling
# ---------------------------
st.set_page_config(page_title="HR Attrition — Dark Dashboard",
                   page_icon="💼",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Minimal in-app dark styling (works without config.toml)
st.markdown(
    """
    <style>
    :root {
        --primary-color: #00FFAA;
        --background-color: #0E1117;
        --secondary-background-color: #262730;
        --text-color: #FAFAFA;
        --font: 'sans serif';
    }
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
        font-family: var(--font);
    }
    .css-1y4p8pa { background-color: var(--background-color); }
    .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .stButton>button {
        background-color: var(--primary-color);
        color: black;
        border-radius: 8px;
    }
    .stMetric > div {
        color: var(--text-color);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Load dataset
# ---------------------------
@st.cache_data
def load_data(path="WA_Fn-UseC_-HR-Employee-Attrition.csv"):
    df = pd.read_csv(path)
    return df

df = load_data()

# Basic pre-processing helper that we use for both training & single predict
@st.cache_resource
def preprocess_and_prepare(df):
    # drop unused constant cols if present
    to_drop = [c for c in ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'] if c in df.columns]
    df_proc = df.drop(columns=to_drop)
    # keep original copy for visualizations
    df_vis = df_proc.copy()

    # label-encode categorical columns and remember encoders
    label_encoders = {}
    for c in df_proc.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_proc[c] = le.fit_transform(df_proc[c])
        label_encoders[c] = le

    # fill na if any (simple)
    df_proc = df_proc.fillna(0)

    # features & target
    X = df_proc.drop(columns=['Attrition'])
    y = df_proc['Attrition']
    # standardize numeric features (for ANN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))
    X_scaled = pd.DataFrame(X_scaled, columns=X.select_dtypes(include=[np.number]).columns, index=X.index)
    # merge scaled numeric with encoded categoricals (already numeric)
    X_prepared = X.copy()
    X_prepared[X_scaled.columns] = X_scaled

    return df_vis, X, X_prepared, y, label_encoders, scaler

df_vis, X_raw, X_prepared, y, label_encoders, scaler = preprocess_and_prepare(df)

# ---------------------------
# Model training (cached)
# ---------------------------
@st.cache_resource(show_spinner=True)
def train_models(X_prepared, y):
    # Train ANN (MLPClassifier as ANN)
    ann = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    ann.fit(X_prepared, y)

    # Create train/validation split for TabNet
    X_train_tab, X_val_tab, y_train_tab, y_val_tab = train_test_split(
        X_prepared.values, y.values, test_size=0.2, random_state=42
    )

    # Train TabNet with early stopping
    tabnet = TabNetClassifier(seed=42, verbose=0)
    tabnet.fit(
        X_train_tab, y_train_tab,
        eval_set=[(X_val_tab, y_val_tab)],
        eval_name=['val'],
        eval_metric=['accuracy'],
        max_epochs=50,
        patience=10,
        batch_size=256,
        virtual_batch_size=64
    )

    # Train a RandomForest for quick global importances (helpful fallback)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_prepared, y)

    return ann, tabnet, rf

with st.spinner("Training models (first run — this may take a while)..."):
    ann_model, tabnet_model, rf_model = train_models(X_prepared, y)

# ---------------------------
# Sidebar navigation & model selection
# ---------------------------
st.sidebar.title("HR Attrition Dashboard")
page = st.sidebar.radio("Navigate", ["Home", "Visualizations", "Prediction", "Model Evaluation"])

st.sidebar.markdown("---")
model_choice = st.sidebar.selectbox("Choose model for prediction", ["ANN (MLP)", "TabNet"])
st.sidebar.caption("Models are trained at session start (cached).")

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
    st.title("💼AttriNet– Predictive HR Attrition Intelligence ")
    # removed "What this app does" section per request
    c1, c2, c3 = st.columns(3)
    c1.metric("Employees", f"{df.shape[0]}")
    c2.metric("Features", f"{df.shape[1]}")
    attr_rate = df['Attrition'].value_counts(normalize=True).get('Yes',0)*100
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
        fig = px.histogram(df, x="Gender", color="Attrition", barmode="group", title="Gender vs Attrition")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.subheader("Age distribution by Attrition")
        fig = px.histogram(df, x="Age", color="Attrition", nbins=25, title="Age distribution (by Attrition)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Education Field vs Attrition")
        if "EducationField" in df.columns:
            fig = px.histogram(df, x="EducationField", color="Attrition", barmode="group")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader("Monthly Income distribution")
        if "MonthlyIncome" in df.columns:
            fig = px.histogram(df, x="MonthlyIncome", color="Attrition", nbins=40, title="Monthly income by Attrition")
            st.plotly_chart(fig, use_container_width=True)

            # Salary bands
            bins = [0, 3000, 6000, 9000, 12000, 15000, 25000]
            labels = ["Very Low","Low","Medium","High","Very High","Top"]
            df['SalaryBand'] = pd.cut(df['MonthlyIncome'], bins=bins, labels=labels)
            fig = px.histogram(df, x="SalaryBand", color="Attrition", barmode="group", title="Attrition by Salary Band")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.subheader("OverTime vs Attrition")
        if "OverTime" in df.columns:
            fig = px.histogram(df, x="OverTime", color="Attrition", barmode="group")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("JobRole vs Attrition (top roles)")
        if "JobRole" in df.columns:
            top_roles = df['JobRole'].value_counts().nlargest(12).index
            fig = px.histogram(df[df['JobRole'].isin(top_roles)], x='JobRole', color='Attrition', barmode='group')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Years at Company vs Attrition")
        if "YearsAtCompany" in df.columns:
            fig = px.histogram(df, x="YearsAtCompany", color="Attrition", nbins=20)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        st.subheader("Correlation heatmap (numeric features)")
        numeric = df.select_dtypes(include=[np.number]).copy()
        # if Attrition encoded as object, convert for corr
        if numeric.shape[1] > 1:
            corr = numeric.corr()
            fig = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,  ##
                title='Correlation heatmap'
            )
            # make annotation text clear and larger
            fig.update_traces(texttemplate="%{z:.2f}", textfont=dict(size=14))
            fig.update_layout(coloraxis_colorbar=dict(title="Correlation"),
                              margin=dict(l=50, r=50, t=50, b=50),
                              font=dict(size=14),
                              width=900,
                              height=900,
                              xaxis=dict(tickangle=45))
                              
            st.plotly_chart(fig, use_container_width=False)

# ---------------------------
# Prediction page
# ---------------------------
elif page == "Prediction":
    st.title("🔮 Predict Attrition")
    st.markdown("Fill employee details below. Click **Predict** to get a model prediction, probability, top factors and HR recommendations.")

    # Build many inputs dynamically
    input_vals = build_input_form(df_vis)

    # Convert single dict to DataFrame (one row)
    input_df = pd.DataFrame([input_vals])

    # Button to predict
    if st.button("🔍 Predict"):
        # encode categorical values using label_encoders
        input_encoded = input_df.copy()
        for c, le in label_encoders.items():
            if c in input_encoded.columns:
                # if user value is unseen, transform via adding to classes (safe fallback)
                try:
                    input_encoded[c] = le.transform(input_encoded[c])
                except Exception:
                    input_encoded[c] = 0

        # Use the prepared X_prepared's column ordering:
        row = pd.DataFrame(np.zeros((1, X_prepared.shape[1])), columns=X_prepared.columns)
        for col in input_encoded.columns:
            if col in row.columns:
                row.at[0, col] = input_encoded.at[0, col]

        # For numeric columns, scale using stored scaler
        num_cols = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else X_prepared.select_dtypes(include=[np.number]).columns.tolist()
        try:
            row_num = row[num_cols]
            row_num_scaled = scaler.transform(row_num)
            row.loc[:, num_cols] = row_num_scaled
        except Exception:
            # fallback: leave as-is
            pass

        # Choose model
        if model_choice == "ANN (MLP)":
            pred = ann_model.predict(row)[0]
            proba = ann_model.predict_proba(row)[0]
            with st.spinner("Computing permutation importances for ANN (fast approximation)..."):
                try:
                    r = permutation_importance(ann_model, X_prepared, y, n_repeats=8, random_state=42, n_jobs=-1)
                    imp = pd.Series(r.importances_mean, index=X_prepared.columns).sort_values(ascending=False)
                except Exception:
                    imp = pd.Series(rf_model.feature_importances_, index=X_prepared.columns).sort_values(ascending=False)
        else:
            preds = tabnet_model.predict(row.values)
            pred = int(preds[0])
            proba = tabnet_model.predict_proba(row.values)[0]
            try:
                imp_values = tabnet_model.feature_importances_
                imp = pd.Series(imp_values, index=X_prepared.columns).sort_values(ascending=False)
            except Exception:
                imp = pd.Series(rf_model.feature_importances_, index=X_prepared.columns).sort_values(ascending=False)

        prob_attrition = float(proba[1]) if len(proba) > 1 else (float(proba) if isinstance(proba, (list, np.ndarray)) else 0.0)

        # Show results
        st.subheader("Prediction Result")
        colp, cold = st.columns([2,1])
        with colp:
            # Gauge chart
            figg = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob_attrition*100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Attrition probability (%)"},
                gauge={'axis': {'range': [0,100]},
                       'bar': {'color': "crimson" if pred==1 else "limegreen"}}
            ))
            st.plotly_chart(figg, use_container_width=True, height=300)

            # Plain label
            if pred == 1:
                st.error("⚠️ Model Prediction: Employee likely to **leave**")
            else:
                st.success("✅ Model Prediction: Employee likely to **stay**")

        with cold:
            st.write("**Model used:**", model_choice)
            st.write("**Probability (No / Yes)**")
            st.write([f"{p:.3f}" for p in proba])

        # Top contributing features (global style)
        st.subheader("Top contributing features (model-driven)")
        top_imp = imp.head(8)
        fig_imp = px.bar(x=top_imp.values[::-1], y=top_imp.index[::-1], orientation='h',
                         labels={'x':'Importance','y':'Feature'}, height=350)
        st.plotly_chart(fig_imp, use_container_width=True)

        # Debug display (raw input & encoded/scaled row) — kept for transparency
        if st.checkbox("Show raw input & encoded (debug)"):
            st.write(pd.DataFrame([input_vals]).T.rename(columns={0:"value"}))
            st.write("Encoded/scaled features passed to model:")
            st.write(row.T.rename(columns={0:'value'}))

# ---------------------------
# Model evaluation page
# ---------------------------
elif page == "Model Evaluation":
    st.title("🧪 Model Evaluation")

    st.markdown("We show classification reports and confusion matrices for both models (ANN & TabNet) on hold-out test split.")

    # prepare test split
    X_train, X_test_split, y_train, y_test_split = train_test_split(X_prepared, y, test_size=0.3, random_state=42)
    # ANN
    y_pred_ann = ann_model.predict(X_test_split)
    y_proba_ann = ann_model.predict_proba(X_test_split)

    # TabNet
    y_pred_tab = tabnet_model.predict(X_test_split.values)
    # generate classification report for each
    st.subheader("ANN (MLP) — metrics")
    st.text(classification_report(y_test_split, y_pred_ann))
    st.subheader("Confusion Matrix (ANN)")
    cm = confusion_matrix(y_test_split, y_pred_ann)
    fig = px.imshow(cm, text_auto=True, labels=dict(x="Pred", y="True"), x=["No","Yes"], y=["No","Yes"], color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("TabNet — metrics")
    st.text(classification_report(y_test_split, y_pred_tab))
    st.subheader("Confusion Matrix (TabNet)")
    cm2 = confusion_matrix(y_test_split, y_pred_tab)
    fig2 = px.imshow(cm2, text_auto=True, labels=dict(x="Pred", y="True"), x=["No","Yes"], y=["No","Yes"], color_continuous_scale="Blues")
    st.plotly_chart(fig2, use_container_width=True)



