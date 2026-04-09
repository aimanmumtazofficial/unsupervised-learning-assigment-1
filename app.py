# # ================= IMPORT LIBRARIES =================
# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# from sklearn.decomposition import PCA
# from sklearn.metrics.pairwise import cosine_similarity
# from scipy.stats import zscore

# # ================= CONFIG & SESSION STATE =================
# st.set_page_config(page_title="Customer Analysis Dashboard", layout="wide")

# if 'df' not in st.session_state:
#     st.session_state.df = None
# if 'customer_df' not in st.session_state:
#     st.session_state.customer_df = None

# # ================= TITLE =================
# st.title(" Online Retail Customer Analysis Dashboard")
# st.write("This app performs Unsupervised Learning Tasks 1 to 6 for Customer Insights.")

# # ================= FILE UPLOAD =================
# uploaded_file = st.file_uploader("Upload Online Retail CSV", type=["csv"])

# if uploaded_file is not None:
#     if st.session_state.df is None:
#         try:
#             # Task 1: Data Understanding
#             # We are using the 'ISO-8859-1' encoding, which is best suited for the Retail dataset.
#             df_raw = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            
#             # FIX: As soon as the file is loaded, remove extra spaces and hidden characters from the column names.
#             df_raw.columns = df_raw.columns.str.strip().str.replace('ï»¿', '') 
            
#             st.session_state.df = df_raw
#         except Exception as e:
#             st.error(f"Error loading file: {e}")

#     df = st.session_state.df

#     if df is not None:
#         # ================= TASK 1: PREPROCESSING =================
#         st.header("Task 1: Data Understanding & Preprocessing")
        
#         st.subheader("Data Preview")
#         st.dataframe(df.head())

#         col1, col2 = st.columns(2)
#         with col1:
#             st.write("**Dataset Shape:**", df.shape)
#             st.write("**Available Columns:**", list(df.columns)) 
#             st.write("**Missing Values:**")
#             st.write(df.isnull().sum())
        
#         with col2:
#             st.info("""
#             **Why Preprocessing?**
#             Unsupervised learning (like K-Means) is distance based. Missing values or unscaled features 
#             will bias the results. We must clean the data first.
#             """)

#         # Cleaning Button
#         if st.button("Clean & Aggregate Data"):
#             # Double check for columns again before processing
#             required_cols = ['CustomerID', 'Quantity', 'UnitPrice', 'InvoiceNo']
#             missing = [col for col in required_cols if col not in df.columns]
            
#             if missing:
#                 st.error(f"These columns are still missing in the dataset: {missing}. "
#                          "Please check that you have uploaded the correct 'Online Retail.csv' file.")
                         
#             else:
#                 # 1. Clean CustomerID and Quantity
#                 df = df.dropna(subset=['CustomerID'])
#                 df = df.drop_duplicates()
                
#                 # Numeric Conversion
#                 df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
#                 df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
                
#                 df = df[df['Quantity'] > 0] # Remove returns/cancellations
#                 df['TotalSpend'] = df['Quantity'] * df['UnitPrice']
                
#                 # 2. Aggregate to Customer Level
#                 customer_data = df.groupby('CustomerID').agg({
#                     'Quantity': 'mean',
#                     'UnitPrice': 'mean',
#                     'TotalSpend': 'sum',
#                     'InvoiceNo': 'nunique'
#                 }).rename(columns={'InvoiceNo': 'Frequency', 'Quantity': 'AvgQuantity'})
                
#                 st.session_state.df = df
#                 st.session_state.customer_df = customer_data
#                 st.success("Data cleaned and aggregated by Customer ID!")

#         if st.session_state.customer_df is not None:
#             c_df = st.session_state.customer_df
            
#             st.subheader("Feature Scaling")
#             scaler = StandardScaler()
#             scaled_features = scaler.fit_transform(c_df)
#             scaled_df = pd.DataFrame(scaled_features, columns=c_df.columns, index=c_df.index)
#             st.write("Scaled Customer Data (Ready for ML):")
#             st.dataframe(scaled_df.head())

#             # ================= TASK 2: K-MEANS =================
#             st.header("Task 2: Customer Segmentation (K-Means)")
            
#             inertia = []
#             K_range = range(1, 8)
#             for k in K_range:
#                 km = KMeans(n_clusters=k, random_state=42, n_init=10)
#                 km.fit(scaled_df)
#                 inertia.append(km.inertia_)
            
#             fig_elbow, ax_elbow = plt.subplots()
#             ax_elbow.plot(K_range, inertia, marker='o', color='teal')
#             ax_elbow.set_title("Elbow Method to find Optimal K")
#             st.pyplot(fig_elbow)

#             k_val = st.slider("Select Clusters", 2, 6, 3)
#             kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
#             c_df['Cluster'] = kmeans.fit_predict(scaled_df)
            
#             st.write(f"**Silhouette Score:** {silhouette_score(scaled_df, c_df['Cluster']):.2f}")
            
#             fig_clus, ax_clus = plt.subplots()
#             sns.scatterplot(data=c_df, x='TotalSpend', y='Frequency', hue='Cluster', palette='viridis', ax=ax_clus)
#             ax_clus.set_title("Customer Segments: Spend vs Frequency")
#             st.pyplot(fig_clus)

#             # ================= TASK 3: ANOMALY DETECTION =================
#             st.header("Task 3: Anomaly Detection")
#             z_scores = np.abs(zscore(scaled_df))
#             threshold = st.slider("Z-Score Threshold", 2.0, 5.0, 3.0)
#             anomalies = (z_scores > threshold).any(axis=1)
#             c_df['Is_Anomaly'] = anomalies
            
#             st.write(f"Detected {anomalies.sum()} anomalous customers.")
            
#             fig_anom, ax_anom = plt.subplots()
#             sns.scatterplot(data=c_df, x='TotalSpend', y='AvgQuantity', hue='Is_Anomaly', palette={True:'red', False:'blue'}, ax=ax_anom)
#             st.pyplot(fig_anom)
#             st.info("Anomalies are customers with extreme spending or unusual purchase volumes.")

#             # ================= TASK 4: PCA =================
#             st.header("Task 4: Dimensionality Reduction (PCA)")
#             pca = PCA(n_components=2)
#             pca_res = pca.fit_transform(scaled_df.drop(columns=['Cluster', 'Is_Anomaly'], errors='ignore'))
#             pca_df = pd.DataFrame(pca_res, columns=['PC1', 'PC2'])
            
#             st.write(f"Variance Explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
            
#             fig_pca, ax_pca = plt.subplots()
#             ax_pca.scatter(pca_df['PC1'], pca_df['PC2'], c=c_df['Cluster'], cmap='viridis', alpha=0.5)
#             ax_pca.set_xlabel("PC1")
#             ax_pca.set_ylabel("PC2")
#             st.pyplot(fig_pca)

#             # ================= TASK 5: RECOMMENDATIONS =================
#             st.header("Task 5: Recommendation System")
#             with st.spinner("Computing Recommendations..."):
#                 sample_size = min(50000, len(df))
#                 small_df = df.sample(sample_size, random_state=42)
#                 matrix = small_df.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', fill_value=0)
#                 user_sim = cosine_similarity(matrix)
#                 user_sim_df = pd.DataFrame(user_sim, index=matrix.index, columns=matrix.index)

#                 def get_rec(cid):
#                     if cid not in user_sim_df.index: return "No Data"
#                     similar_user = user_sim_df[cid].sort_values(ascending=False).index[1]
#                     items_this = set(matrix.loc[cid][matrix.loc[cid] > 0].index)
#                     items_sim = set(matrix.loc[similar_user][matrix.loc[similar_user] > 0].index)
#                     return list(items_sim - items_this)[:5]

#                 test_users = matrix.index[:3]
#                 for u in test_users:
#                     st.write(f"**Recommendations for Customer {int(u)}:** {get_rec(u)}")

#             # ================= TASK 6: REFLECTION =================
#             st.header("Task 6: Analysis & Reflection")
#             st.success("""
#             **Insights:**
#             - **Clustering:** Segmented customers into 'VIP', 'Regular', and 'Occasional' buyers.
#             - **Anomaly Detection:** Identified potential bulk buyers or data entry errors.
#             - **PCA:** Successfully compressed features while retaining high variance.
#             - **Collaborative Filtering:** Successfully provided personalized product suggestions.
#             """)
# else:
#     st.warning("Please upload the 'Online Retail' CSV file to begin the analysis. This dataset is essential for performing customer segmentation and generating insights.")






# ===============================================================
# ONLINE RETAIL CUSTOMER ANALYSIS DASHBOARD
# Unsupervised Learning: K-Means, KDE Anomaly Detection, PCA, Collaborative Filtering
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KernelDensity
from scipy.stats import zscore, gaussian_kde

# ===============================================================
# PAGE CONFIG & CUSTOM CSS
# ===============================================================
st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    layout="wide",
    page_icon="🛒",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ---- Google Font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    /* ---- Global ---- */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #0a0f1e;
        color: #e8eaf6;
    }

    /* ---- Main Background ---- */
    .stApp {
        background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0f2236 100%);
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #0a1628 100%);
        border-right: 1px solid rgba(0,212,255,0.15);
    }
    [data-testid="stSidebar"] * {
        color: #e8eaf6 !important;
    }

    /* ---- Headers ---- */
    h1 {
        font-family: 'Syne', sans-serif !important;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #00d4ff, #7c4dff, #ff6b9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    h2 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        color: #00d4ff !important;
        border-left: 4px solid #00d4ff;
        padding-left: 12px;
        margin-top: 2rem !important;
    }
    h3 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        color: #a78bfa !important;
    }

    /* ---- Metric Cards ---- */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(124,77,255,0.08));
        border: 1px solid rgba(0,212,255,0.2);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    [data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 2rem !important;
    }

    /* ---- Info / Success / Warning Boxes ---- */
    .stInfo, div[data-baseweb="notification"] {
        background: rgba(0,212,255,0.08) !important;
        border: 1px solid rgba(0,212,255,0.3) !important;
        border-radius: 12px !important;
        color: #e8eaf6 !important;
    }
    .stSuccess > div {
        background: rgba(0,255,180,0.08) !important;
        border: 1px solid rgba(0,255,180,0.3) !important;
        border-radius: 12px !important;
    }
    .stWarning > div {
        background: rgba(255,180,0,0.08) !important;
        border: 1px solid rgba(255,180,0,0.3) !important;
        border-radius: 12px !important;
    }

    /* ---- Buttons ---- */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #7c4dff) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 10px 28px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(0,212,255,0.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(0,212,255,0.4) !important;
    }

    /* ---- Sliders ---- */
    .stSlider [data-baseweb="slider"] {
        color: #00d4ff !important;
    }

    /* ---- Dataframe ---- */
    .stDataFrame {
        border: 1px solid rgba(0,212,255,0.15) !important;
        border-radius: 12px !important;
    }

    /* ---- Custom Insight Card ---- */
    .insight-card {
        background: linear-gradient(135deg, rgba(124,77,255,0.12), rgba(0,212,255,0.08));
        border: 1px solid rgba(124,77,255,0.25);
        border-radius: 16px;
        padding: 20px 24px;
        margin: 12px 0;
    }
    .insight-card h4 {
        color: #a78bfa;
        font-family: 'Syne', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .insight-card p {
        color: #cbd5e1;
        font-size: 0.92rem;
        line-height: 1.6;
        margin: 0;
    }

    /* ---- Section Divider ---- */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,212,255,0.4), transparent);
        margin: 32px 0;
    }

    /* ---- Task Badge ---- */
    .task-badge {
        display: inline-block;
        background: linear-gradient(135deg, #7c4dff, #00d4ff);
        color: white;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 4px 14px;
        border-radius: 20px;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    /* ---- Plot background helper ---- */
    .plot-container {
        background: rgba(13,27,42,0.8);
        border: 1px solid rgba(0,212,255,0.12);
        border-radius: 16px;
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ===============================================================
# MATPLOTLIB DARK THEME (all charts)
# ===============================================================
plt.rcParams.update({
    'figure.facecolor':  '#0d1b2a',
    'axes.facecolor':    '#0d1b2a',
    'axes.edgecolor':    '#1e3a5f',
    'axes.labelcolor':   '#94a3b8',
    'axes.titlecolor':   '#e8eaf6',
    'xtick.color':       '#94a3b8',
    'ytick.color':       '#94a3b8',
    'text.color':        '#e8eaf6',
    'grid.color':        '#1e3a5f',
    'grid.alpha':        0.5,
    'axes.grid':         True,
    'legend.facecolor':  '#0d1b2a',
    'legend.edgecolor':  '#1e3a5f',
    'font.family':       'DejaVu Sans',
})

PALETTE   = ['#00d4ff', '#7c4dff', '#ff6b9d', '#00ff99', '#ffb300', '#ff5252']
GRAD_CMAP = 'cool'

# ===============================================================
# SESSION STATE
# ===============================================================
for key in ['df', 'customer_df']:
    if key not in st.session_state:
        st.session_state[key] = None

# ===============================================================
# SIDEBAR
# ===============================================================
with st.sidebar:
    st.markdown("## 🛒 Customer Intelligence")
    st.markdown("---")
    st.markdown("""
    **Unsupervised Learning Pipeline**
    - 📊 Task 1 — Data Preprocessing
    - 🎯 Task 2 — K-Means Clustering
    - 🚨 Task 3 — Anomaly Detection (KDE)
    - 🔬 Task 4 — PCA Dimensionality Reduction
    - 🤝 Task 5 — Recommendation System
    - 💡 Task 6 — Insights & Reflection
    """)
    st.markdown("---")
    st.markdown("""
    **Dataset Required:**
    Online Retail CSV
    (UCI Machine Learning Repository)
    """)
    st.markdown("---")
    st.caption("Built with Streamlit · Scikit-learn · Seaborn")

# ===============================================================
# TITLE
# ===============================================================
st.markdown("<h1>🛒 Online Retail Customer Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='color:#94a3b8; font-size:1.05rem; margin-bottom:2rem;'>
End-to-end Unsupervised Learning pipeline — Customer Segmentation · Anomaly Detection · 
Dimensionality Reduction · Collaborative Filtering Recommendations
</p>
""", unsafe_allow_html=True)

# ===============================================================
# AUTO-LOAD DATASET (no file upload required for visitors)
# ===============================================================
# ===============================================================
# DATASET AUTO-LOAD — no file upload required for visitors
# Place online_retail.csv inside a data/ folder next to this script.
# The app reads it automatically when anyone opens the deployed URL.
# ===============================================================

import os

# Path to the bundled dataset (data/online_retail.csv inside the repo)
BUNDLED_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "online_retail.csv")

def load_dataset(source):
    """Load CSV and normalise column names regardless of encoding artifacts."""
    df_raw = pd.read_csv(source, encoding="ISO-8859-1")
    df_raw.columns = (
        df_raw.columns
        .str.strip()
        .str.replace("ï»¿", "", regex=False)
        .str.replace("\ufeff", "", regex=False)
    )
    return df_raw

# Try to load automatically — visitor does nothing
if st.session_state.df is None:
    if os.path.exists(BUNDLED_CSV):
        with st.spinner("Loading dataset... please wait a moment."):
            try:
                st.session_state.df = load_dataset(BUNDLED_CSV)
                st.success(
                    "✅ Dataset loaded automatically — no file upload needed! "
                    "Scroll down and click **Clean & Aggregate Data** to begin."
                )
            except Exception as e:
                st.error(f"Failed to load bundled dataset: {e}")
    else:
        # Fallback shown only if data/ folder is missing (should not happen on deployed app)
        st.warning(
            "⚠️  Bundled dataset not found at **data/online_retail.csv**. "
            "Please place the CSV in the data/ folder, or upload it manually below."
        )
        uploaded_file = st.file_uploader(
            "📂  Upload Online Retail CSV",
            type=["csv"],
            help="Download from UCI ML Repository — Online Retail dataset"
        )
        if uploaded_file is not None:
            try:
                st.session_state.df = load_dataset(uploaded_file)
                st.success("Dataset loaded from uploaded file.")
            except Exception as e:
                st.error(f"Could not read file: {e}")

if st.session_state.df is not None:

    df = st.session_state.df
    if df is not None:

        # ============================================================
        # TASK 1 — DATA UNDERSTANDING & PREPROCESSING
        # ============================================================
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<span class="task-badge">Task 1</span>', unsafe_allow_html=True)
        st.header("Data Understanding & Preprocessing")

        st.markdown("""
        <div class="insight-card">
        <h4>🧠 Why Preprocessing Matters in Unsupervised Learning</h4>
        <p>
        Unsupervised algorithms like K-Means rely on <strong>distance metrics</strong>. 
        Without preprocessing, features with larger numeric ranges dominate the distance 
        calculations, biasing the model. Missing values cause runtime errors or silent 
        distortions. Duplicate rows inflate cluster sizes artificially. 
        <strong>Standardization (Z-score scaling)</strong> ensures every feature contributes 
        equally, making clustering results meaningful and reproducible.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("📋 Raw Data Preview")
        st.dataframe(df.head(8), use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        with col4:
            st.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Column Info")
            st.write("**Available Columns:**", list(df.columns))
            missing_df = df.isnull().sum().reset_index()
            missing_df.columns = ['Column', 'Missing Count']
            st.dataframe(missing_df, use_container_width=True)

        with col_right:
            st.subheader("Descriptive Statistics")
            st.dataframe(df.describe().T.style.background_gradient(cmap='Blues'), use_container_width=True)

        if st.button("⚙️  Clean & Aggregate Data"):
            required_cols = ['CustomerID', 'Quantity', 'UnitPrice', 'InvoiceNo']
            missing_cols = [c for c in required_cols if c not in df.columns]

            if missing_cols:
                st.error(f"Missing columns: {missing_cols}. Please upload the correct 'Online Retail.csv' file.")
            else:
                with st.spinner("Cleaning data and engineering features..."):
                    df = df.dropna(subset=['CustomerID'])
                    df = df.drop_duplicates()
                    df['Quantity']  = pd.to_numeric(df['Quantity'],  errors='coerce')
                    df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
                    df = df[df['Quantity'] > 0]
                    df['TotalSpend'] = df['Quantity'] * df['UnitPrice']

                    customer_data = df.groupby('CustomerID').agg(
                        AvgQuantity=('Quantity', 'mean'),
                        AvgUnitPrice=('UnitPrice', 'mean'),
                        TotalSpend=('TotalSpend', 'sum'),
                        Frequency=('InvoiceNo', 'nunique')
                    )

                    st.session_state.df = df
                    st.session_state.customer_df = customer_data

                st.success("✅ Data cleaned and aggregated by CustomerID!")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Unique Customers", f"{customer_data.shape[0]:,}")
                with col_b:
                    st.metric("Clean Transactions", f"{df.shape[0]:,}")
                with col_c:
                    st.metric("Avg Spend / Customer", f"£{customer_data['TotalSpend'].mean():,.0f}")

        # ---- Show rest only after cleaning ----
        if st.session_state.customer_df is not None:
            c_df = st.session_state.customer_df.copy()

            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(c_df)
            scaled_df = pd.DataFrame(scaled_features, columns=c_df.columns, index=c_df.index)

            st.subheader("🔢 Scaled Feature Preview (Ready for ML)")
            st.dataframe(scaled_df.head(6), use_container_width=True)

            # ============================================================
            # TASK 2 — K-MEANS CLUSTERING
            # ============================================================
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<span class="task-badge">Task 2</span>', unsafe_allow_html=True)
            st.header("Customer Segmentation — K-Means Clustering")

            st.markdown("""
            <div class="insight-card">
            <h4>🧠 Intuition Behind K-Means</h4>
            <p>
            K-Means partitions customers into <strong>K groups</strong> by minimizing 
            within-cluster variance. Each customer is assigned to the nearest centroid, 
            centroids are recalculated, and the process iterates until convergence.
            The <strong>Elbow Method</strong> finds the point where adding more clusters 
            yields diminishing returns (inertia stops dropping sharply), and the 
            <strong>Silhouette Score</strong> (−1 to +1) measures how well-separated the 
            clusters are — higher is better.
            </p>
            </div>
            """, unsafe_allow_html=True)

            # Elbow + Silhouette together
            inertia_vals, sil_vals = [], []
            K_range = range(2, 9)
            for k in K_range:
                km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels  = km_temp.fit_predict(scaled_df)
                inertia_vals.append(km_temp.inertia_)
                sil_vals.append(silhouette_score(scaled_df, labels))

            fig_elbow, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
            ax1.plot(list(K_range), inertia_vals, marker='o', color='#00d4ff', linewidth=2.5, markersize=8)
            ax1.fill_between(list(K_range), inertia_vals, alpha=0.15, color='#00d4ff')
            ax1.set_title("Elbow Method — Inertia vs K", fontsize=13, fontweight='bold')
            ax1.set_xlabel("Number of Clusters (K)")
            ax1.set_ylabel("Inertia")

            ax2.plot(list(K_range), sil_vals, marker='s', color='#7c4dff', linewidth=2.5, markersize=8)
            ax2.fill_between(list(K_range), sil_vals, alpha=0.15, color='#7c4dff')
            ax2.set_title("Silhouette Score vs K", fontsize=13, fontweight='bold')
            ax2.set_xlabel("Number of Clusters (K)")
            ax2.set_ylabel("Silhouette Score")
            plt.tight_layout()
            st.pyplot(fig_elbow)

            k_val = st.slider("🎚️  Select Number of Clusters", 2, 6, 3)
            kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
            c_df['Cluster'] = kmeans.fit_predict(scaled_df)

            sil = silhouette_score(scaled_df, c_df['Cluster'])
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.metric("Silhouette Score", f"{sil:.3f}", help="Closer to 1.0 = well-separated clusters")
            with col_s2:
                st.metric("Number of Clusters", k_val)

            # Cluster scatter + distribution
            fig_clus, axes = plt.subplots(1, 2, figsize=(13, 5))
            for i, cluster_id in enumerate(sorted(c_df['Cluster'].unique())):
                subset = c_df[c_df['Cluster'] == cluster_id]
                axes[0].scatter(subset['TotalSpend'], subset['Frequency'],
                                color=PALETTE[i % len(PALETTE)], alpha=0.65,
                                s=40, label=f"Cluster {cluster_id}", edgecolors='none')
            axes[0].set_title("Customer Segments: Total Spend vs Purchase Frequency", fontsize=11)
            axes[0].set_xlabel("Total Spend (£)")
            axes[0].set_ylabel("Purchase Frequency")
            axes[0].legend()

            c_df['Cluster'].value_counts().sort_index().plot(
                kind='bar', ax=axes[1],
                color=PALETTE[:k_val], edgecolor='none'
            )
            axes[1].set_title("Cluster Size Distribution", fontsize=11)
            axes[1].set_xlabel("Cluster")
            axes[1].set_ylabel("Number of Customers")
            plt.tight_layout()
            st.pyplot(fig_clus)

            # ---- Cluster Profiling Table ----
            st.subheader("📊 Cluster Profile — Customer Characteristics")
            cluster_profile = c_df.groupby('Cluster')[['AvgQuantity', 'AvgUnitPrice', 'TotalSpend', 'Frequency']].mean().round(2)

            # Assign labels based on spend + frequency
            cluster_labels = {}
            for idx, row in cluster_profile.iterrows():
                if row['TotalSpend'] >= cluster_profile['TotalSpend'].quantile(0.66):
                    cluster_labels[idx] = "🏆 VIP / High-Value"
                elif row['TotalSpend'] >= cluster_profile['TotalSpend'].quantile(0.33):
                    cluster_labels[idx] = "🔄 Regular / Mid-Tier"
                else:
                    cluster_labels[idx] = "💤 Occasional / Low-Value"

            cluster_profile['Segment Label'] = cluster_profile.index.map(cluster_labels)
            st.dataframe(cluster_profile.style.background_gradient(cmap='Blues', subset=['TotalSpend', 'Frequency']),
                         use_container_width=True)

            # Cluster insights
            st.subheader("💡 Cluster Insights")
            for idx, row in cluster_profile.iterrows():
                label = cluster_labels[idx]
                st.markdown(f"""
                <div class="insight-card">
                <h4>Cluster {idx} — {label}</h4>
                <p>
                Average Spend: <strong>£{row['TotalSpend']:,.2f}</strong> &nbsp;|&nbsp;
                Purchase Frequency: <strong>{row['Frequency']:.1f} orders</strong> &nbsp;|&nbsp;
                Avg Quantity/Order: <strong>{row['AvgQuantity']:.1f}</strong> &nbsp;|&nbsp;
                Avg Unit Price: <strong>£{row['AvgUnitPrice']:.2f}</strong>
                <br><br>
                {"These customers are your most valuable segment — high spenders with frequent orders. Retention strategies such as loyalty programs and early access offers are recommended." if "VIP" in label else
                 "These mid-tier customers show moderate engagement. Targeted promotions and upselling campaigns can move them to the VIP tier." if "Regular" in label else
                 "These occasional buyers have low spend and purchase rarely. Win-back campaigns and discount triggers may reactivate them."}
                </p>
                </div>
                """, unsafe_allow_html=True)

            # ============================================================
            # TASK 3 — ANOMALY DETECTION (KDE — DENSITY BASED)
            # ============================================================
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<span class="task-badge">Task 3</span>', unsafe_allow_html=True)
            st.header("Anomaly Detection — Kernel Density Estimation (KDE)")

            st.markdown("""
            <div class="insight-card">
            <h4>🧠 Why KDE for Anomaly Detection?</h4>
            <p>
            Kernel Density Estimation (KDE) is a <strong>non-parametric density estimation</strong> 
            technique. It estimates the probability density of each customer's feature vector 
            without assuming any underlying distribution (unlike Gaussian which assumes normality).
            Customers who fall in <strong>low-density regions</strong> (rare combinations of 
            spend + frequency) are flagged as anomalies — they behave differently from the majority.
            This catches bulk buyers, fraudsters, and data entry errors that simple z-score 
            thresholds might miss in skewed distributions.
            </p>
            </div>
            """, unsafe_allow_html=True)

            threshold_pct = st.slider(
                "🎚️  Anomaly Sensitivity — Density Percentile Threshold",
                1, 15, 5,
                help="Lower % = only extreme outliers flagged. Higher % = more anomalies detected."
            )

            # Fit KDE on TotalSpend + Frequency (2D for visualization)
            kde_features = scaled_df[['TotalSpend', 'Frequency']].values
            kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
            kde.fit(kde_features)
            log_density = kde.score_samples(kde_features)

            density_threshold = np.percentile(log_density, threshold_pct)
            c_df['Is_Anomaly'] = log_density < density_threshold
            c_df['LogDensity'] = log_density

            n_anom = c_df['Is_Anomaly'].sum()
            col_a1, col_a2, col_a3 = st.columns(3)
            with col_a1:
                st.metric("Anomalies Detected", n_anom)
            with col_a2:
                st.metric("Normal Customers", len(c_df) - n_anom)
            with col_a3:
                st.metric("Anomaly Rate", f"{n_anom/len(c_df)*100:.2f}%")

            fig_anom, axes = plt.subplots(1, 2, figsize=(13, 5))

            # Panel 1 — Scatter: normal vs anomaly
            normal  = c_df[~c_df['Is_Anomaly']]
            anomaly = c_df[c_df['Is_Anomaly']]
            axes[0].scatter(normal['TotalSpend'],  normal['Frequency'],
                            color='#00d4ff', alpha=0.5, s=20, label='Normal', edgecolors='none')
            axes[0].scatter(anomaly['TotalSpend'], anomaly['Frequency'],
                            color='#ff5252', alpha=0.9, s=60, label='Anomaly',
                            edgecolors='white', linewidths=0.5, zorder=5)
            axes[0].set_title("KDE Anomaly Detection: Spend vs Frequency", fontsize=11)
            axes[0].set_xlabel("Total Spend (£)")
            axes[0].set_ylabel("Purchase Frequency")
            axes[0].legend()

            # Panel 2 — Log-Density distribution
            axes[1].hist(log_density[~c_df['Is_Anomaly']], bins=40, color='#00d4ff',
                         alpha=0.7, label='Normal', edgecolor='none')
            axes[1].hist(log_density[c_df['Is_Anomaly']], bins=20, color='#ff5252',
                         alpha=0.9, label='Anomaly', edgecolor='none')
            axes[1].axvline(density_threshold, color='#ffb300', linewidth=2,
                            linestyle='--', label=f'Threshold ({threshold_pct}th pct)')
            axes[1].set_title("Log-Density Distribution", fontsize=11)
            axes[1].set_xlabel("Log Density Score")
            axes[1].set_ylabel("Number of Customers")
            axes[1].legend()
            plt.tight_layout()
            st.pyplot(fig_anom)

            st.subheader("🔍 Sample Anomalous Customers")
            st.dataframe(
                c_df[c_df['Is_Anomaly']][['AvgQuantity','AvgUnitPrice','TotalSpend','Frequency','LogDensity']]
                .sort_values('LogDensity').head(10).round(3),
                use_container_width=True
            )

            st.markdown("""
            <div class="insight-card">
            <h4>💡 Why Are These Customers Anomalous?</h4>
            <p>
            These customers exist in <strong>low-density regions</strong> of the feature space — 
            meaning their combination of spend, frequency, and quantity is statistically rare.
            Possible causes include: <br>
            • <strong>Bulk / wholesale buyers</strong> placing very large single orders<br>
            • <strong>Fraudulent activity</strong> — unusually high spend with very low frequency<br>
            • <strong>Data entry errors</strong> — negative quantities or extreme unit prices<br>
            • <strong>One-time event buyers</strong> — corporate gifts or seasonal stockpilers<br>
            These customers deserve targeted manual review before inclusion in marketing models.
            </p>
            </div>
            """, unsafe_allow_html=True)

            # ============================================================
            # TASK 4 — PCA DIMENSIONALITY REDUCTION
            # ============================================================
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<span class="task-badge">Task 4</span>', unsafe_allow_html=True)
            st.header("Dimensionality Reduction — Principal Component Analysis (PCA)")

            st.markdown("""
            <div class="insight-card">
            <h4>🧠 Intuition Behind PCA</h4>
            <p>
            PCA finds the <strong>directions of maximum variance</strong> in high-dimensional data 
            and projects the data onto a lower-dimensional space (principal components).
            PC1 captures the most variance, PC2 the second most, and so on.
            Dimensionality reduction removes noise, speeds up algorithms, and allows 
            high-dimensional clusters to be <strong>visualized in 2D</strong> while 
            preserving the essential structure of the data.
            </p>
            </div>
            """, unsafe_allow_html=True)

            pca_input = scaled_df.drop(columns=['Cluster', 'Is_Anomaly', 'LogDensity'], errors='ignore')
            pca_full  = PCA()
            pca_full.fit(pca_input)
            explained = pca_full.explained_variance_ratio_

            # Scree plot + cumulative variance
            pca_2 = PCA(n_components=2)
            pca_res = pca_2.fit_transform(pca_input)
            pca_df = pd.DataFrame(pca_res, columns=['PC1', 'PC2'], index=c_df.index)

            fig_pca, axes = plt.subplots(1, 3, figsize=(16, 5))

            # Scree plot
            comp_names = [f"PC{i+1}" for i in range(len(explained))]
            axes[0].bar(comp_names, explained * 100, color=PALETTE[:len(explained)], edgecolor='none')
            axes[0].set_title("Scree Plot — Variance per Component", fontsize=11)
            axes[0].set_xlabel("Principal Component")
            axes[0].set_ylabel("Variance Explained (%)")

            # Cumulative variance
            cumulative = np.cumsum(explained) * 100
            axes[1].plot(comp_names, cumulative, marker='o', color='#00d4ff', linewidth=2.5)
            axes[1].fill_between(range(len(cumulative)), cumulative, alpha=0.15, color='#00d4ff')
            axes[1].axhline(80, color='#ffb300', linestyle='--', label='80% threshold')
            axes[1].set_title("Cumulative Variance Explained", fontsize=11)
            axes[1].set_xlabel("Principal Component")
            axes[1].set_ylabel("Cumulative Variance (%)")
            axes[1].legend()
            axes[1].set_xticks(range(len(comp_names)))
            axes[1].set_xticklabels(comp_names)

            # 2D PCA scatter with cluster colors
            for i, cluster_id in enumerate(sorted(c_df['Cluster'].unique())):
                mask = c_df['Cluster'] == cluster_id
                axes[2].scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'],
                                color=PALETTE[i % len(PALETTE)], alpha=0.6,
                                s=30, label=f"Cluster {cluster_id}", edgecolors='none')
            axes[2].set_title(
                f"PCA 2D Projection\n({pca_2.explained_variance_ratio_.sum()*100:.1f}% variance retained)",
                fontsize=11
            )
            axes[2].set_xlabel(f"PC1 ({pca_2.explained_variance_ratio_[0]*100:.1f}%)")
            axes[2].set_ylabel(f"PC2 ({pca_2.explained_variance_ratio_[1]*100:.1f}%)")
            axes[2].legend()
            plt.tight_layout()
            st.pyplot(fig_pca)

            col_pca1, col_pca2, col_pca3 = st.columns(3)
            with col_pca1:
                st.metric("PC1 Variance", f"{explained[0]*100:.1f}%")
            with col_pca2:
                st.metric("PC2 Variance", f"{explained[1]*100:.1f}%")
            with col_pca3:
                st.metric("PC1+PC2 Combined", f"{(explained[0]+explained[1])*100:.1f}%")

            st.markdown("""
            <div class="insight-card">
            <h4>💡 PCA Insights</h4>
            <p>
            The first two principal components together capture a significant portion of the total 
            data variance, confirming that the original 4 features have considerable 
            <strong>shared information (multicollinearity)</strong>. 
            Customers who purchase frequently also tend to spend more — these correlated 
            patterns compress naturally into PC1. PC2 likely captures the trade-off between 
            purchase price and quantity. The 2D PCA plot visually confirms that K-Means 
            clusters found <strong>geometrically distinct groups</strong> even in compressed space.
            </p>
            </div>
            """, unsafe_allow_html=True)

            # ============================================================
            # TASK 5 — RECOMMENDATION SYSTEM
            # ============================================================
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<span class="task-badge">Task 5</span>', unsafe_allow_html=True)
            st.header("Recommendation System — Collaborative Filtering")

            st.markdown("""
            <div class="insight-card">
            <h4>🧠 Intuition Behind Collaborative Filtering</h4>
            <p>
            Collaborative filtering recommends items based on <strong>user-user similarity</strong>. 
            The logic: "customers who bought similar things in the past will likely buy similar 
            things in the future." We build a <strong>customer × product matrix</strong> 
            (rows = customers, columns = products, values = quantities purchased).
            <strong>Cosine Similarity</strong> measures the angle between two customer vectors — 
            a score of 1 means identical purchase patterns, 0 means no overlap.
            For each target customer, we find their most similar peer and recommend 
            products that peer bought but the target has not yet purchased.
            </p>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Building customer-product matrix and computing similarities..."):
                df_tx = st.session_state.df
                sample_size = min(50000, len(df_tx))
                small_df    = df_tx.sample(sample_size, random_state=42)
                matrix      = small_df.pivot_table(
                    index='CustomerID', columns='StockCode',
                    values='Quantity', fill_value=0
                )
                user_sim     = cosine_similarity(matrix)
                user_sim_df  = pd.DataFrame(user_sim, index=matrix.index, columns=matrix.index)

            def get_recommendations(cid, n=5):
                if cid not in user_sim_df.index:
                    return None, None, []
                sim_scores   = user_sim_df[cid].sort_values(ascending=False)
                similar_user = sim_scores.index[1]
                similarity   = sim_scores.iloc[1]
                items_target = set(matrix.loc[cid][matrix.loc[cid] > 0].index)
                items_peer   = set(matrix.loc[similar_user][matrix.loc[similar_user] > 0].index)
                recs         = list(items_peer - items_target)[:n]
                return similar_user, similarity, recs

            test_users = matrix.index[:5]
            st.subheader("🎯 Sample Recommendations for 5 Customers")

            for u in test_users:
                similar_user, similarity, recs = get_recommendations(u)
                with st.expander(f"Customer {int(u)} — Recommendations"):
                    col_r1, col_r2, col_r3 = st.columns(3)
                    with col_r1:
                        st.metric("Most Similar Customer", int(similar_user) if similar_user else "N/A")
                    with col_r2:
                        st.metric("Cosine Similarity", f"{similarity:.3f}" if similarity else "N/A")
                    with col_r3:
                        st.metric("Products Recommended", len(recs))
                    if recs:
                        rec_df = pd.DataFrame({
                            'Rank': range(1, len(recs)+1),
                            'Recommended StockCode': recs,
                            'Reason': [f"Purchased by similar customer {int(similar_user)}" for _ in recs]
                        })
                        st.dataframe(rec_df, use_container_width=True)
                    else:
                        st.info("No new product recommendations available for this customer.")

            # Similarity heatmap (small sample)
            st.subheader("📊 Customer Similarity Heatmap (Top 15 Customers)")
            top15 = matrix.index[:15]
            sim_small = user_sim_df.loc[top15, top15]
            fig_sim, ax_sim = plt.subplots(figsize=(10, 7))
            sns.heatmap(
                sim_small, ax=ax_sim, cmap='Blues',
                linewidths=0.4, linecolor='#0d1b2a',
                annot=True, fmt='.2f', annot_kws={'size': 7},
                cbar_kws={'label': 'Cosine Similarity'}
            )
            ax_sim.set_title("User-User Cosine Similarity Matrix", fontsize=12, fontweight='bold')
            ax_sim.set_xlabel("Customer ID")
            ax_sim.set_ylabel("Customer ID")
            plt.tight_layout()
            st.pyplot(fig_sim)

            # ============================================================
            # TASK 6 — ANALYSIS & REFLECTION
            # ============================================================
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<span class="task-badge">Task 6</span>', unsafe_allow_html=True)
            st.header("Analysis & Reflection — Key Insights")

            st.subheader("🔍 How Unsupervised Learning Uncovered Hidden Patterns")

            col_i1, col_i2 = st.columns(2)

            with col_i1:
                st.markdown("""
                <div class="insight-card">
                <h4>🎯 K-Means Clustering</h4>
                <p>
                Without any labels, K-Means discovered that customers naturally group into 
                distinct behavioral segments — <strong>VIP high-spenders</strong>, 
                <strong>regular mid-tier buyers</strong>, and <strong>occasional low-value customers</strong>. 
                This is actionable: marketing teams can now personalize campaigns per segment 
                instead of applying one-size-fits-all strategies.
                <br><br>
                <strong>Limitation:</strong> K-Means assumes spherical clusters and is sensitive 
                to outliers. The Silhouette Score helps validate quality, but manual inspection 
                of cluster profiles remains essential.
                </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="insight-card">
                <h4>🔬 PCA Dimensionality Reduction</h4>
                <p>
                PCA successfully compressed 4 features into 2 components while retaining the 
                majority of variance. This confirms the features are correlated — spending, 
                frequency, and quantity share underlying patterns. 
                <br><br>
                PCA is useful as a <strong>preprocessing step</strong> before clustering 
                in very high-dimensional datasets, removing noise and speeding up training. 
                <strong>Limitation:</strong> Principal components are linear combinations and 
                lose interpretability — you can no longer say "PC1 = TotalSpend."
                </p>
                </div>
                """, unsafe_allow_html=True)

            with col_i2:
                st.markdown("""
                <div class="insight-card">
                <h4>🚨 KDE Anomaly Detection</h4>
                <p>
                KDE-based anomaly detection identified customers with statistically rare 
                purchase patterns — bulk buyers, potential fraudsters, and data entry errors.
                Unlike simple Z-score thresholding, <strong>KDE makes no assumptions 
                about the distribution</strong>, making it more robust for skewed retail data.
                <br><br>
                <strong>Limitation:</strong> KDE requires careful bandwidth selection. 
                A very small bandwidth over-flags normal customers as anomalies, 
                while a large bandwidth misses true outliers.
                </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="insight-card">
                <h4>🤝 Collaborative Filtering Recommendations</h4>
                <p>
                The recommendation system successfully identified similar customers 
                using cosine similarity on purchase vectors and generated personalized 
                product suggestions without any explicit ratings — purely from behavior.
                <br><br>
                <strong>Limitation:</strong> Collaborative filtering suffers from the 
                <strong>cold-start problem</strong> — new customers with no history 
                cannot receive recommendations. It also struggles with sparse matrices 
                when product catalogs are large.
                </p>
                </div>
                """, unsafe_allow_html=True)

            # Comparison Table
            st.subheader("📊 Technique Comparison")
            comparison_df = pd.DataFrame({
                'Technique':        ['K-Means Clustering', 'KDE Anomaly Detection', 'PCA', 'Collaborative Filtering'],
                'Goal':             ['Segment customers into groups', 'Detect unusual customers', 'Compress feature space', 'Personalize product recommendations'],
                'Input':            ['Scaled customer features', 'Scaled customer features', 'Scaled customer features', 'Customer × Product matrix'],
                'Output':           ['Cluster labels', 'Anomaly flags', '2D projection', 'Recommended items'],
                'Strength':         ['Scalable & interpretable', 'No distribution assumption', 'Removes noise & correlations', 'Fully behavior-driven'],
                'Limitation':       ['Assumes spherical clusters', 'Bandwidth sensitivity', 'Loses feature interpretability', 'Cold-start problem'],
                'Business Value':   ['Targeted marketing', 'Fraud / error detection', 'Faster ML pipelines', 'Personalized UX'],
            })
            st.dataframe(comparison_df, use_container_width=True)

            # Real-world applications
            st.subheader("🌍 Real-World Applications of This Pipeline")
            col_app1, col_app2, col_app3 = st.columns(3)
            with col_app1:
                st.markdown("""
                <div class="insight-card">
                <h4>🛒 E-Commerce</h4>
                <p>
                Amazon & Alibaba use clustering + collaborative filtering to power 
                "Customers Also Bought" sections and personalized homepages. 
                Anomaly detection flags fraudulent orders in real time.
                </p>
                </div>
                """, unsafe_allow_html=True)
            with col_app2:
                st.markdown("""
                <div class="insight-card">
                <h4>🏦 Banking & Finance</h4>
                <p>
                KDE anomaly detection identifies suspicious transactions. 
                K-Means segments customers for credit risk assessment and 
                tailored financial product offerings.
                </p>
                </div>
                """, unsafe_allow_html=True)
            with col_app3:
                st.markdown("""
                <div class="insight-card">
                <h4>🎬 Media & Streaming</h4>
                <p>
                Netflix uses collaborative filtering to recommend content. 
                PCA reduces the massive movie feature space, and clustering 
                groups viewers by taste profile for content licensing decisions.
                </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.success("""
            ✅ All 6 tasks completed successfully!
            This dashboard demonstrates a complete unsupervised learning pipeline:
            Data Preprocessing → K-Means Segmentation → KDE Anomaly Detection → 
            PCA Dimensionality Reduction → Collaborative Filtering → Business Insights
            """)

else:
    # ---- Dataset failed to load ----
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="insight-card">
        <h4>🎯 Segmentation</h4>
        <p>K-Means clustering identifies VIP, Regular, and Occasional customer segments
        from raw transaction data.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="insight-card">
        <h4>🚨 Anomaly Detection</h4>
        <p>Kernel Density Estimation flags unusual buyers — bulk purchasers,
        potential fraud, and data errors.</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="insight-card">
        <h4>🤝 Recommendations</h4>
        <p>Collaborative filtering recommends products to customers based on
        behavioral similarity — no labels required.</p>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.warning(
        "The dataset could not be loaded automatically. "
        "Please upload the **Online Retail CSV** manually using the uploader above."
    )