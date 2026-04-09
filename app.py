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





# ================================================================
# ONLINE RETAIL CUSTOMER INTELLIGENCE DASHBOARD
# Unsupervised Learning: K-Means · KDE Anomaly Detection · PCA · Collaborative Filtering
#
# NO FILE UPLOAD NEEDED — dataset is generated automatically in memory.
# Anyone who opens this app sees the full dashboard instantly.
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KernelDensity
from scipy.stats import zscore

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    layout="wide",
    page_icon="🛒",
    initial_sidebar_state="expanded"
)

# ================================================================
# CUSTOM CSS
# ================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0f1e;
    color: #e8eaf6;
}
.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0f2236 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #0a1628 100%);
    border-right: 1px solid rgba(0,212,255,0.15);
}
[data-testid="stSidebar"] * { color: #e8eaf6 !important; }
h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.6rem !important;
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
h3 { font-family: 'Syne', sans-serif !important; font-weight: 600 !important; color: #a78bfa !important; }
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(124,77,255,0.08));
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 16px;
    padding: 20px;
}
[data-testid="metric-container"] label {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #00d4ff !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.9rem !important;
}
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #7c4dff) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 10px 28px !important;
    box-shadow: 0 4px 20px rgba(0,212,255,0.25) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,212,255,0.4) !important;
}
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
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.insight-card p { color: #cbd5e1; font-size: 0.92rem; line-height: 1.65; margin: 0; }
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.4), transparent);
    margin: 32px 0;
}
.task-badge {
    display: inline-block;
    background: linear-gradient(135deg, #7c4dff, #00d4ff);
    color: white;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.72rem;
    padding: 4px 14px;
    border-radius: 20px;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.auto-badge {
    display: inline-block;
    background: linear-gradient(135deg, #00c853, #00e5ff);
    color: #0a0f1e;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.75rem;
    padding: 6px 18px;
    border-radius: 20px;
    letter-spacing: 1px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ================================================================
# MATPLOTLIB DARK THEME
# ================================================================
plt.rcParams.update({
    'figure.facecolor': '#0d1b2a', 'axes.facecolor': '#0d1b2a',
    'axes.edgecolor': '#1e3a5f',   'axes.labelcolor': '#94a3b8',
    'axes.titlecolor': '#e8eaf6',  'xtick.color': '#94a3b8',
    'ytick.color': '#94a3b8',      'text.color': '#e8eaf6',
    'grid.color': '#1e3a5f',       'grid.alpha': 0.5,
    'axes.grid': True,             'legend.facecolor': '#0d1b2a',
    'legend.edgecolor': '#1e3a5f', 'font.family': 'DejaVu Sans',
})
PALETTE = ['#00d4ff', '#7c4dff', '#ff6b9d', '#00ff99', '#ffb300', '#ff5252']

# ================================================================
# SESSION STATE
# ================================================================
for key in ['df', 'customer_df']:
    if key not in st.session_state:
        st.session_state[key] = None

# ================================================================
# SYNTHETIC DATASET — generated 100% in memory, no CSV needed
# ================================================================
@st.cache_data(show_spinner=False)
def generate_retail_data(seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic Online Retail transaction dataset.
    Columns match the UCI Online Retail dataset exactly:
    InvoiceNo, StockCode, Description, Quantity,
    InvoiceDate, UnitPrice, CustomerID, Country
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    products = [
        ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER",   2.55),
        ("71053",  "WHITE METAL LANTERN",                   3.39),
        ("84406B", "CREAM CUPID HEARTS COAT HANGER",        2.75),
        ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE",   3.39),
        ("84029E", "RED WOOLLY HOTTIE WHITE HEART",          3.39),
        ("22752",  "SET 7 BABUSHKA NESTING BOXES",           7.65),
        ("21730",  "GLASS STAR FROSTED T-LIGHT HOLDER",     4.25),
        ("22633",  "HAND WARMER UNION JACK",                 1.85),
        ("22632",  "HAND WARMER RED POLKA DOT",              1.85),
        ("47566",  "PARTY BUNTING",                          4.95),
        ("85099B", "JUMBO BAG RED RETROSPOT",                1.65),
        ("20727",  "LUNCH BAG BLACK SKULL",                  1.65),
        ("23343",  "JUMBO SHOPPER VINTAGE RED PAISLEY",      2.10),
        ("23166",  "MEDIUM CERAMIC TOP STORAGE JAR",         1.04),
        ("22960",  "JAM MAKING SET WITH JARS",               4.25),
        ("21212",  "PACK OF 72 RETROSPOT CAKE CASES",        0.55),
        ("22086",  "PAPER CHAIN KIT 50S CHRISTMAS",          2.95),
        ("85099C", "JUMBO BAG BAROQUE BLACK WHITE",           1.65),
        ("21034",  "REX CASH+CARRY JUMBO SHOPPER",           1.65),
        ("23298",  "CRYSTAL CANDELABRA",                     9.95),
        ("22423",  "REGENCY CAKESTAND 3 TIER",              12.75),
        ("47580",  "TEA TIME PARTY BUNTING",                 4.95),
        ("22111",  "SCOTTIE DOG HOT WATER BOTTLE",           3.39),
        ("21977",  "PACK OF 60 PINK PAISLEY CAKE CASES",     0.55),
        ("84879",  "ASSORTED COLOUR BIRD ORNAMENT",          1.69),
        ("22386",  "JUMBO BAG PINK POLKADOT",                1.65),
        ("22384",  "LUNCH BAG PINK POLKADOT",                1.65),
        ("22719",  "GINGHAM HEART DOORSTOP",                 4.95),
        ("22720",  "SET OF 3 BUTTERFLY COOKIE CUTTERS",      2.10),
        ("23199",  "SPOTTY BUNNY DECORATION",                2.10),
    ]

    countries = (["United Kingdom"] * 70 +
                 ["Germany", "France", "EIRE", "Spain", "Netherlands",
                  "Belgium", "Switzerland", "Portugal", "Australia",
                  "Norway", "Italy", "Cyprus", "Sweden", "Japan"])

    n_customers = 600
    customer_ids = [10000 + i for i in range(n_customers)]
    segments = rng.choice(
        ["vip", "regular", "occasional"],
        p=[0.10, 0.40, 0.50], size=n_customers
    )
    seg_orders = {"vip": (35, 70), "regular": (8, 28), "occasional": (1, 7)}
    seg_qty    = {"vip": (6, 40),  "regular": (2, 15), "occasional": (1, 5)}

    rows = []
    inv_counter = 500000

    for cid, seg in zip(customer_ids, segments):
        n_orders = int(rng.integers(*seg_orders[seg]))
        country  = random.choice(countries)
        for _ in range(n_orders):
            inv_counter += 1
            n_items  = int(rng.integers(1, 7))
            chosen   = random.sample(products, min(n_items, len(products)))
            day_off  = int(rng.integers(0, 373))
            inv_date = (pd.Timestamp("2010-12-01") +
                        pd.Timedelta(days=day_off)).strftime("%d/%m/%Y %H:%M")
            for stock, desc, base_price in chosen:
                qty   = int(rng.integers(*seg_qty[seg]))
                price = round(float(base_price) * float(rng.uniform(0.85, 1.25)), 2)
                rows.append({
                    "InvoiceNo":   str(inv_counter),
                    "StockCode":   stock,
                    "Description": desc,
                    "Quantity":    qty,
                    "InvoiceDate": inv_date,
                    "UnitPrice":   price,
                    "CustomerID":  float(cid),
                    "Country":     country,
                })

    df = pd.DataFrame(rows)
    # Inject ~2% missing CustomerIDs (realistic noise)
    mask = rng.random(len(df)) < 0.02
    df.loc[mask, "CustomerID"] = np.nan
    # Inject ~0.5% duplicate rows
    n_dups = max(1, int(len(df) * 0.005))
    dups   = df.sample(n_dups, random_state=seed)
    df     = pd.concat([df, dups], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ================================================================
# AUTO-LOAD on every visit — no user action needed at all
# ================================================================
if st.session_state.df is None:
    with st.spinner("Preparing dataset... please wait a few seconds."):
        st.session_state.df = generate_retail_data()

# ================================================================
# SIDEBAR
# ================================================================
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
**Dataset**
Realistic synthetic Online Retail data —
20,000+ transactions · 600 customers
30 products · auto-generated in memory.
No CSV required.
    """)
    st.markdown("---")
    st.caption("Built with Streamlit · Scikit-learn · Seaborn")

# ================================================================
# TITLE
# ================================================================
st.markdown(
    "<h1>🛒 Online Retail Customer Intelligence Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown("""
<p style='color:#94a3b8;font-size:1.05rem;margin-bottom:0.5rem;'>
End-to-end Unsupervised Learning — Customer Segmentation · Anomaly Detection ·
Dimensionality Reduction · Collaborative Filtering
</p>
""", unsafe_allow_html=True)
st.markdown(
    '<span class="auto-badge">✅ DATASET AUTO-LOADED — NO FILE UPLOAD NEEDED</span>',
    unsafe_allow_html=True
)

# ================================================================
# ALL TASKS — always visible, no upload gate
# ================================================================
df = st.session_state.df

# ----------------------------------------------------------------
# TASK 1
# ----------------------------------------------------------------
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

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Rows",     f"{df.shape[0]:,}")
c2.metric("Total Columns",  df.shape[1])
c3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
c4.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Column Info")
    st.write("**Available Columns:**", list(df.columns))
    miss = df.isnull().sum().reset_index()
    miss.columns = ["Column", "Missing Count"]
    st.dataframe(miss, use_container_width=True)
with col_right:
    st.subheader("Descriptive Statistics")
    st.dataframe(
        df.describe().T.style.background_gradient(cmap="Blues"),
        use_container_width=True
    )

if st.button("⚙️  Clean & Aggregate Data"):
    with st.spinner("Cleaning data and engineering features..."):
        df_clean = df.dropna(subset=["CustomerID"]).drop_duplicates().copy()
        df_clean["Quantity"]  = pd.to_numeric(df_clean["Quantity"],  errors="coerce")
        df_clean["UnitPrice"] = pd.to_numeric(df_clean["UnitPrice"], errors="coerce")
        df_clean = df_clean[df_clean["Quantity"] > 0]
        df_clean["TotalSpend"] = df_clean["Quantity"] * df_clean["UnitPrice"]

        customer_data = df_clean.groupby("CustomerID").agg(
            AvgQuantity  = ("Quantity",   "mean"),
            AvgUnitPrice = ("UnitPrice",  "mean"),
            TotalSpend   = ("TotalSpend", "sum"),
            Frequency    = ("InvoiceNo",  "nunique")
        )
        st.session_state.df          = df_clean
        st.session_state.customer_df = customer_data

    st.success("✅ Data cleaned and aggregated by CustomerID!")
    ca, cb, cc = st.columns(3)
    ca.metric("Unique Customers",    f"{customer_data.shape[0]:,}")
    cb.metric("Clean Transactions",  f"{df_clean.shape[0]:,}")
    cc.metric("Avg Spend / Customer",
              f"£{customer_data['TotalSpend'].mean():,.0f}")

# ----------------------------------------------------------------
# Tasks 2-6 shown after cleaning
# ----------------------------------------------------------------
if st.session_state.customer_df is not None:
    c_df = st.session_state.customer_df.copy()

    scaler     = StandardScaler()
    scaled_arr = scaler.fit_transform(c_df)
    scaled_df  = pd.DataFrame(scaled_arr, columns=c_df.columns, index=c_df.index)

    st.subheader("🔢 Scaled Feature Preview (Ready for ML)")
    st.dataframe(scaled_df.head(6), use_container_width=True)

    # ----------------------------------------------------------------
    # TASK 2
    # ----------------------------------------------------------------
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
    The <strong>Elbow Method</strong> finds where inertia stops dropping sharply,
    and the <strong>Silhouette Score</strong> (−1 to +1) measures cluster separation
    — higher is better.
    </p>
    </div>
    """, unsafe_allow_html=True)

    inertia_vals, sil_vals = [], []
    K_list = list(range(2, 9))
    for k in K_list:
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(scaled_df)
        inertia_vals.append(km.inertia_)
        sil_vals.append(silhouette_score(scaled_df, lbl))

    fig_elbow, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(K_list, inertia_vals, marker="o", color="#00d4ff", linewidth=2.5, markersize=8)
    ax1.fill_between(K_list, inertia_vals, alpha=0.15, color="#00d4ff")
    ax1.set_title("Elbow Method — Inertia vs K", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia")
    ax2.plot(K_list, sil_vals, marker="s", color="#7c4dff", linewidth=2.5, markersize=8)
    ax2.fill_between(K_list, sil_vals, alpha=0.15, color="#7c4dff")
    ax2.set_title("Silhouette Score vs K", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette Score")
    plt.tight_layout()
    st.pyplot(fig_elbow)

    k_val  = st.slider("🎚️  Select Number of Clusters", 2, 6, 3)
    kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
    c_df["Cluster"] = kmeans.fit_predict(scaled_df)

    sil = silhouette_score(scaled_df, c_df["Cluster"])
    s1, s2 = st.columns(2)
    s1.metric("Silhouette Score",   f"{sil:.3f}", help="Closer to 1.0 = well-separated")
    s2.metric("Number of Clusters", k_val)

    fig_clus, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, cid in enumerate(sorted(c_df["Cluster"].unique())):
        sub = c_df[c_df["Cluster"] == cid]
        axes[0].scatter(sub["TotalSpend"], sub["Frequency"],
                        color=PALETTE[i % len(PALETTE)], alpha=0.65,
                        s=40, label=f"Cluster {cid}", edgecolors="none")
    axes[0].set_title("Customer Segments: Total Spend vs Frequency", fontsize=11)
    axes[0].set_xlabel("Total Spend (£)")
    axes[0].set_ylabel("Purchase Frequency")
    axes[0].legend()
    c_df["Cluster"].value_counts().sort_index().plot(
        kind="bar", ax=axes[1], color=PALETTE[:k_val], edgecolor="none"
    )
    axes[1].set_title("Cluster Size Distribution", fontsize=11)
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Number of Customers")
    plt.tight_layout()
    st.pyplot(fig_clus)

    st.subheader("📊 Cluster Profile — Customer Characteristics")
    profile = (c_df.groupby("Cluster")
               [["AvgQuantity", "AvgUnitPrice", "TotalSpend", "Frequency"]]
               .mean().round(2))
    seg_labels = {}
    for idx, row in profile.iterrows():
        if row["TotalSpend"] >= profile["TotalSpend"].quantile(0.66):
            seg_labels[idx] = "🏆 VIP / High-Value"
        elif row["TotalSpend"] >= profile["TotalSpend"].quantile(0.33):
            seg_labels[idx] = "🔄 Regular / Mid-Tier"
        else:
            seg_labels[idx] = "💤 Occasional / Low-Value"
    profile["Segment Label"] = profile.index.map(seg_labels)
    st.dataframe(
        profile.style.background_gradient(
            cmap="Blues", subset=["TotalSpend", "Frequency"]
        ),
        use_container_width=True
    )

    st.subheader("💡 Cluster Insights")
    for idx, row in profile.iterrows():
        label = seg_labels[idx]
        if "VIP" in label:
            desc = ("Most valuable customers — high spenders with frequent orders. "
                    "Retain them with loyalty programs and early-access offers.")
        elif "Regular" in label:
            desc = ("Mid-tier customers with moderate engagement. Targeted promotions "
                    "and upselling can move them up to the VIP tier.")
        else:
            desc = ("Occasional buyers who purchase rarely. Win-back campaigns "
                    "and time-limited discounts may reactivate them.")
        st.markdown(f"""
        <div class="insight-card">
        <h4>Cluster {idx} — {label}</h4>
        <p>
        Avg Spend: <strong>£{row['TotalSpend']:,.2f}</strong> &nbsp;|&nbsp;
        Frequency: <strong>{row['Frequency']:.1f} orders</strong> &nbsp;|&nbsp;
        Avg Qty: <strong>{row['AvgQuantity']:.1f}</strong> &nbsp;|&nbsp;
        Avg Unit Price: <strong>£{row['AvgUnitPrice']:.2f}</strong>
        <br><br>{desc}
        </p>
        </div>
        """, unsafe_allow_html=True)

    # ----------------------------------------------------------------
    # TASK 3
    # ----------------------------------------------------------------
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="task-badge">Task 3</span>', unsafe_allow_html=True)
    st.header("Anomaly Detection — Kernel Density Estimation (KDE)")

    st.markdown("""
    <div class="insight-card">
    <h4>🧠 Why KDE for Anomaly Detection?</h4>
    <p>
    Kernel Density Estimation (KDE) is a <strong>non-parametric density estimation</strong>
    technique. It estimates the probability density of each customer's feature vector
    without assuming any underlying distribution.
    Customers who fall in <strong>low-density regions</strong> (rare combinations of
    spend and frequency) are flagged as anomalies — they behave very differently
    from the majority. This catches bulk buyers, fraudsters, and data errors that
    simple Z-score thresholds miss in skewed distributions.
    </p>
    </div>
    """, unsafe_allow_html=True)

    threshold_pct = st.slider(
        "🎚️  Anomaly Sensitivity — Density Percentile Threshold", 1, 15, 5,
        help="Lower % = only extreme outliers. Higher % = more anomalies flagged."
    )
    kde_feat = scaled_df[["TotalSpend", "Frequency"]].values
    kde      = KernelDensity(kernel="gaussian", bandwidth=0.5)
    kde.fit(kde_feat)
    log_dens = kde.score_samples(kde_feat)

    dens_thresh          = np.percentile(log_dens, threshold_pct)
    c_df["Is_Anomaly"]   = log_dens < dens_thresh
    c_df["LogDensity"]   = log_dens

    n_anom = int(c_df["Is_Anomaly"].sum())
    a1, a2, a3 = st.columns(3)
    a1.metric("Anomalies Detected", n_anom)
    a2.metric("Normal Customers",   len(c_df) - n_anom)
    a3.metric("Anomaly Rate",       f"{n_anom / len(c_df) * 100:.2f}%")

    fig_anom, axes = plt.subplots(1, 2, figsize=(13, 5))
    normal  = c_df[~c_df["Is_Anomaly"]]
    anomaly = c_df[ c_df["Is_Anomaly"]]
    axes[0].scatter(normal["TotalSpend"],  normal["Frequency"],
                    color="#00d4ff", alpha=0.5, s=20,
                    label="Normal",  edgecolors="none")
    axes[0].scatter(anomaly["TotalSpend"], anomaly["Frequency"],
                    color="#ff5252", alpha=0.9, s=60,
                    label="Anomaly", edgecolors="white",
                    linewidths=0.5, zorder=5)
    axes[0].set_title("KDE Anomaly Detection: Spend vs Frequency", fontsize=11)
    axes[0].set_xlabel("Total Spend (£)")
    axes[0].set_ylabel("Purchase Frequency")
    axes[0].legend()
    axes[1].hist(log_dens[~c_df["Is_Anomaly"]], bins=40,
                 color="#00d4ff", alpha=0.7, label="Normal",  edgecolor="none")
    axes[1].hist(log_dens[ c_df["Is_Anomaly"]], bins=20,
                 color="#ff5252", alpha=0.9, label="Anomaly", edgecolor="none")
    axes[1].axvline(dens_thresh, color="#ffb300", linewidth=2,
                    linestyle="--", label=f"Threshold ({threshold_pct}th pct)")
    axes[1].set_title("Log-Density Distribution", fontsize=11)
    axes[1].set_xlabel("Log Density Score")
    axes[1].set_ylabel("Number of Customers")
    axes[1].legend()
    plt.tight_layout()
    st.pyplot(fig_anom)

    st.subheader("🔍 Sample Anomalous Customers")
    st.dataframe(
        c_df[c_df["Is_Anomaly"]]
        [["AvgQuantity", "AvgUnitPrice", "TotalSpend", "Frequency", "LogDensity"]]
        .sort_values("LogDensity").head(10).round(3),
        use_container_width=True
    )

    st.markdown("""
    <div class="insight-card">
    <h4>💡 Why Are These Customers Anomalous?</h4>
    <p>
    These customers exist in <strong>low-density regions</strong> of the feature space —
    their combination of spend, frequency, and quantity is statistically rare.<br><br>
    • <strong>Bulk / wholesale buyers</strong> — very large single orders<br>
    • <strong>Fraudulent activity</strong> — high spend with very low frequency<br>
    • <strong>Data entry errors</strong> — extreme unit prices or quantities<br>
    • <strong>One-time event buyers</strong> — corporate gifts or seasonal stockpilers<br><br>
    These customers deserve manual review before inclusion in marketing models.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------------------------------------------
    # TASK 4
    # ----------------------------------------------------------------
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="task-badge">Task 4</span>', unsafe_allow_html=True)
    st.header("Dimensionality Reduction — Principal Component Analysis (PCA)")

    st.markdown("""
    <div class="insight-card">
    <h4>🧠 Intuition Behind PCA</h4>
    <p>
    PCA finds the <strong>directions of maximum variance</strong> in high-dimensional data
    and projects it onto a lower-dimensional space (principal components).
    PC1 captures the most variance, PC2 the second most.
    This removes noise, speeds up algorithms, and allows high-dimensional clusters
    to be <strong>visualized in 2D</strong> while preserving the essential structure.
    </p>
    </div>
    """, unsafe_allow_html=True)

    pca_input = scaled_df.drop(
        columns=["Cluster", "Is_Anomaly", "LogDensity"], errors="ignore"
    )
    pca_full = PCA()
    pca_full.fit(pca_input)
    explained  = pca_full.explained_variance_ratio_
    comp_names = [f"PC{i+1}" for i in range(len(explained))]

    pca_2   = PCA(n_components=2)
    pca_res = pca_2.fit_transform(pca_input)
    pca_df  = pd.DataFrame(pca_res, columns=["PC1", "PC2"], index=c_df.index)

    fig_pca, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(comp_names, explained * 100,
                color=PALETTE[:len(explained)], edgecolor="none")
    axes[0].set_title("Scree Plot — Variance per Component", fontsize=11)
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Variance Explained (%)")

    cumulative = np.cumsum(explained) * 100
    axes[1].plot(comp_names, cumulative, marker="o", color="#00d4ff", linewidth=2.5)
    axes[1].fill_between(range(len(cumulative)), cumulative, alpha=0.15, color="#00d4ff")
    axes[1].axhline(80, color="#ffb300", linestyle="--", label="80% threshold")
    axes[1].set_title("Cumulative Variance Explained", fontsize=11)
    axes[1].set_xlabel("Principal Component")
    axes[1].set_ylabel("Cumulative Variance (%)")
    axes[1].legend()
    axes[1].set_xticks(range(len(comp_names)))
    axes[1].set_xticklabels(comp_names)

    for i, cid in enumerate(sorted(c_df["Cluster"].unique())):
        mask = c_df["Cluster"] == cid
        axes[2].scatter(pca_df.loc[mask, "PC1"], pca_df.loc[mask, "PC2"],
                        color=PALETTE[i % len(PALETTE)], alpha=0.6,
                        s=30, label=f"Cluster {cid}", edgecolors="none")
    axes[2].set_title(
        f"PCA 2D Projection\n"
        f"({pca_2.explained_variance_ratio_.sum()*100:.1f}% variance retained)",
        fontsize=11
    )
    axes[2].set_xlabel(f"PC1 ({pca_2.explained_variance_ratio_[0]*100:.1f}%)")
    axes[2].set_ylabel(f"PC2 ({pca_2.explained_variance_ratio_[1]*100:.1f}%)")
    axes[2].legend()
    plt.tight_layout()
    st.pyplot(fig_pca)

    p1, p2, p3 = st.columns(3)
    p1.metric("PC1 Variance",     f"{explained[0]*100:.1f}%")
    p2.metric("PC2 Variance",     f"{explained[1]*100:.1f}%")
    p3.metric("PC1+PC2 Combined", f"{(explained[0]+explained[1])*100:.1f}%")

    st.markdown("""
    <div class="insight-card">
    <h4>💡 PCA Insights</h4>
    <p>
    The first two principal components together capture a significant portion of the
    total data variance, confirming that the original 4 features have considerable
    <strong>shared information (multicollinearity)</strong>.
    Customers who purchase frequently also tend to spend more — these correlated
    patterns compress naturally into PC1. PC2 captures the trade-off between purchase
    price and quantity. The 2D PCA plot visually confirms that K-Means clusters found
    <strong>geometrically distinct groups</strong> even in compressed space.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------------------------------------------
    # TASK 5
    # ----------------------------------------------------------------
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="task-badge">Task 5</span>', unsafe_allow_html=True)
    st.header("Recommendation System — Collaborative Filtering")

    st.markdown("""
    <div class="insight-card">
    <h4>🧠 Intuition Behind Collaborative Filtering</h4>
    <p>
    Collaborative filtering recommends items based on <strong>user-user similarity</strong>.
    The logic: customers who bought similar things in the past will likely buy similar
    things in the future. We build a <strong>customer × product matrix</strong>
    (rows = customers, columns = products, values = quantities purchased).
    <strong>Cosine Similarity</strong> measures the angle between two customer vectors —
    score of 1 = identical purchase patterns, 0 = no overlap.
    For each target customer we find their most similar peer and recommend products
    that peer bought but the target has not yet purchased.
    </p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Building customer-product matrix and computing similarities..."):
        df_tx    = st.session_state.df
        matrix   = df_tx.pivot_table(
            index="CustomerID", columns="StockCode",
            values="Quantity", fill_value=0
        )
        user_sim    = cosine_similarity(matrix)
        user_sim_df = pd.DataFrame(
            user_sim, index=matrix.index, columns=matrix.index
        )

    def get_recommendations(cid, n=5):
        if cid not in user_sim_df.index:
            return None, None, []
        scores       = user_sim_df[cid].sort_values(ascending=False)
        similar_user = scores.index[1]
        similarity   = float(scores.iloc[1])
        items_target = set(matrix.loc[cid][matrix.loc[cid] > 0].index)
        items_peer   = set(matrix.loc[similar_user][matrix.loc[similar_user] > 0].index)
        recs         = list(items_peer - items_target)[:n]
        return similar_user, similarity, recs

    st.subheader("🎯 Sample Recommendations for 5 Customers")
    for u in matrix.index[:5]:
        sim_user, sim_score, recs = get_recommendations(u)
        with st.expander(f"Customer {int(u)} — Recommendations"):
            r1, r2, r3 = st.columns(3)
            r1.metric("Most Similar Customer", int(sim_user) if sim_user else "N/A")
            r2.metric("Cosine Similarity",
                      f"{sim_score:.3f}" if sim_score is not None else "N/A")
            r3.metric("Products Recommended", len(recs))
            if recs:
                rec_df = pd.DataFrame({
                    "Rank":                  range(1, len(recs) + 1),
                    "Recommended StockCode": recs,
                    "Reason": [
                        f"Purchased by similar customer {int(sim_user)}"
                        for _ in recs
                    ],
                })
                st.dataframe(rec_df, use_container_width=True)
            else:
                st.info("No new product recommendations for this customer.")

    st.subheader("📊 Customer Similarity Heatmap (Top 15 Customers)")
    top15     = matrix.index[:15]
    sim_small = user_sim_df.loc[top15, top15]
    fig_sim, ax_sim = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        sim_small, ax=ax_sim, cmap="Blues",
        linewidths=0.4, linecolor="#0d1b2a",
        annot=True, fmt=".2f", annot_kws={"size": 7},
        cbar_kws={"label": "Cosine Similarity"}
    )
    ax_sim.set_title("User-User Cosine Similarity Matrix",
                     fontsize=12, fontweight="bold")
    ax_sim.set_xlabel("Customer ID")
    ax_sim.set_ylabel("Customer ID")
    plt.tight_layout()
    st.pyplot(fig_sim)

    # ----------------------------------------------------------------
    # TASK 6
    # ----------------------------------------------------------------
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="task-badge">Task 6</span>', unsafe_allow_html=True)
    st.header("Analysis & Reflection — Key Insights")

    st.subheader("🔍 How Unsupervised Learning Uncovered Hidden Patterns")
    i1, i2 = st.columns(2)
    with i1:
        st.markdown("""
        <div class="insight-card">
        <h4>🎯 K-Means Clustering</h4>
        <p>
        Without any labels, K-Means discovered that customers naturally group into
        distinct behavioral segments — <strong>VIP high-spenders</strong>,
        <strong>regular mid-tier buyers</strong>, and
        <strong>occasional low-value customers</strong>.
        Marketing teams can now personalize campaigns per segment instead of
        one-size-fits-all strategies.<br><br>
        <strong>Limitation:</strong> Assumes spherical clusters and is sensitive to
        outliers. Silhouette Score helps validate quality, but manual cluster inspection
        remains essential.
        </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-card">
        <h4>🔬 PCA Dimensionality Reduction</h4>
        <p>
        PCA successfully compressed 4 features into 2 components while retaining the
        majority of variance. This confirms the features are correlated — spending,
        frequency, and quantity share underlying patterns.<br><br>
        Useful as a <strong>preprocessing step</strong> before clustering in
        high-dimensional datasets. <strong>Limitation:</strong> Principal components
        are linear combinations and lose direct interpretability.
        </p>
        </div>
        """, unsafe_allow_html=True)
    with i2:
        st.markdown("""
        <div class="insight-card">
        <h4>🚨 KDE Anomaly Detection</h4>
        <p>
        KDE-based anomaly detection identified customers with statistically rare
        purchase patterns — bulk buyers, potential fraudsters, and data errors.
        Unlike simple Z-score thresholding, <strong>KDE makes no distribution
        assumptions</strong>, making it more robust for skewed retail data.<br><br>
        <strong>Limitation:</strong> Requires careful bandwidth selection. A very
        small bandwidth over-flags normal customers; a large one misses true outliers.
        </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-card">
        <h4>🤝 Collaborative Filtering</h4>
        <p>
        The recommendation system identified similar customers using cosine similarity
        and generated personalized product suggestions purely from behavior —
        no explicit ratings needed.<br><br>
        <strong>Limitation:</strong> Suffers from the <strong>cold-start problem</strong>
        — new customers with no purchase history cannot receive recommendations.
        Also struggles with very sparse product matrices.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("📊 Technique Comparison")
    comparison_df = pd.DataFrame({
        "Technique":      ["K-Means Clustering", "KDE Anomaly Detection",
                           "PCA", "Collaborative Filtering"],
        "Goal":           ["Segment customers into groups",
                           "Detect unusual customers",
                           "Compress feature space",
                           "Personalize product recommendations"],
        "Input":          ["Scaled customer features"] * 3 +
                          ["Customer × Product matrix"],
        "Output":         ["Cluster labels", "Anomaly flags",
                           "2D projection",  "Recommended items"],
        "Strength":       ["Scalable & interpretable",
                           "No distribution assumption",
                           "Removes noise & correlations",
                           "Fully behavior-driven"],
        "Limitation":     ["Assumes spherical clusters",
                           "Bandwidth sensitivity",
                           "Loses feature interpretability",
                           "Cold-start problem"],
        "Business Value": ["Targeted marketing", "Fraud / error detection",
                           "Faster ML pipelines", "Personalized UX"],
    })
    st.dataframe(comparison_df, use_container_width=True)

    st.subheader("🌍 Real-World Applications of This Pipeline")
    app1, app2, app3 = st.columns(3)
    with app1:
        st.markdown("""
        <div class="insight-card">
        <h4>🛒 E-Commerce</h4>
        <p>
        Amazon and Alibaba use clustering + collaborative filtering to power
        "Customers Also Bought" sections. Anomaly detection flags fraudulent
        orders in real time.
        </p>
        </div>
        """, unsafe_allow_html=True)
    with app2:
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
    with app3:
        st.markdown("""
        <div class="insight-card">
        <h4>🎬 Media & Streaming</h4>
        <p>
        Netflix uses collaborative filtering to recommend content.
        PCA reduces the massive movie feature space, and clustering
        groups viewers by taste profile for licensing decisions.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.success(
        "✅ All 6 tasks completed successfully! "
        "Data Preprocessing → K-Means Segmentation → KDE Anomaly Detection → "
        "PCA Dimensionality Reduction → Collaborative Filtering → Business Insights"
    )
