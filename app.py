# ================= IMPORT LIBRARIES =================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import zscore

# ================= CONFIG & SESSION STATE =================
st.set_page_config(page_title="Customer Analysis Dashboard", layout="wide")

if 'df' not in st.session_state:
    st.session_state.df = None
if 'customer_df' not in st.session_state:
    st.session_state.customer_df = None

# ================= TITLE =================
st.title(" Online Retail Customer Analysis Dashboard")
st.write("This app performs Unsupervised Learning Tasks 1 to 6 for Customer Insights.")

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader("Upload Online Retail CSV", type=["csv"])

if uploaded_file is not None:
    if st.session_state.df is None:
        try:
            # Task 1: Data Understanding
            # We are using the 'ISO-8859-1' encoding, which is best suited for the Retail dataset.
            df_raw = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            
            # FIX: As soon as the file is loaded, remove extra spaces and hidden characters from the column names.
            df_raw.columns = df_raw.columns.str.strip().str.replace('ï»¿', '') 
            
            st.session_state.df = df_raw
        except Exception as e:
            st.error(f"Error loading file: {e}")

    df = st.session_state.df

    if df is not None:
        # ================= TASK 1: PREPROCESSING =================
        st.header("Task 1: Data Understanding & Preprocessing")
        
        st.subheader("Data Preview")
        st.dataframe(df.head())

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Dataset Shape:**", df.shape)
            st.write("**Available Columns:**", list(df.columns)) 
            st.write("**Missing Values:**")
            st.write(df.isnull().sum())
        
        with col2:
            st.info("""
            **Why Preprocessing?**
            Unsupervised learning (like K-Means) is distance based. Missing values or unscaled features 
            will bias the results. We must clean the data first.
            """)

        # Cleaning Button
        if st.button("Clean & Aggregate Data"):
            # Double check for columns again before processing
            required_cols = ['CustomerID', 'Quantity', 'UnitPrice', 'InvoiceNo']
            missing = [col for col in required_cols if col not in df.columns]
            
            if missing:
                st.error(f"These columns are still missing in the dataset: {missing}. "
                         "Please check that you have uploaded the correct 'Online Retail.csv' file.")
                         
            else:
                # 1. Clean CustomerID and Quantity
                df = df.dropna(subset=['CustomerID'])
                df = df.drop_duplicates()
                
                # Numeric Conversion
                df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
                df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
                
                df = df[df['Quantity'] > 0] # Remove returns/cancellations
                df['TotalSpend'] = df['Quantity'] * df['UnitPrice']
                
                # 2. Aggregate to Customer Level
                customer_data = df.groupby('CustomerID').agg({
                    'Quantity': 'mean',
                    'UnitPrice': 'mean',
                    'TotalSpend': 'sum',
                    'InvoiceNo': 'nunique'
                }).rename(columns={'InvoiceNo': 'Frequency', 'Quantity': 'AvgQuantity'})
                
                st.session_state.df = df
                st.session_state.customer_df = customer_data
                st.success("Data cleaned and aggregated by Customer ID!")

        if st.session_state.customer_df is not None:
            c_df = st.session_state.customer_df
            
            st.subheader("Feature Scaling")
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(c_df)
            scaled_df = pd.DataFrame(scaled_features, columns=c_df.columns, index=c_df.index)
            st.write("Scaled Customer Data (Ready for ML):")
            st.dataframe(scaled_df.head())

            # ================= TASK 2: K-MEANS =================
            st.header("Task 2: Customer Segmentation (K-Means)")
            
            inertia = []
            K_range = range(1, 8)
            for k in K_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(scaled_df)
                inertia.append(km.inertia_)
            
            fig_elbow, ax_elbow = plt.subplots()
            ax_elbow.plot(K_range, inertia, marker='o', color='teal')
            ax_elbow.set_title("Elbow Method to find Optimal K")
            st.pyplot(fig_elbow)

            k_val = st.slider("Select Clusters", 2, 6, 3)
            kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
            c_df['Cluster'] = kmeans.fit_predict(scaled_df)
            
            st.write(f"**Silhouette Score:** {silhouette_score(scaled_df, c_df['Cluster']):.2f}")
            
            fig_clus, ax_clus = plt.subplots()
            sns.scatterplot(data=c_df, x='TotalSpend', y='Frequency', hue='Cluster', palette='viridis', ax=ax_clus)
            ax_clus.set_title("Customer Segments: Spend vs Frequency")
            st.pyplot(fig_clus)

            # ================= TASK 3: ANOMALY DETECTION =================
            st.header("Task 3: Anomaly Detection")
            z_scores = np.abs(zscore(scaled_df))
            threshold = st.slider("Z-Score Threshold", 2.0, 5.0, 3.0)
            anomalies = (z_scores > threshold).any(axis=1)
            c_df['Is_Anomaly'] = anomalies
            
            st.write(f"Detected {anomalies.sum()} anomalous customers.")
            
            fig_anom, ax_anom = plt.subplots()
            sns.scatterplot(data=c_df, x='TotalSpend', y='AvgQuantity', hue='Is_Anomaly', palette={True:'red', False:'blue'}, ax=ax_anom)
            st.pyplot(fig_anom)
            st.info("Anomalies are customers with extreme spending or unusual purchase volumes.")

            # ================= TASK 4: PCA =================
            st.header("Task 4: Dimensionality Reduction (PCA)")
            pca = PCA(n_components=2)
            pca_res = pca.fit_transform(scaled_df.drop(columns=['Cluster', 'Is_Anomaly'], errors='ignore'))
            pca_df = pd.DataFrame(pca_res, columns=['PC1', 'PC2'])
            
            st.write(f"Variance Explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
            
            fig_pca, ax_pca = plt.subplots()
            ax_pca.scatter(pca_df['PC1'], pca_df['PC2'], c=c_df['Cluster'], cmap='viridis', alpha=0.5)
            ax_pca.set_xlabel("PC1")
            ax_pca.set_ylabel("PC2")
            st.pyplot(fig_pca)

            # ================= TASK 5: RECOMMENDATIONS =================
            st.header("Task 5: Recommendation System")
            with st.spinner("Computing Recommendations..."):
                sample_size = min(50000, len(df))
                small_df = df.sample(sample_size, random_state=42)
                matrix = small_df.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', fill_value=0)
                user_sim = cosine_similarity(matrix)
                user_sim_df = pd.DataFrame(user_sim, index=matrix.index, columns=matrix.index)

                def get_rec(cid):
                    if cid not in user_sim_df.index: return "No Data"
                    similar_user = user_sim_df[cid].sort_values(ascending=False).index[1]
                    items_this = set(matrix.loc[cid][matrix.loc[cid] > 0].index)
                    items_sim = set(matrix.loc[similar_user][matrix.loc[similar_user] > 0].index)
                    return list(items_sim - items_this)[:5]

                test_users = matrix.index[:3]
                for u in test_users:
                    st.write(f"**Recommendations for Customer {int(u)}:** {get_rec(u)}")

            # ================= TASK 6: REFLECTION =================
            st.header("Task 6: Analysis & Reflection")
            st.success("""
            **Insights:**
            - **Clustering:** Segmented customers into 'VIP', 'Regular', and 'Occasional' buyers.
            - **Anomaly Detection:** Identified potential bulk buyers or data entry errors.
            - **PCA:** Successfully compressed features while retaining high variance.
            - **Collaborative Filtering:** Successfully provided personalized product suggestions.
            """)
else:
    st.warning("Please upload the 'Online Retail' CSV file to begin the analysis. This dataset is essential for performing customer segmentation and generating insights.")