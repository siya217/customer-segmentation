import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.title("Customer Segmentation using Clustering")

st.write("App started...")

# Load data
df = pd.read_excel("marketing_campaign1.xlsx")

st.write("Data loaded successfully")

# Cleaning
df['Income'].fillna(df['Income'].median(), inplace=True)
df['Age'] = 2025 - df['Year_Birth']

features = [
    'Age','Income','Recency',
    'MntWines','MntFruits','MntMeatProducts',
    'MntFishProducts','MntSweetProducts','MntGoldProds',
    'NumWebPurchases','NumStorePurchases','NumCatalogPurchases'
]

X = df[features]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

st.subheader("Clustered Customers")
st.dataframe(df[['ID','Age','Income','Cluster']])

st.subheader("Cluster Summary")
st.dataframe(df.groupby('Cluster')[features].mean())


