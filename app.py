# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 16:30:25 2023

@author: excel
"""

import streamlit as st
import numpy as np
import joblib

# Load saved models
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
kmeans = joblib.load("kmeans_model.pkl")

st.title("Customer Segmentation App (K=3 Model)")
st.write("Enter Customer Details:")

# ----------- INPUT FIELDS -----------

income = st.number_input("Income", value=58138.0)
kidhome = st.number_input("Kidhome", value=0.0)
teenhome = st.number_input("Teenhome", value=0.0)
recency = st.number_input("Recency", value=58.0)

mntwines = st.number_input("MntWines", value=635.0)
mntfruits = st.number_input("MntFruits", value=88.0)
mntmeatproducts = st.number_input("MntMeatProducts", value=546.0)
mntfishproducts = st.number_input("MntFishProducts", value=172.0)
mntsweetproducts = st.number_input("MntSweetProducts", value=88.0)
mntgoldprods = st.number_input("MntGoldProds", value=88.0)

numdealspurchases = st.number_input("NumDealsPurchases", value=3.0)
numwebpurchases = st.number_input("NumWebPurchases", value=8.0)
numcatalogpurchases = st.number_input("NumCatalogPurchases", value=10.0)
numstorepurchases = st.number_input("NumStorePurchases", value=4.0)
numwebvisitsmonth = st.number_input("NumWebVisitsMonth", value=7.0)

age = st.number_input("Age", value=57.0)
total_spending = st.number_input("Total_Spending", value=1617.0)
tenure = st.number_input("Tenure", value=2.0)

education_Basic = st.number_input("Education_Basic (0/1)", value=0.0)
education_Graduation = st.number_input("Education_Graduation (0/1)", value=1.0)
education_Master = st.number_input("Education_Master (0/1)", value=0.0)
education_PhD = st.number_input("Education_PhD (0/1)", value=0.0)

marital_status_Alone = st.number_input("Marital_Alone (0/1)", value=0.0)
marital_status_Divorced = st.number_input("Marital_Divorced (0/1)", value=0.0)
marital_status_Married = st.number_input("Marital_Married (0/1)", value=0.0)
marital_status_Single = st.number_input("Marital_Single (0/1)", value=1.0)
marital_status_Together = st.number_input("Marital_Together (0/1)", value=0.0)
marital_status_Widow = st.number_input("Marital_Widow (0/1)", value=0.0)
marital_status_YOLO = st.number_input("Marital_YOLO (0/1)", value=0.0)

# ----------- PREDICTION -----------

if st.button("Predict Cluster"):

    input_data = np.array([[ 
        income, kidhome, teenhome, recency,
        mntwines, mntfruits, mntmeatproducts,
        mntfishproducts, mntsweetproducts, mntgoldprods,
        numdealspurchases, numwebpurchases,
        numcatalogpurchases, numstorepurchases,
        numwebvisitsmonth, age, total_spending,
        tenure,
        education_Basic, education_Graduation,
        education_Master, education_PhD,
        marital_status_Alone, marital_status_Divorced,
        marital_status_Married, marital_status_Single,
        marital_status_Together, marital_status_Widow,
        marital_status_YOLO
    ]])

    scaled = scaler.transform(input_data)
    pca_transformed = pca.transform(scaled)
    cluster = kmeans.predict(pca_transformed)[0]

    # -------- Cluster Interpretation --------

    if cluster == 0:
        segment = "Budget / Low Spending Customer"
        insight = "This customer has low purchasing activity. Target with discount offers and promotional campaigns."
    elif cluster == 1:
        segment = "Premium / High Value Customer"
        insight = "This customer contributes high revenue. Ideal for loyalty programs and exclusive offers."
    else:
        segment = "Regular / Moderate Customer"
        insight = "This customer shows moderate engagement. Target with personalized recommendations."

    # -------- Spending Level --------

    if total_spending > 1500:
        spending_level = "High Spending"
    elif total_spending > 800:
        spending_level = "Moderate Spending"
    else:
        spending_level = "Low Spending"

    # -------- Output --------

    st.success(f"Customer belongs to Cluster {cluster}")
    st.subheader(f"Segment: {segment}")
    st.write(f"Spending Level: {spending_level}")
    st.info(f"Business Insight: {insight}")

