import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="RetailMiner Dashboard", layout="wide")

st.title("🛒 RetailMiner – Supermarket Transaction Intelligence Platform")

# ---------------- LOAD DATA ---------------- #

@st.cache_data
def load_data():
    anomalies = pd.read_csv(r"H:\PROJECTS\AIML PROJECTS\RetailMiner\outputs\anomalies.csv")
    clusters = pd.read_csv(r"H:\PROJECTS\AIML PROJECTS\RetailMiner\outputs\clusters.csv")
    rules = pd.read_csv(r"H:\PROJECTS\AIML PROJECTS\RetailMiner\outputs\rules.csv")
    peak_hours = pd.read_csv(r"H:\PROJECTS\AIML PROJECTS\RetailMiner\outputs\peak_hours.csv")
    monthly_sales = pd.read_csv(r"H:\PROJECTS\AIML PROJECTS\RetailMiner\outputs\monthly_sales.csv")
    revenue_spikes = pd.read_csv(r"H:\PROJECTS\AIML PROJECTS\RetailMiner\outputs\revenue_spikes.csv")
    return anomalies, clusters, rules, peak_hours, monthly_sales, revenue_spikes

anomalies, clusters, rules, peak_hours, monthly_sales, revenue_spikes = load_data()

# ---------------- SIDEBAR ---------------- #

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Anomaly Analytics",
        "Customer Segmentation",
        "Market Basket",
        "Seasonal Trends",
        "Model Info"
    ]
)

# ---------------- DASHBOARD ---------------- #

if menu == "Dashboard":

    st.subheader("📊 Key Metrics")

    c1,c2,c3 = st.columns(3)
    c1.metric("Total Anomalies", anomalies.shape[0])
    c2.metric("Total Customers", clusters.shape[0])
    c3.metric("Association Rules", rules.shape[0])

    st.divider()

    col1,col2 = st.columns(2)

    with col1:
        fig = px.bar(peak_hours,x="Hour",y=peak_hours.columns[1],title="Peak Shopping Hours")
        st.plotly_chart(fig,width="stretch")

    with col2:
        fig = px.pie(clusters,names="Cluster",title="Customer Distribution by Cluster")
        st.plotly_chart(fig,width="stretch")

    st.subheader("📈 Monthly Revenue")
    fig = px.line(monthly_sales,x="MonthYear",y="TotalAmount")
    st.plotly_chart(fig,width="stretch")

# ---------------- ANOMALY ---------------- #

elif menu == "Anomaly Analytics":

    st.subheader("⚠️ Anomaly Distribution")

    fig = px.histogram(anomalies,x="InvoiceTotal",nbins=50,title="Abnormal Invoice Amounts")
    st.plotly_chart(fig,width="stretch")

    st.subheader("Detected Anomalies")
    st.dataframe(anomalies)

# ---------------- CLUSTERS ---------------- #

elif menu == "Customer Segmentation":

    st.subheader("👥 Customer Segments")

    fig = px.scatter(clusters,x="Frequency",y="Monetary",color="Cluster")
    st.plotly_chart(fig,width="stretch")

    fig2 = px.histogram(clusters,x="Cluster",title="Cluster Count")
    st.plotly_chart(fig2,width="stretch")

    st.dataframe(clusters)

# ---------------- MARKET BASKET ---------------- #

elif menu == "Market Basket":

    st.subheader("🛒 Top Association Rules")

    top_rules = rules.sort_values("lift",ascending=False).head(20)
    st.dataframe(top_rules[['antecedents','consequents','support','confidence','lift']])

    fig = px.bar(top_rules,x="lift",y=top_rules.index.astype(str),orientation="h",title="Top Lift Rules")
    st.plotly_chart(fig,width="stretch")

# ---------------- SEASONAL ---------------- #

elif menu == "Seasonal Trends":

    st.subheader("📅 Monthly Sales")

    fig = px.line(monthly_sales,x="MonthYear",y="TotalAmount")
    st.plotly_chart(fig,width="stretch")

    st.subheader("🎉 Revenue Spikes")
    st.dataframe(revenue_spikes)

# ---------------- MODEL INFO ---------------- #

elif menu == "Model Info":

    st.subheader("🤖 Models Used")

    st.markdown("""
    ### 🔹 Isolation Forest
    - Purpose: Detect abnormal billing patterns  
    - Features: InvoiceTotal, ItemCount, Hour  

    ### 🔹 K-Means Clustering
    - Purpose: Customer segmentation  
    - Features: Recency, Frequency, Monetary, AvgSpend  

    ### 🔹 Apriori Algorithm
    - Purpose: Market basket analysis  
    - Output: Association rules  

    ### 🔹 Temporal Mining
    - Peak shopping hours  
    - Monthly trends   
    - Revenue spikes  
    """)
    st.subheader("📁 Project Outputs")

    st.write("Anomalies:", anomalies.shape)
    st.write("Clusters:", clusters.shape)
    st.write("Rules:", rules.shape)

# ---------------- FOOTER ---------------- #

st.sidebar.success("RetailMiner Platform Active 🚀")
