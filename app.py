import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# LOAD DATA =========================
@st.cache_data
def load_data():
    df1 = pd.read_csv("data/df_test.csv") 
    df2 = pd.read_csv("data/df_train.csv")
    df = pd.concat([df1, df2], ignore_index=True)
    return df

df = load_data()

# SIDEBAR =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Sentiment Analysis", "Topic Insights", "Review Explorer"]
)

st.title("🛍️ Tokopedia Review Analysis")


# OVERVIEW =========================
if page == "Overview":
    st.header("📊 Overview")

    total_reviews = len(df)
    sentiment_counts = df["sentiment"].value_counts()

    col1, col2 = st.columns(2)
    col1.metric("Total Reviews", total_reviews)
    col2.metric("Positive Reviews", sentiment_counts.get("positive", 0))

    st.subheader("Sentiment Distribution")

    fig, ax = plt.subplots()
    sentiment_counts.plot(kind="bar", ax=ax)
    st.pyplot(fig)

# SENTIMENT ANALYSIS =========================
elif page == "Sentiment Analysis":
    st.header("💬 Sentiment Analysis")

    sentiment = st.selectbox(
        "Select Sentiment",
        df["sentiment"].unique()
    )

    filtered = df[df["sentiment"] == sentiment]

    st.write(f"Showing {len(filtered)} reviews")

    for i, row in filtered.head(10).iterrows():
        st.write(f"👉 {row['review_text']}")


# TOPIC INSIGHTS =========================
elif page == "Topic Insights":
    st.header("🧠 Topic Insights")

    st.subheader("🔴 Negative Review Topics")

    st.markdown("""
    **1. Product Quality Issues**  
    - damaged / spoiled items  

    **2. Delivery Issues**  
    - shipping delays, courier problems  

    **3. Product Mismatch**  
    - wrong size, color, expectations  
    """)

    st.subheader("🟢 Positive Review Topics")

    st.markdown("""
    **1. Accurate Orders**  
    - matches description  

    **2. Good Product Quality**  
    - satisfying quality  

    **3. Fast Delivery & Packaging**  
    - quick shipping, safe packaging  
    """)


# REVIEW EXPLORER =========================
elif page == "Review Explorer":
    st.header("🔍 Explore Reviews")

    sentiment_filter = st.selectbox(
        "Filter by Sentiment",
        df["sentiment"].unique()
    )

    filtered = df[df["sentiment"] == sentiment_filter]

    st.write(f"Total: {len(filtered)} reviews")

    st.dataframe(filtered[["review_text", "sentiment"]].head(50))