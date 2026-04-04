import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

# FUNCTIONS =========================
# for nice plotting
def without_hue(plot, feature, yp=20):
    total = len(feature)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width() / 2 - 0.08
        y = p.get_y() + p.get_height() + yp
        ax.annotate(percentage, (x, y), size=10)
    plt.show()

def adjusted_prediction(model, text, priors, labels):
    # X = vectorizer.transform([text])
    X = [text]
    # X = pd.DataFrame({'clean_text': [text]})

    probs = model.predict_proba(X)[0]  # shape: (3,)

    adjusted = []
    for i, p in enumerate(probs):
        adj = p / ((0.5 + priors[i]) / 2)
        adjusted.append(adj)

    adjusted = np.array(adjusted)

    # normalize (VERY IMPORTANT)
    adjusted = adjusted / adjusted.sum()

    label_idx = np.argmax(adjusted)

    return labels[label_idx], adjusted

def get_topic(lda_model, vectorizer, text, sentiment_label=None):
    X = vectorizer.transform([text])
    if sentiment_label=="negative":
        lda_model = lda_neg
    else: lda_model = lda_pos
    
    topic_dist = lda_model.transform(X)
    topic_idx = topic_dist.argmax()

    return topic_idx, topic_dist

# from utils import sparse_to_array
def sparse_to_array(X):
    return X.toarray()

# LOAD DATA =========================
@st.cache_data
def load_data():
    df = pd.read_csv("data/df_final.csv")
    return df

df = load_data()

@st.cache_resource
def load_model():
    model = joblib.load('models/basic_model_lr.joblib')
    vectorizer = joblib.load('models/vectorizer.joblib')
    lda_neg = joblib.load('models/lda_neg.joblib')
    lda_pos = joblib.load('models/lda_pos.joblib')

    return model, vectorizer, lda_neg, lda_pos

model, vectorizer, lda_neg, lda_pos = load_model()

# MODEL idk donwloader?
labels = model.classes_.tolist()
priors = df["sentiment_label"].value_counts(normalize=True).reindex(labels).fillna(0).tolist()

# SIDEBAR =========================
st.title("🛍️ Tokopedia Review Analysis")

tab1, tab3, tab4 = st.tabs(["Overview", "Topics", "Explorer"])

# st.sidebar.title("Navigation")
# page = st.sidebar.radio(
#     "Go to",
#     ["Overview", "Sentiment Analysis", "Topic Insights", "Review Explorer"]
# )

# OVERVIEW =========================
with tab1:
    st.header("Overview")

    total_reviews = len(df)
    sentiment_counts = df["sentiment_label"].value_counts()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", total_reviews)
    col2.metric("Positive Reviews", sentiment_counts.get("positive", 0))
    col3.metric("Negative Reviews", sentiment_counts.get("negative", 0))

    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="sentiment_label", data=df, palette="muted")
    without_hue(ax, df.sentiment_label, yp=200)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig)

    st.markdown("""
    **Insight:** See the imbalanced dataset over there? With positive reviews 
    account for around 93 percent? I thought of doing nothing about it, but then remember
    I got one card on my sleeve: probability adjustment! (Well, it's kinda similar with
    my final project.)
    """)

    st.subheader("📉 Quarterly Negative Trend (%)")

    df["review_date"] = pd.to_datetime(df["review_date"])
    df["quarter"] = df["review_date"].dt.to_period("Q")

    negative_by_quarter = df[df["sentiment_label"] == "negative"].groupby("quarter").size()
    total_by_quarter = df.groupby("quarter").size()
    negative_pct = (negative_by_quarter / total_by_quarter * 100).fillna(0)

    negative_pct.index = negative_pct.index.astype(str)
    
    fig, ax = plt.subplots()
    negative_pct.plot(ax=ax)

    ax.set_xlabel("")  # remove label
    ax.set_ylabel("Negative (%)")
    step = 4
    ax.set_xticks(range(0, len(negative_pct), step))
    ax.set_xticklabels(negative_pct.index[::step], rotation=45)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig)
    
    st.markdown("""
    **Insight:** Except of few anomalies in the early years (probably due
    to a fewer user or IDK), the proportion of negative review tends to be stable, 
    at least in a long period, although in short period, there seems to have a bit 
    of fluctuations. You may check them in detail, but for now, who cares? 
    """)

    st.subheader("Review Length Distribution")
    fig, ax = plt.subplots()
    sns.boxenplot(df.clean_length[df.clean_length<=df.clean_length.quantile(0.99)])
    ax.set_xlabel("Under 99th Percentile")
    ax.set_ylabel("Review Length")
    st.pyplot(fig)

    st.markdown("""
    **Insight:** Most reviews (~99 percent) have short length (under 60 words), with 
    median just under 10 words. Therefore, I don't think a complex model that can understand 
    context will help with the classification. Rather, I chose a simpler approach 
    based on keywords, using TF-IDF. 
                """)

# SENTIMENT ANALYSIS =========================
# with tab2:
#     st.header("Sentiment Analysis")

#     sentiment = st.selectbox(
#         "Select Sentiment",
#         df["sentiment_label"].unique()
#     )

#     filtered = df[df["sentiment_label"] == sentiment]

#     st.write(f"Showing {len(filtered)} reviews")

#     for i, row in filtered.head(10).iterrows():
#         st.write(f"👉 {row['review_text']}")


# TOPIC INSIGHTS =========================
with tab3:
    st.header("Topic Insights")

    st.subheader("Negative Review Topics")
    st.markdown("""
    **1. Product quality issues**: damaged, spoiled, or poor quality items  

    **2. Delivery issues**: shipping delays, courier problems  

    **3. Product mismatch**: wrong size, color, or material
    """)

    fig, ax = plt.subplots()
    neg_df = df[df["sentiment_label"] == "negative"]
    sns.countplot(x="dominant_topic", data=neg_df, palette="muted")
    without_hue(ax, neg_df["dominant_topic"], yp=3)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("Positive Review Topics")

    st.markdown("""
    **1. Accurate orders**:matches description  

    **2. Good product quality**:satisfying quality  

    **3. Fast delivery & packaging**: quick shipping, safe packaging  
    """)

    fig, ax = plt.subplots()
    pos_df = df[df["sentiment_label"] == "positive"]
    sns.countplot(x="dominant_topic", data=pos_df, palette="muted")
    without_hue(ax, pos_df["dominant_topic"], yp=100)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("Sample Reviews")

    topics = st.selectbox(
        "Select Topics",
        df["dominant_topic"].unique()
    )

    filtered = df[df["dominant_topic"] == topics]

    st.write(f"Showing {len(filtered)} reviews")

    for i, row in filtered.head(5).iterrows():
        st.write(f"👉 {row['review_text']}")


# REVIEW EXPLORER =========================
with tab4:
    st.header("🔍 Review Explorer")

    user_input = st.text_area(
        "Input the review:",
        placeholder="Example: barang datang terlambat, kurir lama, dan kemasan rusak..."
    )

    if st.button("Analyze"):
        if user_input.strip() == "":
            st.warning("Input cannot be empty")
        else:
            # --- Sentiment ---
            label, probs = adjusted_prediction(
                model, user_input, priors, labels
            )

            st.subheader("Sentiment Prediction")

            emoji_map = {
                "negative": "😡",
                "neutral": "😐",
                "positive": "😊"
            }

            st.markdown(f"### {label.upper()} {emoji_map[label]}")

            # --- Probability display ---
            st.write("Adjusted Confidence:")

            sorted_idx = np.argsort(probs)[::-1]
            for i in sorted_idx:
                st.write(f"- {labels[i]}: {probs[i]:.2f}")

            # --- Topic ---
            topic_idx, topic_dist = get_topic(
                lda_neg, vectorizer, user_input, sentiment_label=label
            )

            # mapping if label == negative: from 012 to 120
            if label == "negative":
                topic_idx = (topic_idx + 1) % 3

            topic_labels = {
                0: "Product Expectation Match",
                1: "Product Quality",
                2: "Delivery & Packaging"
            }

            st.subheader("Topic Insight")
            st.write(f"Most related topic: **{topic_labels.get(topic_idx, 'Unknown')}**")


# # Temporary code in app.py
# if st.sidebar.button("🔨 Emergency Retrain"):
#     import subprocess
#     subprocess.run(["python", "scripts/train.py"])
#     st.sidebar.success("Model retrained on Cloud hardware!")