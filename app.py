# app.py

import streamlit as st
import pandas as pd
import requests
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel  # To load saved PySpark model

# ---------------------------
# 1. Initialize Spark Session
# ---------------------------
spark = SparkSession.builder.appName("NewsSentimentApp").getOrCreate()

# ---------------------------
# 2. Load Trained PySpark Model
# ---------------------------
# Assume you have saved your trained model in folder "sentiment_model"
# Save after training using: model.write().overwrite().save("sentiment_model")
model = PipelineModel.load("D:\news_sentimental\sentiment_model\sentiment_model")

# ---------------------------
# 3. Fetch Live News from GNews
# ---------------------------
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter GNews API Key", value="f923b3c1b261dba46561aa6c95a5bc54")
num_articles = st.sidebar.slider("Number of Headlines to Fetch", 5, 50, 10)

@st.cache_data(ttl=3600)  # Cache results for 1 hour
def fetch_headlines(api_key, max_articles):
    url = f"https://gnews.io/api/v4/top-headlines?token={api_key}&lang=en&max={max_articles}"
    response = requests.get(url)
    data = response.json()
    headlines = [article['title'] for article in data.get("articles", [])]
    return headlines

if api_key != "YOUR_GNEWS_API_KEY":
    headlines = fetch_headlines(api_key, num_articles)
else:
    st.warning("Please enter your GNews API Key in the sidebar to fetch headlines.")
    headlines = []

# ---------------------------
# 4. Convert Headlines to PySpark DataFrame
# ---------------------------
if headlines:
    headline_tuples = [(h,) for h in headlines]
    df_headlines = spark.createDataFrame(headline_tuples, ["headline"])

    # ---------------------------
    # 5. Classify Headlines with ML Model
    # ---------------------------
    predictions = model.transform(df_headlines)

    # Convert to Pandas for Streamlit visualization
    df_pandas = predictions.select("headline", "prediction").toPandas()

    # Map numeric labels to sentiment names
    label_map = {0: "Negative", 1: "Positive", 2: "Neutral", 3: "Compound"}
    df_pandas["Sentiment"] = df_pandas["prediction"].map(label_map)

    # ---------------------------
    # 6. Streamlit UI
    # ---------------------------
    st.title("📰 Real-Time News Sentiment Dashboard")
    st.markdown("Classifying latest news headlines into **Positive, Negative, Neutral, and Compound** categories using PySpark ML.")

    # Filter by sentiment
    sentiment_filter = st.multiselect("Filter by Sentiment", options=list(label_map.values()), default=list(label_map.values()))
    df_filtered = df_pandas[df_pandas["Sentiment"].isin(sentiment_filter)]

    # Display table of headlines
    st.subheader("Latest Headlines")
    st.dataframe(df_filtered[["headline", "Sentiment"]])

    # Display sentiment distribution
    st.subheader("Sentiment Distribution")
    sentiment_counts = df_filtered["Sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

    # Optional: Top headlines by sentiment
    st.subheader("Top Headlines by Sentiment")
    for sentiment in sentiment_filter:
        st.markdown(f"**{sentiment} Headlines:**")
        top_headlines = df_filtered[df_filtered["Sentiment"] == sentiment]["headline"].tolist()
        for i, h in enumerate(top_headlines, start=1):
            st.write(f"{i}. {h}")
else:
    st.info("Waiting for headlines to display. Enter your GNews API Key.")
