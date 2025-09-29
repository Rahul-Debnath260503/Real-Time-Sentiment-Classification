import streamlit as st
import requests
from textblob import TextBlob
import pandas as pd

# Set up your NewsAPI key
API_KEY = "f923b3c1b261dba46561aa6c95a5bc54"

# Function to fetch news from NewsAPI
def fetch_news(query):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize=50&apiKey={API_KEY}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return articles

# Function to analyze sentiment
def analyze_sentiment(text):
    if not text:
        return "neutral"
    score = TextBlob(text).sentiment.polarity
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"

# Streamlit UI
st.set_page_config(page_title="Real-Time News Sentiment", layout="wide")
st.title("📰 Real-Time News Sentiment Analysis")

# Query Input
query = st.text_input("Enter a news topic (e.g., AI, sports, elections):", value="Artificial Intelligence")

if query:
    with st.spinner("Fetching and analyzing news..."):
        articles = fetch_news(query)
        if not articles:
            st.warning("No news articles found.")
        else:
            titles = []
            descriptions = []
            sentiments = []

            for article in articles:
                title = article["title"]
                desc = article["description"] or ""
                sentiment = analyze_sentiment(desc)

                titles.append(title)
                descriptions.append(desc)
                sentiments.append(sentiment)

            # Create DataFrame
            df = pd.DataFrame({
                "Title": titles,
                "Description": descriptions,
                "Sentiment": sentiments
            })

            # Display sentiment counts
            st.subheader("📊 Sentiment Distribution")
            sentiment_counts = df["Sentiment"].value_counts()
            st.bar_chart(sentiment_counts)

            # Filter option
            st.subheader("🔍 View Articles by Sentiment")
            selected_sentiment = st.radio("Select Sentiment", ["all", "positive", "neutral", "negative"])
            if selected_sentiment != "all":
                filtered_df = df[df["Sentiment"] == selected_sentiment]
            else:
                filtered_df = df

            # Display Data
            st.dataframe(filtered_df[["Title", "Sentiment"]])
