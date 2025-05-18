
import streamlit as st
st.set_page_config(page_title="Job Recommender", layout="centered")
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# File paths
DATA_PATH = "data/job_posting.csv"
MODEL_PATH = "models/tfidf_vectorizer.pkl"

# Load data and model
@st.cache_data
def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        data.dropna(subset=["job_description"], inplace=True)
        data["job_description"] = data["job_description"].fillna("").astype(str)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        raise

@st.cache_resource
def load_model(filepath):
    try:
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise

# Recommend Jobs
def recommend_jobs(user_input, tfidf_matrix, vectorizer, data):
    user_query_tfidf = vectorizer.transform([user_input])
    cosine_sim = cosine_similarity(user_query_tfidf, tfidf_matrix).flatten()
    data["similarity_score"] = cosine_sim
    recommended_jobs = (
        data.sort_values(by="similarity_score", ascending=False)
        .head(5)
        .loc[:, ["Cleaned Job Title", "Category", "country", "average_hourly_rate", "link"]]
    )
    return recommended_jobs

# Streamlit interface

def main():
    # 🔵 CSS Styling
    st.markdown("""
        <style>
            body {
                background-color: #87CEEB;
                font-family: 'Segoe UI', sans-serif;
            }
            .title {
                font-size: 36px;
                color: #2c3e50;
                text-align: center;
                font-weight: bold;
                padding-bottom: 10px;
            }
            .subtitle {
                text-align: center;
                color: #34495e;
                font-size: 18px;
                padding-bottom: 20px;
            }
            .job-card {
                background-color: #ffffff;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            }
            .job-title {
                color: #2980b9;
                font-size: 22px;
                font-weight: 600;
            }
            .job-detail {
                font-size: 16px;
                color: #2c3e50;
                margin-top: 5px;
            }
            .job-link {
                margin-top: 10px;
                display: inline-block;
                background-color: #27ae60;
                color: white;
                padding: 8px 12px;
                border-radius: 5px;
                text-decoration: none;
            }
        </style>
    """, unsafe_allow_html=True)

    # 🔵 Page Content
    st.markdown('<div class="title">🔍Job Recommendation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Enter your desired job title or description to get personalized job matches!</div>', unsafe_allow_html=True)

    # Load data and model
    data = load_data(DATA_PATH)
    vectorizer = load_model(MODEL_PATH)

    st.success(f"✅ Loaded {len(data)} job listings.")
    st.info("🧠 Model loaded and ready to recommend.")

    # Input
    user_input = st.text_input("Job Title/Description:", "")

    if user_input:
        tfidf_matrix = vectorizer.transform(data["job_description"])
        recommendations = recommend_jobs(user_input, tfidf_matrix, vectorizer, data)

        if not recommendations.empty:
            st.markdown("### 🎯 Top Recommendations:")
            for _, row in recommendations.iterrows():
                st.markdown(f"""
                    <div class="job-card">
                        <div class="job-title">{row['Cleaned Job Title']}</div>
                        <div class="job-detail">📂 Category: {row['Category']}</div>
                        <div class="job-detail">🌍 Location: {row['country']}</div>
                        <div class="job-detail">💵 Hourly Rate: ${row['average_hourly_rate']}</div>
                        <a class="job-link" href="{row['link']}" target="_blank">🔗 View Job</a>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("🚫 No recommendations found. Please try a different input.")
    else:
        st.info("ℹ️ Please enter a job title or description to see recommendations.")

if __name__ == "__main__":
    main()
