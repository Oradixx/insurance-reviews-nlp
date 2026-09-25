"""Insurance reviews dashboard: rating prediction, zero-shot topic detection,
summaries and question answering with Llama 3 running locally through Ollama.

Run from the repository root:  streamlit run app/app.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # avoids an OpenMP clash between torch and sklearn on macOS

import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from common import (is_valid_english, load_clean_dataset, retrieve_english_reviews, show_dashboard,
                    show_keywords, show_recent_reviews, show_sources)

LLM_MODEL = 'llama3'
TOPICS = ["Pricing", "Coverage", "Enrollment", "Customer Service", "Claims Processing", "Cancellation"]

st.set_page_config(page_title="Insurance NLP Dashboard", page_icon="🚀", layout="wide")


@st.cache_resource
def load_models():
    # 1. Star rating (1 to 5)
    sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    # 2. Zero-shot classifier for the review's main subject
    zero_shot_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    # 3. Sentence embeddings for semantic search (same model as in the notebook)
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return sentiment_model, zero_shot_model, embedder


sentiment_model, zero_shot_model, embedder = load_models()
df, all_embeddings = load_clean_dataset()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to:", ["📊 Global Dashboard", "🎯 Prediction & Explanation",
                                       "📑 AI Summary by Insurer", "🔍 RAG & QA Engine"])
st.sidebar.divider()
st.sidebar.caption("⚡ Powered by Llama 3 (local, via Ollama)")

# ==========================================
# PAGE 0: GLOBAL DASHBOARD
# ==========================================
if app_mode == "📊 Global Dashboard":
    show_dashboard(df)

# ==========================================
# PAGE 1: PREDICTION (STARS + ZERO-SHOT TOPIC)
# ==========================================
elif app_mode == "🎯 Prediction & Explanation":
    st.title("🎯 Rating & Subject Prediction")
    st.write("Test our model! The AI will predict the rating and detect the main subject of the review.")

    with st.container(border=True):
        user_input = st.text_area("✍️ Write a review here:",
                                  "The customer service is unreachable and prices have increased again this year, it's a shame!")
        predict_btn = st.button("Predict Rating & Topic", type="primary")

    if predict_btn:
        with st.spinner('Analyzing with Neural Networks...'):
            # 1. Star rating
            result = sentiment_model(user_input)[0]
            stars_num = int(result['label'].split()[0])

            # 2. Main subject (zero-shot)
            zero_shot_result = zero_shot_model(user_input, candidate_labels=TOPICS)
            top_category = zero_shot_result['labels'][0]
            top_cat_score = zero_shot_result['scores'][0]

            st.success("Analysis complete!")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("📊 Star Prediction")
                st.metric(label="Predicted rating", value=f"{stars_num} / 5", delta="⭐" * stars_num)
                st.progress(result['score'], text=f"Confidence: {result['score']*100:.1f}%")

            with col2:
                st.subheader("🏷️ Topic Detection")
                st.metric(label="Primary Subject", value=f"{top_category}")
                st.progress(top_cat_score, text=f"Zero-Shot Confidence: {top_cat_score*100:.1f}%")

            with col3:
                show_keywords(user_input)

# ==========================================
# PAGE 2: SUMMARY BY INSURER (LLAMA 3)
# ==========================================
elif app_mode == "📑 AI Summary by Insurer":
    st.title("📑 AI Executive Summary (Llama 3)")

    if not df.empty and 'assureur' in df.columns:
        insurer = st.selectbox("🏢 Select an insurance company:", sorted(df['assureur'].dropna().unique().tolist()))

        if st.button("Generate Executive Summary", type="primary"):
            df_insurer = df[df['assureur'] == insurer]

            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric(label="Average rating", value=f"{df_insurer['note'].mean():.2f} ⭐",
                          delta=f"{len(df_insurer)} reviews")

            with col2:
                st.subheader("🤖 AI Generated Summary:")
                reviews = [str(r).strip() for r in df_insurer['avis_cor_en'].dropna() if is_valid_english(r)][:10]
                text_to_summarize = "\n".join([f"- {r}" for r in reviews])

                if len(text_to_summarize) > 50:
                    with st.spinner("Llama 3 is analyzing the reviews..."):
                        try:
                            response = ollama.chat(model=LLM_MODEL, messages=[
                                {'role': 'system', 'content': 'You are an expert data analyst. Read the following customer reviews and write a clear, 2-sentence executive summary highlighting the main positive and negative points. Do not invent information.'},
                                {'role': 'user', 'content': text_to_summarize}
                            ])
                            st.success(f"**{response['message']['content']}**")
                        except Exception as e:
                            st.error(f"Make sure Ollama is running (ollama serve) and the model is pulled (ollama pull {LLM_MODEL}). Error: {e}")
                else:
                    st.info("Not enough English data.")

            show_recent_reviews(df_insurer)

# ==========================================
# PAGE 3: RAG & QA ENGINE (LLAMA 3)
# ==========================================
elif app_mode == "🔍 RAG & QA Engine":
    st.title("🔍 RAG System (Llama 3)")
    question = st.text_input("❓ Your question (in English):", "Why do people complain about the pricing?")

    if st.button("Search & Generate Answer", type="primary"):
        if not df.empty and all_embeddings is not None and question:

            with st.spinner("1. Semantic Retrieval (MiniLM)..."):
                top_indices = retrieve_english_reviews(question, df, all_embeddings, embedder, k=5)
                rag_context = "\n".join([f"Review {i+1}: {str(df.iloc[idx]['avis_cor_en'])}"
                                         for i, idx in enumerate(top_indices)])

            with st.spinner("2. Generating Answer (Llama 3)..."):
                try:
                    prompt = f"""Use ONLY the following reviews to answer the question. Answer naturally in one or two sentences. If the answer is not in the reviews, say 'I cannot find the answer in the provided reviews'.

Reviews:
{rag_context}

Question: {question}"""

                    response = ollama.chat(model=LLM_MODEL, messages=[
                        {'role': 'system', 'content': 'You are a helpful customer support assistant.'},
                        {'role': 'user', 'content': prompt}
                    ])

                    st.success("✅ Context analyzed & Answer generated!")
                    st.markdown(f"### 🤖 Llama 3 Answer:\n> **{response['message']['content']}**")

                except Exception as e:
                    st.error(f"Make sure Ollama is running! Error: {e}")

            show_sources(df, top_indices)
