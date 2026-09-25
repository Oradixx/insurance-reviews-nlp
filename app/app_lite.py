"""Lighter version of the dashboard that only uses Hugging Face models (no Ollama needed):
BART for summaries and Flan-T5 for question answering. The main app (app.py) uses Llama 3 instead,
which hallucinated less than Flan-T5 in our tests (see the notebook's conclusion).

Run from the repository root:  streamlit run app/app_lite.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # avoids an OpenMP clash between torch and sklearn on macOS

import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from common import (is_valid_english, load_clean_dataset, retrieve_english_reviews, show_dashboard,
                    show_keywords, show_recent_reviews, show_sources)

st.set_page_config(page_title="Insurance NLP Dashboard", page_icon="📈", layout="wide")


@st.cache_resource
def load_models():
    # Star rating (1 to 5)
    sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    # Page 2: BART, trained to write one-sentence summaries
    summarizer = pipeline("summarization", model="facebook/bart-large-xsum")
    # Page 3: Flan-T5 for question answering
    qa_generator = pipeline("text2text-generation", model="google/flan-t5-base")
    # Sentence embeddings for semantic search (same model as in the notebook)
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return sentiment_model, summarizer, qa_generator, embedder


sentiment_model, summarizer, qa_generator, embedder = load_models()
df, all_embeddings = load_clean_dataset()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to:",
                            ["📊 Global Dashboard",
                             "🎯 Prediction & Explanation",
                             "📑 AI Summary by Insurer",
                             "🔍 RAG & QA Engine"])

st.sidebar.divider()
st.sidebar.caption("NLP Project #2")
st.sidebar.caption("Developed with Streamlit & HuggingFace")

# ==========================================
# PAGE 0: GLOBAL DASHBOARD
# ==========================================
if app_mode == "📊 Global Dashboard":
    show_dashboard(df)

# ==========================================
# PAGE 1: PREDICTION
# ==========================================
elif app_mode == "🎯 Prediction & Explanation":
    st.title("🎯 Rating Prediction")
    st.write("Test our model! Enter a review and the AI will predict the associated rating.")

    with st.container(border=True):
        user_input = st.text_area("✍️ Write a review here:",
                                  "The customer service is unreachable and prices have increased again this year, it's a shame!")
        predict_btn = st.button("Predict rating", type="primary")

    if predict_btn:
        with st.spinner('Analyzing with the neural network...'):
            result = sentiment_model(user_input)[0]
            score = result['score']
            stars_num = int(result['label'].split()[0])

            st.success("Analysis complete!")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 AI Prediction")
                st.metric(label="Predicted rating", value=f"{stars_num} / 5", delta="⭐" * stars_num)
                st.progress(score, text=f"AI Confidence: {score*100:.1f}%")

            with col2:
                show_keywords(user_input)

# ==========================================
# PAGE 2: SUMMARY (BART-LARGE-XSUM)
# ==========================================
elif app_mode == "📑 AI Summary by Insurer":
    st.title("📑 AI Executive Summary")

    if not df.empty and 'assureur' in df.columns:
        insurer = st.selectbox("🏢 Select an insurance company:", sorted(df['assureur'].dropna().unique().tolist()))

        if st.button("Generate Executive Summary", type="primary"):
            df_insurer = df[df['assureur'] == insurer]
            average_rating = df_insurer['note'].mean()

            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric(label=f"Average rating for {insurer}", value=f"{average_rating:.2f} ⭐",
                          delta=f"{len(df_insurer)} reviews")

            with col2:
                st.subheader("🤖 AI Generated Summary:")

                reviews = []
                for review in df_insurer['avis_cor_en'].dropna():
                    if is_valid_english(review):
                        reviews.append(str(review).strip())
                    if len(reviews) == 6:
                        break

                text_to_summarize = ". ".join(reviews)

                if len(text_to_summarize) > 50:
                    with st.spinner("Generating summary using BART-Large-XSUM..."):
                        try:
                            # XSum-style model: one clean summary sentence
                            summary = summarizer(text_to_summarize[:1500], max_length=50, min_length=15, do_sample=False)
                            st.success(f"**{summary[0]['summary_text']}**")
                        except Exception as e:
                            st.error(f"Error during generation: {e}")
                else:
                    st.info("Not enough English data to generate a relevant summary.")

            show_recent_reviews(df_insurer)

# ==========================================
# PAGE 3: RAG & QA ENGINE (FLAN-T5)
# ==========================================
elif app_mode == "🔍 RAG & QA Engine":
    st.title("🔍 RAG System (Retrieval-Augmented Generation)")
    st.write("Ask a question. The AI retrieves context and generates a natural answer in English.")

    question = st.text_input("❓ Your question (in English):", "Why do people complain about the pricing?")

    if st.button("Search & Generate Answer", type="primary"):
        if not df.empty and all_embeddings is not None and question:

            with st.spinner("1. Semantic Retrieval (MiniLM)..."):
                top_indices = retrieve_english_reviews(question, df, all_embeddings, embedder, k=5)
                rag_context = " . ".join([str(df.iloc[idx]['avis_cor_en']) for idx in top_indices])

            with st.spinner("2. Generating Answer (Flan-T5)..."):
                try:
                    # Ask for a synthesized answer so the model does not just copy the context
                    rag_prompt = f"""Question: {question}

Based on the context below, provide a short and synthesized answer. Do not copy and paste the text.

Context: {rag_context}

Short Answer:"""

                    answer = qa_generator(rag_prompt, max_length=60, do_sample=False)
                    answer_text = answer[0]['generated_text']

                    if "unanswerable" in answer_text.lower() or len(answer_text) < 5:
                        st.warning("⚠️ The AI could not find a clear answer in the available customer reviews.")
                    else:
                        st.success("✅ Context analyzed & Answer generated!")
                        st.markdown(f"### 🤖 AI Answer:\n> **{answer_text.capitalize()}**")

                except Exception as e:
                    st.error(f"The AI failed to generate an answer. Error: {e}")

            show_sources(df, top_indices)
