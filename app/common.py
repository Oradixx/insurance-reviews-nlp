"""Shared helpers for the two Streamlit apps: data loading and the English-text filter."""
import pickle
import re
from pathlib import Path

import pandas as pd
import streamlit as st

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CSV_PATH = PROCESSED_DIR / "reviews_clean.csv"
EMBEDDINGS_PATH = PROCESSED_DIR / "review_embeddings.pkl"

# Words that betray an untranslated (French) review, and words expected in an English one.
FRENCH_MARKERS = [r'\ble\b', r'\bla\b', r'\bles\b', r'\bdes\b', r'\bun\b', r'\bune\b', r'\best\b', r'\bsont\b',
                  r'\bje\b', r'\bil\b', r'\belle\b', r'\bnous\b', r'\bvous\b', r'\bpour\b', r'\bque\b', r'\bqui\b',
                  r'\bdans\b', r'\bsur\b', r'\bpas\b', r'\bplus\b', r'\bc est\b', r'\bj ai\b', r'\bou\b', r'\bavec\b']
ENGLISH_MARKERS = [r'\bthe\b', r'\bis\b', r'\bto\b', r'\band\b', r'\bof\b', r'\bin\b', r'\bit\b', r'\bfor\b',
                   r'\bwith\b', r'\bmy\b', r'\byou\b', r'\bthat\b', r'\bprices?\b', r'\bcar\b', r'\binsurance\b',
                   r'\bnot\b', r'\bbut\b', r'\bi\b']


def is_valid_english(text):
    """Return True only for text that looks like proper English.

    Some translations failed and kept French or mixed words, so reviews with 2+ French markers
    are rejected, and at least one common English word is required.
    """
    if not isinstance(text, str) or len(text) < 5:
        return False
    text_lower = text.lower()
    if sum(1 for pattern in FRENCH_MARKERS if re.search(pattern, text_lower)) >= 2:
        return False
    return sum(1 for pattern in ENGLISH_MARKERS if re.search(pattern, text_lower)) >= 1


@st.cache_data
def load_clean_dataset():
    """Load the cleaned reviews and their embeddings produced by the notebook."""
    try:
        df = pd.read_csv(CSV_PATH)
        df['avis_cor_en'] = df['avis_cor_en'].fillna("")
        df['avis'] = df['avis'].fillna("")
        with open(EMBEDDINGS_PATH, "rb") as f:
            embeddings = pickle.load(f)
        return df, embeddings
    except Exception as e:
        st.error(f"⚠️ Error loading files. Run the notebook first to create {CSV_PATH.name} "
                 f"and {EMBEDDINGS_PATH.name} in data/processed/. ({e})")
        return pd.DataFrame(), None


def show_dashboard(df):
    """Page 0: global statistics on the dataset."""
    st.title("📊 Global Reviews Dashboard")
    st.write("Overview of the insurance customer reviews dataset.")
    if df.empty:
        return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", f"{len(df):,}")
    col2.metric("Total Insurers", df['assureur'].nunique())
    col3.metric("Average Rating", f"{df['note'].mean():.2f} / 5")
    col4.metric("1-Star Reviews", f"{(len(df[df['note'] == 1]) / len(df) * 100):.1f} %")
    st.divider()
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("⭐ Rating Distribution")
        st.bar_chart(df['note'].value_counts().sort_index(), color="#FF4B4B")
    with col_chart2:
        st.subheader("🏢 Top 10 Insurers (by volume)")
        st.bar_chart(df['assureur'].value_counts().head(10), color="#0068C9")


def show_keywords(user_input):
    """Naive keyword display: the longer words of the review (a heuristic, not a model explanation)."""
    st.subheader("💡 Keywords")
    st.info("Simple heuristic: words of 5 letters or more. For model explanations, see the SHAP plot in the notebook.")
    strong_words = [w for w in user_input.split() if len(w) >= 5]
    st.markdown(f"**Keywords:** `{', '.join(strong_words[:6])}...`")


def retrieve_english_reviews(question, df, embeddings, embedder, k=5):
    """Semantic retrieval: indices of the k most similar reviews that are proper English."""
    from sklearn.metrics.pairwise import cosine_similarity

    query_vector = embedder.encode([question])
    similarities = cosine_similarity(query_vector, embeddings)[0]
    top_indices = []
    for idx in similarities.argsort()[::-1]:
        if is_valid_english(str(df.iloc[idx]['avis_cor_en'])):
            top_indices.append(idx)
        if len(top_indices) == k:
            break
    return top_indices


def show_sources(df, indices):
    st.divider()
    st.subheader("📚 Source Context (Top 3 English Reviews)")
    for idx in indices[:3]:
        row = df.iloc[idx]
        with st.chat_message("user"):
            st.write(f"**{row['assureur']}** ({row['note']}⭐) : {row['avis_cor_en']}")


def show_recent_reviews(df_insurer):
    with st.expander("Show recent original reviews (English)"):
        shown = 0
        for _, row in df_insurer.iterrows():
            if is_valid_english(row['avis_cor_en']):
                st.markdown(f"- *{row['avis_cor_en']}* ({row['note']}⭐)")
                shown += 1
            if shown == 5:
                break
