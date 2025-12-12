# app.py
import re
import io
import pdfplumber
import pandas as pd
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
import nltk
from typing import List
from collections import Counter

# Ensure NLTK punkt is available
nltk.download('punkt', quiet=True)

st.set_page_config(page_title="PDF Keyword Extractor + WordCloud", layout="wide")

st.title("PDF Keyword Paragraph Extractor & WordCloud")
st.markdown(
    """
    Upload a PDF or use an existing path. Enter **multiple keywords** (comma-separated).
    The app will extract paragraphs that contain **any** of the keywords (case-insensitive),
    display them, show a word cloud of the extracted text, and let you download results.
    """
)

# --- Sidebar: options ---
st.sidebar.header("Options")
use_sample = st.sidebar.checkbox("Use internal sample PDF path (/mnt/data/OG 24-25.pdf)", value=True)
uploaded_file = st.sidebar.file_uploader("Or upload PDF file", type=["pdf"])
min_paragraph_len = st.sidebar.number_input("Minimum paragraph characters to include", value=30, step=5)
match_whole_words = st.sidebar.checkbox("Match whole words only (no substrings)", value=False)
show_sentence_view = st.sidebar.checkbox("Show sentence-level matches instead of paragraphs", value=False)

# Keyword input
keywords_input = st.text_input("Enter keywords (comma-separated)", value="sustainability, environment, green")
keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
if not keywords:
    st.warning("Please enter at least one keyword (comma-separated).")

# Helper: read PDF -> text
@st.cache_data(show_spinner=False)
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> List[str]:
    """
    Returns list of page texts
    """
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return pages

@st.cache_data(show_spinner=False)
def extract_text_from_pdf_path(pdf_path: str) -> List[str]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return pages

def split_into_paragraphs(page_text: str) -> List[str]:
    """Split by blank line(s) or two+ newlines, fallback to sentence grouping if needed."""
    if not page_text:
        return []
    # Normalize newlines
    t = page_text.replace('\r', '\n')
    # Split on 2 or more newlines
    paras = [p.strip() for p in re.split(r'\n\s*\n+', t) if p.strip()]
    # If no paragraph separators, fallback to chunking by sentences (every 3 sentences)
    if not paras:
        sents = sent_tokenize(t)
        paras = []
        chunk_size = 3
        for i in range(0, len(sents), chunk_size):
            paras.append(" ".join(sents[i:i+chunk_size]))
    return paras

def match_keywords_in_text(text: str, keywords: List[str], whole_words=False) -> bool:
    if not text or not keywords:
        return False
    flags = re.IGNORECASE
    for kw in keywords:
        escaped = re.escape(kw)
        if whole_words:
            pattern = r'\b' + escaped + r'\b'
        else:
            pattern = escaped
        if re.search(pattern, text, flags):
            return True
    return False

def extract_paragraphs_from_pages(pages: List[str], keywords: List[str], min_len=30, whole_words=False, sentence_view=False):
    """
    Returns list of dicts: {'page': int, 'paragraph': str}
    If sentence_view True, extract sentences instead of paragraphs
    """
    results = []
    for i, page_text in enumerate(pages, start=1):
        if sentence_view:
            units = sent_tokenize(page_text) if page_text else []
        else:
            units = split_into_paragraphs(page_text)
        for u in units:
            if len(u) < min_len:
                continue
            if match_keywords_in_text(u, keywords, whole_words=whole_words):
                results.append({"page": i, "text": u})
    return results

# Load PDF either from path or uploaded bytes
pages_text = []
if use_sample and (uploaded_file is None):
    sample_path = "/mnt/data/OG 24-25.pdf"
    try:
        pages_text = extract_text_from_pdf_path(sample_path)
        st.success(f"Loaded sample PDF from `{sample_path}` ({len(pages_text)} pages)")
    except Exception as e:
        st.error(f"Could not load sample PDF from path: {e}")
        if uploaded_file is None:
            st.info("Please upload a PDF in the sidebar.")
else:
    if uploaded_file:
        try:
            pdf_bytes = uploaded_file.read()
            pages_text = extract_text_from_pdf_bytes(pdf_bytes)
            st.success(f"Uploaded PDF '{uploaded_file.name}' ({len(pages_text)} pages)")
        except Exception as e:
            st.error(f"Failed to read uploaded PDF: {e}")

# Button to run extraction
run_button = st.button("Extract paragraphs containing keywords")

if run_button:
    if not pages_text:
        st.error("No PDF loaded. Use the sample path or upload a PDF.")
    elif not keywords:
        st.error("No keywords provided. Enter comma-separated keywords.")
    else:
        with st.spinner("Extracting..."):
            matches = extract_paragraphs_from_pages(
                pages_text,
                keywords,
                min_len=min_paragraph_len,
                whole_words=match_whole_words,
                sentence_view=show_sentence_view
            )

        if not matches:
            st.warning("No paragraphs/sentences matched the keywords.")
        else:
            st.success(f"Found {len(matches)} matching paragraphs/sentences.")

            # DataFrame of results
            df_matches = pd.DataFrame(matches)
            # Show with expanders by page
            st.subheader("Matches (expand to view)")
            pages_grouped = df_matches.groupby("page")
            cols = st.columns([1, 3])
            # Show a simple table
            st.dataframe(df_matches[["page", "text"]].rename(columns={"text":"paragraph"}), height=300)

            # Expanders per page
            for page_num, group in pages_grouped:
                with st.expander(f"Page {page_num} — {len(group)} matches", expanded=False):
                    for idx, row in group.reset_index(drop=True).iterrows():
                        st.markdown(f"**Match {idx+1}**")
                        st.write(row['text'])
                        st.mark_markdown = lambda *a, **k: None  # placeholder if needed

            # WordCloud from all matched text
            st.subheader("Word Cloud from matched paragraphs")
            text_joined = " ".join(df_matches['text'].tolist())

            if text_joined.strip():
                # Simple cleaning function for wordcloud
                wc = WordCloud(width=900, height=450, background_color="white", collocations=False).generate(text_joined)
                fig, ax = plt.subplots(figsize=(12,6))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("No text to generate word cloud.")

            # Frequency counts (top 30)
            st.subheader("Top terms (simple whitespace tokenization)")
            toks = re.findall(r"\b\w+\b", text_joined.lower())
            # remove common short tokens optionally
            toks = [t for t in toks if len(t) > 2]
            top = Counter(toks).most_common(30)
            top_df = pd.DataFrame(top, columns=["term","count"])
            st.table(top_df)

            # Download CSV of matches
            csv_bytes = df_matches.to_csv(index=False).encode('utf-8')
            st.download_button("Download matches as CSV", data=csv_bytes, file_name="matches.csv", mime="text/csv")

            # Offer the user the combined text for copy/paste
            st.subheader("Combined matched text (copy/paste)")
            st.text_area("Combined text", value=text_joined, height=200)

# Footer / instructions
st.markdown("---")
st.markdown(
    """
    **Notes & tips**
    - Use commas to separate keywords (e.g., `sustainability, environment, carbon`).
    - Toggle "Match whole words only" to avoid substring hits (e.g., `art` matching `part`).
    - If PDF is long, extraction may take a few seconds.
    """
)
