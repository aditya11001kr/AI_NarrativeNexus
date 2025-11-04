import os
import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai

# Functional modular imports
from text_processing import preprocess_series, nlp_preprocess
from topic_utils import display_topics

# Configure Gemini API
API_KEY = "AIzaSyDN831ZT32JlIyg7P-R5CpcpQBBAd62xq4"
genai.configure(api_key=API_KEY)
MODEL = "gemini-2.5-flash"

# Safely load models with error handling
def safe_load_model(path):
    try:
        model = joblib.load(path)
        return model, None
    except Exception as e:
        return None, str(e)

lda, lda_error = safe_load_model(r"D:\Springboard\15-9-2025\topic_modeling\models\lda_model.pkl")
count_vectorizer, cv_error = safe_load_model(r"D:\Springboard\15-9-2025\topic_modeling\models\lda_vectorizer.pkl")
nmf, nmf_error = safe_load_model(r"D:\Springboard\15-9-2025\topic_modeling\models\nmf_model.pkl")
tfidf_vectorizer, tfidf_error = safe_load_model(r"D:\Springboard\15-9-2025\topic_modeling\models\nmf_vectorizer.pkl")
pipeline, pipeline_error = safe_load_model(r"D:\Springboard\15-9-2025\topic_modeling\models\topic_classifier.pkl")
rf_sentiment, rf_error = safe_load_model(r"D:\Springboard\Sentimental_Analysis\random_forest_model.pkl")
sent_vectorizer, sv_error = safe_load_model(r"D:\Springboard\Sentimental_Analysis\tfidf_vectorizer.pkl")

# Summarization function with error handling
def summarize(text: str, max_tokens: int = 200) -> str:
    prompt = (
        "Please read the following text carefully and generate a clear, concise summary of about 5 to 6 sentences. "
        "Focus on the main points, key events, outcomes, and important details without unnecessary repetition or overly technical jargon.\n\n"
        "Make the summary easily understandable for a general audience, maintaining accuracy and coherence. "
        "If the text contains any sentiment or notable topics, highlight them briefly. "
        "Provide the summary in a well-structured paragraph form.\n\n"
        f"{text}"
    )
    try:
        response = genai.GenerativeModel(MODEL).generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        if "429" in str(e):
            return "[Quota exceeded: Please try again later or upgrade your API plan.]"
        return f"[Error generating summary: {e}]"

# Streamlit config
st.set_page_config(page_title="AI Narrative Nexus", layout="wide")

# Sidebar
st.sidebar.title("AI Narrative Nexus")
st.sidebar.header("Dynamic Text Analysis Platform")

pages = ["Home", "Data Visualization", "Evaluation & Analysis", "Live Demo", "Text Summarization", "About"]
page = st.sidebar.radio("Go to", pages)

st.sidebar.markdown("Model (placeholder)")
model_selected = st.sidebar.selectbox("Select model", ["SentimentNet"])

# HOME
if page == "Home":
    st.markdown("## 🧭 AI Narrative Nexus - Dynamic Text Analysis Platform")
    st.write("Upload text, run topic modeling, sentiment analysis, summarization, and explore interactive visualizations.")
    st.markdown("""
### What this platform does
- Topic Modeling: LDA & NMF discover latent themes.
- Sentiment Analysis: document & sentence-level sentiment.
- Text Summarization: extractive & abstractive options.
- Interactive Visualizations: distributions, word clouds, t-SNE/UMAP, sentiment timelines.

### Quick Start
Go to Live Demo ➔ paste/upload/URL ➔ click Analyze ➔ view results in Data Visualization & Evaluation & Analysis.

### Why it helps
- Quick thematic overviews.
- Sentiment hotspots and trends.
- Executive summaries for decision-makers.
- Exportable reports and dashboards.
""")

# DATA VISUALIZATION
elif page == "Data Visualization":
    st.title('📊 Data Visualization')
    if os.path.exists("startup_cleaned.csv"):
        df = pd.read_csv("startup_cleaned.csv")
        st.dataframe(df.head())
    else:
        st.warning("Visualization demo: 'startup_cleaned.csv' not found.")

# EVALUATION & ANALYSIS
elif page == "Evaluation & Analysis":
    st.title("Evaluation & Analysis")
    base_path = r"D:\Springboard\15-9-2025\topic_modeling"
    images = ["confusion_matrix.png", "lda_confusion_matrix.png", "nmf_confusion_matrix.png"]
    found_any = False
    for img in images:
        img_path = os.path.join(base_path, img)
        if os.path.exists(img_path):
            st.image(img_path, caption=os.path.splitext(img)[0])
            found_any = True
        else:
            st.warning(f"Image file not found: {img_path}")
    if not found_any:
        st.info("No evaluation images found to display. Please check your files or add evaluation metrics here.")

# LIVE DEMO
elif page == "Live Demo":
    st.title("Live Demo: Upload or Paste Text")

    uploaded_file = st.file_uploader("Upload (.txt, .csv, .docx, .pdf)", type=["txt", "csv", "docx", "pdf"])
    text_input = st.text_area("Or paste text below:")

    content = ""
    if uploaded_file:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext == ".txt":
            content = uploaded_file.read().decode("utf-8")
        elif ext == ".csv":
            try:
                df = pd.read_csv(uploaded_file)
                content = df.to_string()
            except:
                st.error("Cannot process CSV.")
        elif ext == ".docx":
            import docx
            doc = docx.Document(uploaded_file)
            content = "\n".join([p.text for p in doc.paragraphs])
        elif ext == ".pdf":
            import pdfplumber
            content = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for p in pdf.pages:
                    page_text = p.extract_text()
                    if page_text:
                        content += page_text + "\n"
    elif text_input.strip():
        content = text_input.strip()

    if content and st.button("Run Full Analysis"):
        with st.spinner("Analyzing..."):
            if any([lda_error, cv_error, nmf_error, tfidf_error, pipeline_error, rf_error, sv_error]):
                st.error("One or more models failed to load. Please check logs.")
                st.write(f"LDA load error: {lda_error}")
                st.write(f"Count_vectorizer load error: {cv_error}")
                st.write(f"NMF load error: {nmf_error}")
                st.write(f"TF-IDF vectorizer load error: {tfidf_error}")
                st.write(f"Pipeline load error: {pipeline_error}")
                st.write(f"Random Forest sentiment model load error: {rf_error}")
                st.write(f"Sentiment vectorizer load error: {sv_error}")
            else:
                cleaned = preprocess_series([content])[0]
                st.markdown("**Cleaned Text**")
                st.write(cleaned[:800] + "..." if len(cleaned) > 800 else cleaned)

                X_counts = count_vectorizer.transform([cleaned])
                lda_topics = lda.transform(X_counts)
                top_lda = int(lda_topics[0].argmax())
                st.markdown("**LDA Topic (highest prob):**")
                st.write(f"Topic #{top_lda}")

                X_tfidf = tfidf_vectorizer.transform([cleaned])
                nmf_topics = nmf.transform(X_tfidf)
                top_nmf = int(nmf_topics[0].argmax())
                st.markdown("**NMF Topic (highest prob):**")
                st.write(f"Topic #{top_nmf}")

                X_vect = sent_vectorizer.transform([cleaned])
                sent_pred = rf_sentiment.predict(X_vect)[0]
                st.markdown("**Sentiment:**")
                st.write("Positive" if int(sent_pred) == 1 else "Negative")

                st.markdown("**Summary (demo):**")
                st.write(summarize(cleaned))

    elif not uploaded_file and not text_input.strip():
        st.info("Upload a file or paste text to use the pipeline.")

# TEXT SUMMARIZATION
elif page == "Text Summarization":
    st.title("Text Summarization")
    st.write("Enter or upload text to generate a summary.")

    text_input = st.text_area("Paste text to summarize here", height=200)
    uploaded_file = st.file_uploader("Or upload a .txt or .pdf file", type=["txt", "pdf"], key="sum_file")

    content = ""
    if uploaded_file:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext == ".txt":
            content = uploaded_file.read().decode("utf-8")
        elif ext == ".pdf":
            import pdfplumber
            content = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        content += page_text + "\n"
    elif text_input.strip():
        content = text_input.strip()

    if content:
        if st.button("Summarize"):
            summary_text = summarize(content)
            st.markdown("**Summary:**")
            st.write(summary_text)

# ABOUT
elif page == "About":
    st.title("About AI Narrative Nexus")
    st.write("""
    **AI Narrative Nexus** is an end-to-end platform for text analysis:
    - Text input and cleaning
    - Topic modeling (LDA, NMF)
    - Sentiment analysis (ML models)
    - Text summarization
    - Rich interactive visualizations
    """)
