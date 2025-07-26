import streamlit as st
import fitz
from dotenv import load_dotenv
import google.generativeai as genai
import os
import time

from utils.parse_resume import extract_text_from_pdf
from utils.ats_score import compute_ats_score
from utils.classify_resume import predict_resume_category
from prompts.tones import TONE_PROMPTS
from utils.resume_rewriter import generate_rewrite_prompt

# Load Gemini key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash")

role_emojis = {
    "HR": "🤝",
    "Data Science": "📊",
    "Advocate": "⚖️",
    "Arts": "🎨",
    "Web Designing": "💻",
    "Mechanical Engineer": "⚙️",
    "Sales": "💰",
    "Health and fitness": "💪",
    "Civil Engineer": "🏗️",
    "Java Developer": "☕",
    "Business Analyst": "📈",
    "SAP Developer": "🖥️",
    "Automation Testing": "🧪",
    "Electrical Engineering": "⚡",
    "Operations Manager": "🛠️",
    "Python Developer": "🐍",
    "DevOps Engineer": "☁️",
    "Network Security Engineer": "🔒",
    "PMO": "🗓️",
    "Database": "🗄️"}
st.set_page_config("Roastume", layout="wide", page_icon="🔥")

# UI
st.markdown("""
<style>
.title { font-size: 3rem; font-weight: 800; color: #facc15; }
.subtitle { font-size: 1.2rem; color: #94a3b8; margin-bottom: 2rem; }
.roast-card { background-color: #1f2937; padding: 24px; border-radius: 16px; color: #f8fafc; }
</style>
<div class="title">🔥 Roastume</div>
<div class="subtitle">Where your resume gets judged — and rewritten — by AI personalities.</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader("📄 Upload Resume", type=["pdf", "txt"])
with col2:
    tone = st.selectbox("🎭 Choose AI Mood", list(TONE_PROMPTS.keys()))

if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file) if uploaded_file.name.endswith("pdf") else uploaded_file.read().decode()

    st.markdown("## 📊 AI + ML Evaluation")
    score = compute_ats_score(resume_text)
    category = predict_resume_category(resume_text)

    col3, col4 = st.columns(2)
    col3.metric("ATS Score", f"{score}/100")
    col3.progress(score)
    if category == "Unknown":
        st.warning("🤔 Couldn't confidently predict resume category.")
    else:
        emoji = role_emojis.get(category, "🧠")
        st.success(f"{emoji} ML predicts: **{category}** resume")

    st.markdown("## 🤖 AI’s Feedback")
    with st.spinner("Roasting your resume..."):
        prompt = TONE_PROMPTS[tone].replace("{text}", resume_text)
        response = None  # initialize

        try:
            response = model.generate_content(prompt)
        except Exception as e:
            if "429" in str(e):
                st.warning("⚠️ Gemini API rate limit hit. Please wait and try again.")
            else:
                st.error("💥 Gemini API call failed.")
                st.exception(e)

        if response:
            st.markdown(f"""
            <div class="roast-card">
            <h4>🗣️ Feedback – <span style='color:#4ade80'>{tone}</span></h4>
            <p style='line-height:1.6'>{response.text.replace('\n', '<br>')}</p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("## ✨ Download AI-Improved Resume")
    rewrite_prompt = generate_rewrite_prompt(resume_text, tone)
    with st.spinner("Rewriting your resume..."):
        improved_resume = model.generate_content(rewrite_prompt).text
        st.text_area("Preview", improved_resume, height=400)
        st.download_button("📥 Download Updated Resume (.txt)", improved_resume, file_name="Updated_Resume.txt")
