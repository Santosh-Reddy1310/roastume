def generate_rewrite_prompt(resume_text, tone):
    return f"""
You are a professional resume coach. Rewrite the following resume based on your AI feedback in the tone of '{tone}'.

Make the resume:
- ATS-optimized (keyword-rich, recruiter-friendly)
- Clearly structured with bullet points
- Quantified (include impact, numbers if mentioned)
- In professional formatting with section headers

Here is the original resume:
============================
{resume_text}
============================

Please rewrite only the resume. Do not include commentary or analysis.
"""
