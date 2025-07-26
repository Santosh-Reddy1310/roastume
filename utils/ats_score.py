def compute_ats_score(text):
    words = text.split()
    base_score = min(len(words) // 10, 100)
    boost = sum([text.lower().count(k) for k in ["python", "data", "project", "machine", "deep", "tensorflow"]])
    return min(base_score + boost * 2, 100)
