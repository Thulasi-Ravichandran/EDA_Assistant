from transformers import pipeline

summarizer = pipeline("text2text-generation", model="google/flan-t5-base")

def generate_summary(text_chunk):
    prompt = (
        "Summarize this dataset EDA:\n"
        f"{text_chunk}\n\n"
        "Return 3 bullet points about dataset size, missing values, and numeric trends."
    )
    result = summarizer(prompt, max_length=200, do_sample=False)[0]['generated_text']
    return result.strip()
