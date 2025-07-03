import pandas as pd
import gradio as gr
from eda_module import basic_eda
from summary_generator import generate_summary

def prettify_eda_summary(eda):
    text = f"The dataset contains {eda['shape'][0]} rows and {eda['shape'][1]} columns.\n\n"

    text += "Column Types:\n"
    for col, dtype in eda['data_types'].items():
        text += f"- {col}: {dtype}\n"

    text += "\nMissing Values:\n"
    for col, nulls in eda['null_counts'].items():
        text += f"- {col}: {nulls} missing values\n"

    text += "\nSummary Statistics (Mean, Std):\n"
    for col, stats in eda['describe'].items():
        if isinstance(stats, dict) and 'mean' in stats and 'std' in stats:
            text += f"- {col}: mean = {stats['mean']:.2f}, std = {stats['std']:.2f}\n"

    return text

def analyze_and_generate(file):
    df = pd.read_csv(file.name)
    eda_results = basic_eda(df)
    raw_summary = prettify_eda_summary(eda_results)
    ai_summary = generate_summary(raw_summary)
    return raw_summary, ai_summary

# Gradio interface with raw and AI summary
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 GenAI EDA Assistant\nUpload a CSV file to get raw and AI-generated summaries.")

    file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
    raw_box = gr.Textbox(label="📋 Raw EDA Summary", lines=20)
    ai_box = gr.Textbox(label="🤖 AI-Generated Summary", lines=10)
    btn = gr.Button("Analyze")

    btn.click(fn=analyze_and_generate, inputs=file_input, outputs=[raw_box, ai_box])

demo.launch()
