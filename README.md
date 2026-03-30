#  LLM Evaluation & Agentic Reasoning Optimization System

##  Overview
This project builds a complete LLM evaluation and optimization pipeline to improve multi-step reasoning in small language models. It benchmarks performance on the GSM8K dataset and demonstrates measurable improvements using structured prompting and parameter-efficient fine-tuning.

---

##  Features
- Multi-phase evaluation: Baseline, SCoT prompting, and LoRA fine-tuned models  
- Automated answer extraction, validation, and error classification  
- Agent-style step-by-step reasoning with structured prompts  
- LoRA fine-tuning using PEFT for efficient model optimization  
- Interactive Streamlit dashboard for visualization and testing  

---

## Tech Stack
- Python, PyTorch  
- Hugging Face Transformers, PEFT (LoRA)  
- Streamlit, Pandas, Matplotlib  
- GSM8K Dataset  

---

## Project Structure
genai/
├── backend/
├── training/
├── frontend/
├── main.py
├── requirements.txt

---

## Getting Started

### Install dependencies
pip install -r requirements.txt

### Run evaluation
python main.py

### Run dashboard
streamlit run frontend.py

### Train model
python training/training_lora.py

---

## Results
- Baseline: ~80%  
- SCoT Prompting: ~82%  
- LoRA Fine-tuned: ~85%  

---

## 💡 Key Highlights
- Demonstrates real-world AI system design  
- Combines evaluation, optimization, and training  
- Shows measurable improvement in reasoning capability  


