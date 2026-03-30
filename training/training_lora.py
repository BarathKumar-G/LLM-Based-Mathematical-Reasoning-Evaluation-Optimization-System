from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

print("Loading model...")

# -------------------------------
# QLoRA CONFIG 
# -------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# -------------------------------
# LOAD TOKENIZER
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# -------------------------------
# LOAD MODEL (QLoRA)
# -------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config, 
    device_map="auto"
)

# -------------------------------
# LoRA Config
# -------------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# -------------------------------
# MEMORY OPTIMIZATION (IMPORTANT)
# -------------------------------
model.gradient_checkpointing_enable()
model.config.use_cache = False

print("QLoRA model ready!")

# -------------------------------
# Load Dataset
# -------------------------------
dataset = load_dataset("gsm8k", "main")
dataset = dataset["train"].select(range(2000))

# -------------------------------
# Format Data 
# -------------------------------
def format_data(example):
    return {
        "text": f"""You are a careful mathematician.

Solve step by step.

Question:
{example['question']}

Final Answer:
{example['answer']}
"""
    }

dataset = dataset.map(format_data)

# -------------------------------
# Tokenization
# -------------------------------
def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(tokenize, batched=True)

# -------------------------------
# Data Collator
# -------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# -------------------------------
# Training args (QLoRA optimized)
# -------------------------------
training_args = TrainingArguments(
    output_dir="./finetuned_model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    logging_steps=20,
    save_steps=200,
    fp16=True,
    report_to="none"
)

# -------------------------------
# Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

print("Training started...")
trainer.train()

# -------------------------------
# Save model
# -------------------------------
model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")

print("Model saved!")