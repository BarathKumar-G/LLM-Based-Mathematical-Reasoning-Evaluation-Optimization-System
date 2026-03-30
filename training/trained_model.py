from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
import torch
import re
import os

# -------------------------------
# CONFIG
# -------------------------------
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"

# Resolve adapter path relative to THIS file's location
ADAPTER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetuned_model")

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------
# LOAD MODEL
# -------------------------------
def load_finetuned_model():
    print(f"Loading base model: {BASE_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print(f"Loading LoRA adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    # Verify LoRA is loaded
    print(f"Model type: {type(model).__name__}")
    print("Fine-tuned model ready!")

    return model, tokenizer, device


# -------------------------------
# INFERENCE (uses Phi-3 chat template)
# -------------------------------
def ask_model(prompt, model, tokenizer, device):
    """Send a prompt to the model using the proper Phi-3 chat template."""
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Only decode the NEW tokens (skip the prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# -------------------------------
# EXTRACT NUMBER (same priority logic as backend)
# -------------------------------
def extract_number(text):
    """Extract the answer number from text using priority-based matching."""
    if text is None:
        return None

    text = re.sub(r"(?<=\d),(?=\d)", "", text)

    # PRIORITY 1: GSM8K #### format
    match = re.search(r"####\s*(-?\d+\.?\d*)", text)
    if match:
        return match.group(1)

    # PRIORITY 2: "answer is/:/=" pattern
    match = re.search(r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*\$?\s*(-?\d+\.?\d*)", text, re.IGNORECASE)
    if match:
        return match.group(1)

    # PRIORITY 3: Last "= <number>" on its own line
    matches = re.findall(r"=\s*\$?\s*(-?\d+\.?\d*)\s*$", text, re.MULTILINE)
    if matches:
        return matches[-1]

    # FALLBACK: Last number
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else None


# -------------------------------
# ERROR TYPE
# -------------------------------
def classify_error(pred, true):
    if pred is None:
        return "no_answer"
    if pred != true:
        return "wrong_calculation"
    return "correct"


# -------------------------------
# PHASE 3 (FINETUNED + SCoT prompt)
# -------------------------------
def run_phase3(model, tokenizer, dataset, limit=5):
    """Phase 3: Fine-tuned model with SCoT prompt — same prompt as Phase 2."""
    correct = 0
    results = []

    print("\nRunning Phase 3 (Fine-tuned + SCoT)...")

    for i in range(limit):
        item = dataset[i]

        question = item["question"]
        true_answer = extract_number(item["answer"])

        # IMPORTANT: Use the SAME SCoT prompt as Phase 2
        # This isolates the effect of fine-tuning vs prompt engineering
        prompt = (
            "You are a highly accurate mathematician. Solve the following math problem.\n\n"
            "Instructions:\n"
            "1. Read the problem carefully and identify what is being asked.\n"
            "2. Break the problem into clear steps.\n"
            "3. Perform each calculation carefully, showing your work.\n"
            "4. Double-check your arithmetic.\n"
            "5. Write your final numeric answer in this EXACT format: #### <number>\n\n"
            "IMPORTANT: You MUST end your response with #### followed by ONLY the final number.\n"
            "Do NOT write anything after the #### answer line.\n\n"
            f"Question: {question}"
        )

        output = ask_model(prompt, model, tokenizer, device)
        pred_answer = extract_number(output)

        # Numeric comparison
        try:
            is_correct = (
                pred_answer is not None
                and true_answer is not None
                and abs(float(pred_answer) - float(true_answer)) < 1e-5
            )
        except (ValueError, TypeError):
            is_correct = pred_answer == true_answer

        error_type = classify_error(pred_answer, true_answer) if not is_correct else "correct"

        results.append({
            "question": question,
            "prediction": pred_answer,
            "actual": true_answer,
            "correct": is_correct,
            "error_type": error_type
        })

        if is_correct:
            correct += 1

        print(f"[Phase3-{i+1}/{limit}] Pred={pred_answer} True={true_answer} {'✓' if is_correct else '✗'}")

    accuracy = correct / limit
    print(f"Phase 3 Accuracy: {accuracy:.1%} ({correct}/{limit})")

    return accuracy, results