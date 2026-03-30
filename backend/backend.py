import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


# =========================
# MODEL LOADER
# =========================
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_base_model():
    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.eval()

    print("Model loaded!")
    return model, tokenizer, device


# =========================
# INFERENCE (uses Phi-3 chat template)
# =========================
def ask_model(prompt, model, tokenizer, device):
    """Send a prompt to the model using the proper Phi-3 chat template."""
    # Format as a chat message — Phi-3-instruct expects this format
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

    # Only decode the NEW tokens (skip the prompt tokens)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# =========================
# EVALUATOR (improved extraction)
# =========================
def extract_number(text):
    """Extract the answer number from text using priority-based matching."""
    if text is None:
        return None

    # Remove commas from numbers: 15,000 -> 15000
    text = re.sub(r"(?<=\d),(?=\d)", "", text)

    # PRIORITY 1: Look for GSM8K format  #### <number>
    match = re.search(r"####\s*(-?\d+\.?\d*)", text)
    if match:
        return match.group(1)

    # PRIORITY 2: Look for "answer is <number>" or "answer: <number>"
    match = re.search(r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*\$?\s*(-?\d+\.?\d*)", text, re.IGNORECASE)
    if match:
        return match.group(1)

    # PRIORITY 3: Look for "= <number>" at the end of a line (last calculation result)
    matches = re.findall(r"=\s*\$?\s*(-?\d+\.?\d*)\s*$", text, re.MULTILINE)
    if matches:
        return matches[-1]

    # FALLBACK: Last number in the text
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else None


def evaluate(pred, true):
    """Compare predicted and true answers using numeric comparison."""
    pred_num = extract_number(pred)
    true_num = extract_number(true)

    # Numeric comparison to handle "26.0" == "26" and similar cases
    try:
        is_correct = (
            pred_num is not None
            and true_num is not None
            and abs(float(pred_num) - float(true_num)) < 1e-5
        )
    except (ValueError, TypeError):
        is_correct = pred_num == true_num

    return is_correct, pred_num, true_num


# =========================
# ERROR ANALYSIS
# =========================
def classify_error(pred, true):
    if pred is None:
        return "no_answer"
    if pred != true:
        return "wrong_calculation"
    return "correct"


# =========================
# PHASE 1 (BASELINE)
# =========================
def run_phase1(model, tokenizer, device, dataset, limit=10):
    correct = 0
    results = []

    print("\nRunning Phase 1 (Baseline)...")

    for i in range(limit):
        item = dataset[i]

        question = item["question"]
        true_answer = item["answer"]

        prompt = (
            "Solve this math problem. Show your work, then give the final numeric answer.\n\n"
            f"Question: {question}\n\n"
            "After solving, write your final answer as: #### <number>"
        )

        output = ask_model(prompt, model, tokenizer, device)

        is_correct, pred, true = evaluate(output, true_answer)

        error_type = classify_error(pred, true) if not is_correct else "correct"

        results.append({
            "question": question,
            "prediction": pred,
            "actual": true,
            "correct": is_correct,
            "error_type": error_type
        })

        if is_correct:
            correct += 1

        print(f"[Phase1-{i+1}/{limit}] Pred={pred} True={true} {'✓' if is_correct else '✗'}")

    accuracy = correct / limit
    print(f"Phase 1 Accuracy: {accuracy:.1%} ({correct}/{limit})")
    return accuracy, results


# =========================
# PHASE 2 (SCoT - Structured Chain of Thought)
# =========================
def run_phase2(model, tokenizer, device, dataset, limit=10):
    correct = 0
    results = []

    print("\nRunning Phase 2 (SCoT - Structured Chain of Thought)...")

    for i in range(limit):
        item = dataset[i]

        question = item["question"]
        true_answer = item["answer"]

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

        is_correct, pred, true = evaluate(output, true_answer)

        error_type = classify_error(pred, true) if not is_correct else "correct"

        results.append({
            "question": question,
            "prediction": pred,
            "actual": true,
            "correct": is_correct,
            "error_type": error_type
        })

        if is_correct:
            correct += 1

        print(f"[Phase2-{i+1}/{limit}] Pred={pred} True={true} {'✓' if is_correct else '✗'}")

    accuracy = correct / limit
    print(f"Phase 2 Accuracy: {accuracy:.1%} ({correct}/{limit})")
    return accuracy, results


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    model, tokenizer, device = load_base_model()

    dataset = load_dataset("gsm8k", "main")["test"]

    # Run both phases
    acc1, res1 = run_phase1(model, tokenizer, device, dataset, limit=10)
    acc2, res2 = run_phase2(model, tokenizer, device, dataset, limit=10)

    print("\n====================")
    print(f"Phase 1 Accuracy: {acc1:.1%}")
    print(f"Phase 2 Accuracy: {acc2:.1%}")