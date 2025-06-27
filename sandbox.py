import torch
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
)

# 1. Load Model and Tokenizer
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 2. Prepare the Dataset
prefix = "Analyze the TCP scan data and provide a risk assessment:\n\n"

source_text_data = """Time Window: very-short
SYN Rate: very low
SYN Percentage: minimal
Port Spread: one
ACK Response Rate: no response
Source Diversity: centralized"""

target_text_data = """Scan Intensity: none<extra_id_0>Risk Assessment: none<extra_id_0>Scan Archetype: classic<extra_id_0>Suggested Action: Monitor only<extra_id_0>Explanation: Low scanning activity; single/few sources; minimal ACK responses; over very-short timeframe; = classic scan pattern"""

data = {
    "train": {
        "source_text": [prefix + source_text_data],
        "target_text": [target_text_data],
    }
}

train_dataset = Dataset.from_dict(data["train"])

# 3. Tokenize the Data
def tokenize_function(examples):
    inputs = tokenizer(
        examples["source_text"],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    labels = tokenizer(
        examples["target_text"],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).input_ids
    labels[labels == tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return inputs

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)

# 4. Set Up Training Arguments
training_args = TrainingArguments(
    output_dir="./results_complex_final",
    num_train_epochs=200,
    per_device_train_batch_size=1,
    learning_rate=5e-5,
    logging_dir="./logs_complex_final",
    logging_steps=20,
    save_steps=100,
    evaluation_strategy="no",
    remove_unused_columns=True,
)

# 5. Fine-Tune the Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
)

trainer.train()

# 6. Inference
input_text_for_inference = prefix + source_text_data
input_ids = tokenizer.encode(input_text_for_inference, return_tensors="pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_ids = input_ids.to(device)

output_ids = model.generate(input_ids, max_length=256, num_beams=4, early_stopping=True)

# Decode while keeping special tokens
predicted_text_raw = tokenizer.decode(output_ids[0], skip_special_tokens=False)

# --- THE FIX: A more robust, multi-step cleanup process ---
# First, do all replacements
predicted_text_formatted = predicted_text_raw.replace('<extra_id_0>', '\n')
predicted_text_formatted = predicted_text_formatted.replace(tokenizer.pad_token, '').replace(tokenizer.eos_token, '')

# Now, clean up whitespace on each line individually
lines = predicted_text_formatted.split('\n')
stripped_lines = [line.strip() for line in lines]
predicted_text_formatted = '\n'.join(stripped_lines)


print("--- Source Input ---")
print(source_text_data)
print("\n--- Predicted Output (Formatted) ---")
print(predicted_text_formatted)