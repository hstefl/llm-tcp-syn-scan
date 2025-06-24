from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd

# === Load and format data ===
df = pd.read_csv("in/dataset.csv")

def format_example(row):
    prompt = f"""Evaluate TCP SYN scan risk:
Time Window: {row['Time Window']}
SYN Rate: {row['SYN Rate']}
SYN Percentage: {row['SYN Percentage']}
Port Spread: {row['Port Spread']}
ACK Response Rate: {row['ACK Response Rate']}
Source Diversity: {row['Source Diversity']}
Scan Intensity: {row['Scan Intensity']}"""

    response = f"""Risk Assessment: {row['Risk Assessment']}
Scan Archetype: {row['Scan Archetype']}
Suggested Action: {row['Suggested Action']}
Explanation: {row['Explanation']}"""
    return {"prompt": prompt, "response": response}

dataset = Dataset.from_list([format_example(row) for _, row in df.iterrows()])

# === Tokenization ===
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize(example):
    full_input = example["prompt"] + "\n" + example["response"]
    return tokenizer(full_input, padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize)

# === Load model with LoRA ===
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05, inference_mode=False)
model = get_peft_model(model, peft_config)

# === Training setup ===
training_args = TrainingArguments(
    output_dir="out/",
    per_device_train_batch_size=4,
    num_train_epochs=5,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained("out/tinyllama-scan-model")
tokenizer.save_pretrained("out/tinyllama-scan-model")
