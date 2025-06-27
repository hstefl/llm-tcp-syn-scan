import torch
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    pipeline,
)

# 1. Load Model and Tokenizer
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 2. Load and Preprocess the Dataset from CSV
full_dataset = load_dataset('csv', data_files='in/dataset.csv')
dataset = full_dataset['train']

# A function to format each row into our source/target format
prefix = "Analyze the TCP scan data and provide a risk assessment:\n\n"
source_columns = ["Time Window", "SYN Rate", "SYN Percentage", "Port Spread", "ACK Response Rate", "Source Diversity"]
target_columns = ["Scan Intensity", "Risk Assessment", "Scan Archetype", "Suggested Action", "Explanation"]


def create_prompt_and_target(example):
    source_parts = [f"{col}: {example[col]}" for col in source_columns]
    example['source_text'] = prefix + "\n".join(source_parts)

    target_parts = [f"{col}: {example[col]}" for col in target_columns]
    example['target_text'] = "<extra_id_0>".join(target_parts)
    return example


dataset = dataset.map(create_prompt_and_target, remove_columns=dataset.column_names)


# 3. Tokenize the processed dataset
def tokenize_function(examples):
    inputs = tokenizer(
        examples["source_text"],
        padding="max_length",
        truncation=True,
        max_length=512,  # Increased to handle longer lines
    )
    labels = tokenizer(
        examples["target_text"],
        padding="max_length",
        truncation=True,
        max_length=512,  # Increased to handle longer lines
    ).input_ids

    cleaned_labels = []
    for label_example in labels:
        cleaned_labels.append([l if l != tokenizer.pad_token_id else -100 for l in label_example])

    inputs["labels"] = cleaned_labels
    return inputs


tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['source_text', 'target_text'])

# 4. Set Up Training Arguments for Memorization
training_args = TrainingArguments(
    output_dir="./results_memorization_task",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs_memorization_task",
    logging_steps=100,
    evaluation_strategy="no",
    save_strategy="no",
    save_total_limit=1,
)

# 5. Fine-Tune the Model on the Entire Dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# 6. Save the final "memorized" model
final_model_path = "./final_memorized_model"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)

# 7. Use the Trained Model for Inference
print("\n--- Training Complete ---")
print(f"Final memorized model saved to {final_model_path}")

# Load the trained model for inference
memorized_pipeline = pipeline(
    "text2text-generation",
    model=final_model_path,
    tokenizer=final_model_path,
    device=0 if torch.cuda.is_available() else -1
)

# Test with one of the examples from your training set
test_source_text = prefix + """Time Window: very-short
SYN Rate: very low
SYN Percentage: minimal
Port Spread: one
ACK Response Rate: no response
Source Diversity: centralized"""

result = memorized_pipeline(test_source_text, max_length=512)  # Also increase here for generation

# Post-process the output
generated_text = result[0]['generated_text']
lines = generated_text.replace('<extra_id_0>', '\n').replace(tokenizer.pad_token, '').replace(tokenizer.eos_token,
                                                                                              '').split('\n')
stripped_lines = [line.strip() for line in lines]
formatted_text = '\n'.join(stripped_lines)

print("\n--- Inference Test ---")
print("Input:")
print(test_source_text)
print("\nPredicted Output (from memorized model):")
print(formatted_text)