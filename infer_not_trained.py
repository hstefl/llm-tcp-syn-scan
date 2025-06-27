import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline

# --- THE ONLY MAJOR CHANGE IS HERE ---
# We now point to the original model on the Hugging Face Hub
# instead of our local fine-tuned model directory.
MODEL_NAME = "google/flan-t5-base"


def generate_assessment(scan_data: dict, inference_pipeline) -> str:
    """
    Generates a response using the BASE pre-trained model.

    Args:
        scan_data (dict): A dictionary containing the scan parameters.
        inference_pipeline: The loaded Hugging Face pipeline object.

    Returns:
        str: The formatted, human-readable model output.
    """
    # 1. We still create the same prompt format to see how the base model handles it.
    prefix = "Analyze the TCP scan data and provide a risk assessment:\n\n"

    source_parts = [f"{key}: {value}" for key, value in scan_data.items()]
    prompt = prefix + "\n".join(source_parts)

    # 2. Generate the output from the model
    result = inference_pipeline(prompt, max_length=512)
    generated_text = result[0]['generated_text']

    # 3. Post-processing
    # NOTE: The base model doesn't know about '<extra_id_0>', so we don't need to replace it.
    # We will just do basic cleanup of other special tokens.
    pad_token = inference_pipeline.tokenizer.pad_token
    eos_token = inference_pipeline.tokenizer.eos_token
    formatted_text = generated_text.replace(pad_token, '').replace(eos_token, '').strip()

    return formatted_text


# This block runs when you execute the script directly
if __name__ == "__main__":
    print(f"✅ Loading base model '{MODEL_NAME}' from Hugging Face Hub...")

    # 1. Load the original base model and tokenizer into a pipeline
    try:
        base_model_pipeline = pipeline(
            "text2text-generation",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            device=0 if torch.cuda.is_available() else -1  # Use GPU if available
        )
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit()

    # 2. Define the same sample input data
    sample_scan_data = {
        "Analyze the TCP scan data and provide a risk assessment"
        "Time Window": "very-short",
        "SYN Rate": "very low",
        "SYN Percentage": "minimal",
        "Port Spread": "one",
        "ACK Response Rate": "no response",
        "Source Diversity": "centralized"
    }

    print("\n--- Generating Assessment using BASE Model ---")

    # 3. Call the function to get the assessment
    final_output = generate_assessment(sample_scan_data, base_model_pipeline)

    # 4. Print the results
    print("\n📝 Input to Model:")
    for key, value in sample_scan_data.items():
        print(f"{key}: {value}")

    print("\n✨ Predicted Output (from Base Model):")
    print(final_output)