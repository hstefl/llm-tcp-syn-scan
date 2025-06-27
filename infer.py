import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline

# Define the path to your fine-tuned model
MODEL_PATH = "./final_memorized_model"


def generate_assessment(scan_data: dict, inference_pipeline) -> str:
    """
    Generates a risk assessment using the fine-tuned model.

    Args:
        scan_data (dict): A dictionary containing the scan parameters.
        inference_pipeline: The loaded Hugging Face pipeline object.

    Returns:
        str: The formatted, human-readable risk assessment.
    """
    # 1. Recreate the exact prompt format used during training
    prefix = "Analyze the TCP scan data and provide a risk assessment:\n\n"

    source_parts = [f"{key}: {value}" for key, value in scan_data.items()]
    prompt = prefix + "\n".join(source_parts)

    # 2. Generate the output from the model
    # We also increase max_length here to allow for long explanations.
    result = inference_pipeline(prompt, max_length=512)
    generated_text = result[0]['generated_text']

    # 3. Recreate the exact post-processing steps from training
    # First, replace our placeholder with a newline
    formatted_text = generated_text.replace('<extra_id_0>', '\n')

    # Now, remove any other special tokens that might appear
    pad_token = inference_pipeline.tokenizer.pad_token
    eos_token = inference_pipeline.tokenizer.eos_token
    formatted_text = formatted_text.replace(pad_token, '').replace(eos_token, '')

    # Finally, clean up whitespace on each line individually for a clean output
    lines = formatted_text.split('\n')
    stripped_lines = [line.strip() for line in lines]
    formatted_text = '\n'.join(stripped_lines)

    return formatted_text


# This block runs when you execute the script directly
if __name__ == "__main__":
    print(f"✅ Loading model from {MODEL_PATH}...")

    # 1. Load the fine-tuned model and tokenizer into a pipeline
    # This is the standard and easiest way to use a model for inference.
    # It handles tokenization, model prediction, and decoding.
    try:
        risk_assessment_pipeline = pipeline(
            "text2text-generation",
            model=MODEL_PATH,
            tokenizer=MODEL_PATH,
            device=0 if torch.cuda.is_available() else -1  # Use GPU if available
        )
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please ensure you have trained the model and it exists at the specified path.")
        exit()

    # 2. Define some sample input data
    # This should be one of the examples from your CSV to test memorization
    sample_scan_data = {
        "Time Window": "very-short",
        "SYN Rate": "very low",
        "SYN Percentage": "minimal",
        "Port Spread": "one",
        "ACK Response Rate": "no response",
        "Source Diversity": "centralized"
    }

    print("\n--- Generating Assessment for Sample Data ---")

    # 3. Call the function to get the assessment
    final_output = generate_assessment(sample_scan_data, risk_assessment_pipeline)

    # 4. Print the results
    print("\n📝 Input to Model:")
    for key, value in sample_scan_data.items():
        print(f"{key}: {value}")

    print("\n✨ Predicted Output:")
    print(final_output)