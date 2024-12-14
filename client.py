#!/usr/bin/env python3

import tritonclient.http as httpclient
import numpy as np
from transformers import AutoTokenizer
import argparse

def call_triton(input_text):
    """
    Communicates with Triton Inference Server to get embeddings for the input text.

    Args:
        input_text (str): Input text for inference.

    Returns:
        list: List of embeddings from different models.
    """
    # Initialize Triton client
    client = httpclient.InferenceServerClient(url="localhost:8000")
    
    # Prepare input data
    inputs = []
    inputs.append(httpclient.InferInput("TEXT", [1], "BYTES"))
    inputs[0].set_data_from_numpy(np.array([input_text.encode('utf-8')], dtype=object))
    
    # Prepare output data
    output_names = ["EMBEDDING_ONNX", "EMBEDDING_TRT_FP16", "EMBEDDING_TRT_FP32", 
                    "EMBEDDING_TRT_INT8", "EMBEDDING_TRT_BEST"]
    outputs = [httpclient.InferRequestedOutput(name) for name in output_names]
    
    # Perform inference
    results = client.infer(model_name="ensemble", inputs=inputs, outputs=outputs)
    
    # Extract embeddings
    embeddings = [results.as_numpy(name) for name in output_names]
    
    return embeddings

def check_quality(input_text):
    """
    Calculates deviations between ONNX and TensorRT embeddings.

    Args:
        input_text (str): Input text for which deviations are calculated.

    Returns:
        list: List of mean squared errors between ONNX and other embeddings.
    """
    embeddings = call_triton(input_text)
    onnx_embedding = embeddings[0]
    trt_embeddings = embeddings[1:]
    
    # Calculate mean squared errors
    deviations = [np.mean((onnx_embedding - trt_emb) ** 2) for trt_emb in trt_embeddings]
    
    return deviations

def main():
    """
    Main function to parse arguments and calculate average deviations.
    """
    parser = argparse.ArgumentParser(description="Check Quality of TensorRT Embeddings")
    parser.add_argument("--texts", type=str, nargs='+', default=["This is a sample input for inference."],
                        help="List of input texts for quality check")
    args = parser.parse_args()
    
    total_deviation = np.zeros(4)
    count = 0
    
    # Process each text and calculate deviations
    for text in args.texts:
        deviations = check_quality(text)
        total_deviation += deviations
        count += 1
    
    # Calculate average deviations
    average_deviation = total_deviation / count
    print("Average Deviations:")
    print(f"ONNX vs TRT_FP16: {average_deviation[0]:.6f}")
    print(f"ONNX vs TRT_FP32: {average_deviation[1]:.6f}")
    print(f"ONNX vs TRT_INT8: {average_deviation[2]:.6f}")
    print(f"ONNX vs TRT_BEST: {average_deviation[3]:.6f}")

if __name__ == "__main__":
    main()
