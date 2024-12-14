#!/usr/bin/env python3

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import onnx
import onnxruntime
import numpy as np
import argparse
import os


from fvcore.nn import FlopCountAnalysis, parameter_count

class TransformerONNXExporter(nn.Module):
    def __init__(self, model_name: str, output_dim: int):
        super(TransformerONNXExporter, self).__init__()
        # Инициализируем выбранный трансформер
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_dim = self.transformer.config.hidden_size
        # Инициализируем полносвязанный слой для понижения размерности
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # Получаем последние скрытые состояния
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # shape: (batch_size, seq_length, hidden_dim)
        # Применяем полносвязанный слой
        reduced = self.fc(last_hidden_state)  # shape: (batch_size, seq_length, output_dim)
        return reduced

def export_to_onnx(model, sample_inputs, onnx_path):
    torch.onnx.export(
        model,
        args=sample_inputs,
        f=onnx_path,
        opset_version=18,
        input_names=['INPUT_IDS', 'ATTENTION_MASK'],
        output_names=['EMBEDDING'],
        dynamic_axes={
            'INPUT_IDS': {0: 'BATCH_SIZE'},
            'ATTENTION_MASK': {0: 'BATCH_SIZE'},
            'EMBEDDING': {0: 'BATCH_SIZE'}
        }
    )
    print(f"Model exported to {onnx_path}")

def sanitize_onnx(onnx_path, model):
    # Загружаем модель
    ort_session = onnxruntime.InferenceSession(onnx_path)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # Создаем тестовые входные данные
    sample_text = "This is a sample input for ONNX export."
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    inputs = tokenizer(sample_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Получаем выходы из PyTorch модели
    with torch.no_grad():
        torch_output = model(input_ids, attention_mask)

    # Получаем выходы из ONNX модели
    ort_inputs = {
        'INPUT_IDS': to_numpy(input_ids),
        'ATTENTION_MASK': to_numpy(attention_mask)
    }
    ort_outs = ort_session.run(None, ort_inputs)
    onnx_output = ort_outs[0]

    # Сравниваем выходы
    try:
        np.testing.assert_allclose(
            to_numpy(torch_output),
            onnx_output,
            rtol=1e-03,
            atol=1e-05
        )
        print("ONNX model has been sanitized successfully and outputs match.")
    except AssertionError as e:
        print("Sanity check failed. Outputs do not match.")
        print(e)

def export_tokenizer(model_name: str, save_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_dir)
    print(f"Tokenizer exported to {save_dir}")

def compute_flops(model, sample_inputs):
    # Перенос модели в режим оценки
    model.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(model, sample_inputs)
        total_flops = flops.total()
        per_layer_flops = flops.by_module()

    param_count = parameter_count(model)
    print(f"Total FLOPs: {total_flops}")
    print(f"Parameter count: {param_count}")

    # Классификация слоёв
    arithmetic_limited = []
    memory_limited = []
    batch_size_threshold = {}

    for module, flop in per_layer_flops.items():
        # Пример простой классификации
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            arithmetic_limited.append((module.__class__.__name__, flop))
        else:
            memory_limited.append((module.__class__.__name__, flop))

    print("\nArithmetic limited layers:")
    for layer, fl in arithmetic_limited:
        print(f"{layer}: {fl}")

    print("\nMemory limited layers:")
    for layer, fl in memory_limited:
        print(f"{layer}: {fl}")

    # Пример определения порогового batch size (зависит от конкретной модели и оборудования)
    # Здесь мы просто указываем примерное значение
    batch_size_threshold = 32
    print(f"\nBatch size threshold for arithmetic limitation: {batch_size_threshold}")

def main():
    parser = argparse.ArgumentParser(description="Export Transformer Model to ONNX and TensorRT")
    parser.add_argument("--model_name", type=str, default="distilbert-base-cased-distilled-squad",
                        help="Name of the pre-trained model from HuggingFace")
    parser.add_argument("--output_dim", type=int, default=768,
                        help="Output dimension for the fully connected layer (N)")
    parser.add_argument("--onnx_path", type=str, default="model.onnx",
                        help="Path to save the exported ONNX model")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer",
                        help="Directory to save the exported tokenizer")
    args = parser.parse_args()

    # Инициализируем модель
    model = TransformerONNXExporter(args.model_name, args.output_dim)
    model.eval()

    # Подготавливаем пример входных данных
    sample_text = "This is a sample input for ONNX export."
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    inputs = tokenizer(sample_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    sample_inputs = (input_ids, attention_mask)

    # Экспортируем модель в ONNX
    export_to_onnx(model, sample_inputs, args.onnx_path)

    # Выполняем санити-чек
    sanitize_onnx(args.onnx_path, model)

    # Экспортируем токенайзер
    export_tokenizer(args.model_name, args.tokenizer_dir)

    # Вычисляем FLOPs
    compute_flops(model, sample_inputs)

if __name__ == "__main__":
    main()
