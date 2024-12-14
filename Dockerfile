# FROM python:3.9-slim

# # Установите системные зависимости
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     libssl-dev \
#     libffi-dev \
#     libxml2-dev \
#     libxslt1-dev \
#     zlib1g-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Установите PyTorch с поддержкой CUDA 11.7
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# # Установите остальные зависимости
# RUN pip install tritonclient[all] numpy matplotlib onnx

# # Копируйте скрипт и модели внутрь контейнера
# WORKDIR /workspace
# COPY torch2onnx.py .
# COPY distilbert.onnx .

# # Установите права на выполнение скрипта
# RUN chmod +x torch2onnx.py

# # Команда по умолчанию
# CMD ["python3", "torch2onnx.py"]



# Используем базовый образ Triton Inference Server
FROM nvcr.io/nvidia/tritonserver:23.12-py3

# Установите Python-библиотеки, необходимые для подготовки данных и инференса
RUN pip install transformers onnxruntime numpy

# Создайте рабочую директорию для моделей и конфигураций
WORKDIR /workspace

# Скопируйте модельный репозиторий и ресурсы
COPY model_repository /models
COPY assets /workspace/assets

# Открываем стандартные порты Triton
EXPOSE 7000 7001 7002

# Указываем модельный репозиторий для Triton
CMD ["tritonserver", "--model-repository=/models"]
