#!/bin/bash

# Проверка наличия trtexec
if ! command -v trtexec &> /dev/null
then
    echo "trtexec не найден. Убедитесь, что TensorRT установлен и trtexec доступен в PATH."
    exit
fi

# Параметры модели
ONNX_MODEL="model.onnx"
OUTPUT_DIR="tensorrt_models"

# Создать каталог для сохранения моделей, если он не существует
mkdir -p ${OUTPUT_DIR}

# Параметры для trtexec (примерные, замените на актуальные)
MIN_SHAPES="INPUT_IDS:1x13,ATTENTION_MASK:1x13"
MAX_SHAPES="INPUT_IDS:8x13,ATTENTION_MASK:8x13"

# Конвертация в FP32
trtexec --onnx=${ONNX_MODEL} --saveEngine=${OUTPUT_DIR}/model_TRT_FP32.plan --minShapes=${MIN_SHAPES} --maxShapes=${MAX_SHAPES}

# Конвертация в FP16
trtexec --onnx=${ONNX_MODEL} --saveEngine=${OUTPUT_DIR}/model_TRT_FP16.plan --fp16 --minShapes=${MIN_SHAPES} --maxShapes=${MAX_SHAPES}

# Конвертация в INT8 (требует калибровочных данных)
trtexec --onnx=${ONNX_MODEL} --saveEngine=${OUTPUT_DIR}/model_TRT_INT8.plan --int8 --calib=calib_data.txt --minShapes=${MIN_SHAPES} --maxShapes=${MAX_SHAPES}

# Конвертация с оптимальными флагами
trtexec --onnx=${ONNX_MODEL} --saveEngine=${OUTPUT_DIR}/model_TRT_BEST.plan --best --minShapes=${MIN_SHAPES} --maxShapes=${MAX_SHAPES}

echo "Конвертация завершена. Файлы сохранены в ${OUTPUT_DIR}."
