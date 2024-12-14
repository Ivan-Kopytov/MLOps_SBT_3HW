import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.models = {
            "model_ONNX": pb_utils.InferenceServerClient(url="localhost:8000").get_model_metadata("model_ONNX"),
            "model_TRT_FP16": pb_utils.InferenceServerClient(url="localhost:8000").get_model_metadata("model_TRT_FP16"),
            "model_TRT_FP32": pb_utils.InferenceServerClient(url="localhost:8000").get_model_metadata("model_TRT_FP32"),
            "model_TRT_INT8": pb_utils.InferenceServerClient(url="localhost:8000").get_model_metadata("model_TRT_INT8"),
            "model_TRT_BEST": pb_utils.InferenceServerClient(url="localhost:8000").get_model_metadata("model_TRT_BEST"),
        }

    def execute(self, requests):
        responses = []
        for request in requests:
            input_text = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy().astype(str).tolist()
            client = pb_utils.InferenceServerClient(url="localhost:8000")
            embeddings = []

            for model_name in ["model_ONNX", "model_TRT_FP16", "model_TRT_FP32", "model_TRT_INT8", "model_TRT_BEST"]:
                inputs = [
                    pb_utils.InferenceRequest.InputTensor.from_numpy("INPUT_IDS", ...),  # Заполните соответствующими данными
                    pb_utils.InferenceRequest.InputTensor.from_numpy("ATTENTION_MASK", ...)
                ]
                infer_request = pb_utils.InferenceRequest(model_name=model_name, inputs=inputs)
                infer_response = client.infer(infer_request)
                embedding = infer_response.as_numpy("EMBEDDING")
                embeddings.append(embedding)

            output_tensors = [
                pb_utils.Tensor("EMBEDDING_ONNX", embeddings[0]),
                pb_utils.Tensor("EMBEDDING_TRT_FP16", embeddings[1]),
                pb_utils.Tensor("EMBEDDING_TRT_FP32", embeddings[2]),
                pb_utils.Tensor("EMBEDDING_TRT_INT8", embeddings[3]),
                pb_utils.Tensor("EMBEDDING_TRT_BEST", embeddings[4]),
            ]

            inference_response = pb_utils.InferenceResponse(output_tensors=output_tensors)
            responses.append(inference_response)
        return responses
