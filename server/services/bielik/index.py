from server.vllm import vllm, VllmOptions

service = vllm(VllmOptions(
    model_id="speakleash/Bielik-11B-v2.6-Instruct-AWQ",
    hf_token=None, 
    env_vars={
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"
    },
    quantization="awq_marlin",
    gpu_memory_utilization=0.95,
    max_model_len=4096,
))