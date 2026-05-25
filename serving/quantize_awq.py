from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def quantize(model_path: str, output_path: str) -> None:
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.quantize(tokenizer, quant_config=quant_config)
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
