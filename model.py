import torch
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def create_model(model_id, tokenizer, lora_rank, lora_dropout):
  bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16,
    bnb_4bit_use_double_quant = True,
  )
  model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels = 60,
    quantization_config = bnb_config,
    device_map = "auto",
    trust_remote_code = True
  )
  model.config.pad_token_id = tokenizer.pad_token_id
  model = prepare_model_for_kbit_training(model)
  peft_config = LoraConfig(
      r = lora_rank,
      lora_alpha = lora_rank * 2,
      target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
      lora_dropout = lora_dropout,
      bias = "none",
      task_type = "SEQ_CLS",
  )
  model = get_peft_model(model, peft_config)

  return model