import math
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, AdaLoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftConfig, PeftModel

def get_bnb_config():
  bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16,
    bnb_4bit_use_double_quant = True,
  )
  return bnb_config


def create_lora_model(pretrain_model_id, peft_model, tokenizer, lora_rank, lora_dropout):
  bnb_config = get_bnb_config()

  # Create model from pretrained model with quantization
  model = AutoModelForSequenceClassification.from_pretrained(
    pretrain_model_id,
    num_labels = 60,
    quantization_config = bnb_config,
    device_map = "auto",
    trust_remote_code = True,
    pad_token_id = tokenizer.eos_token_id
  )
  model = prepare_model_for_kbit_training(model)

  # Define PEFT LoRA or rsLoRA
  rslora = True if peft_model == "rslora" else False
  lora_config = LoraConfig(
    r = lora_rank,
    lora_alpha = lora_rank * 2,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout = lora_dropout,
    use_rslora = rslora,
    bias = "none",
    task_type = "SEQ_CLS",
  )

  # Merge pretrained model with PEFT
  model = get_peft_model(model, lora_config)

  return model


def create_adalora_model(pretrain_model_id, tokenizer, init_r, target_r, reg_weight, batch_size, num_epoch):
  bnb_config = get_bnb_config()

  # Create model from pretrained model with quantization
  model = AutoModelForSequenceClassification.from_pretrained(
    pretrain_model_id,
    num_labels = 60,
    quantization_config = bnb_config,
    device_map = "auto",
    trust_remote_code = True,
    pad_token_id = tokenizer.eos_token_id
  )
  model = prepare_model_for_kbit_training(model)

  # Define PEFT AdaLoRA
  toal_step = math.ceil(11500/batch_size)*num_epoch
  tinit = toal_step * 0.1
  tfinal = toal_step * 0.2
  adalora_config = AdaLoraConfig(
    total_step = toal_step, # total
    tinit = tinit, #steps of initial warmup
    tfinal = tfinal, #steps of final fine-tuning
    init_r = init_r,  
    target_r = target_r,
    orth_reg_weight = reg_weight,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias = "none",
    task_type = "SEQ_CLS",
  )

  # Merge pretrained model with PEFT
  model = get_peft_model(model, adalora_config)

  return model


def load_peft_model(peft_model_id):
  # Load config, pretrained model id
  bnb_config = get_bnb_config()
  peft_config = PeftConfig.from_pretrained(peft_model_id)
  pretrained_model_id = peft_config.base_model_name_or_path

  # Get tokenizer and padding
  tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"

  model = AutoModelForSequenceClassification.from_pretrained(
    peft_config.base_model_name_or_path,
    num_labels = 60,
    quantization_config=bnb_config,
    device_map = "auto",
    pad_token_id = tokenizer.eos_token_id
  )

  model = PeftModel.from_pretrained(model, peft_model_id)
  model.eval()

  return model, tokenizer, pretrained_model_id