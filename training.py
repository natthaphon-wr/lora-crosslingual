import argparse
import logging
import yaml
import os

from datasets import load_from_disk
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding

from dataset import preprocess
from model import create_model
from eval_metrics import compute_metrics

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s'
)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Arguments for training")
  parser.add_argument("--config_path", type=str, help="YAML configuration")

  args = parser.parse_args()
  config_path = args.config_path

  # Read YAML configuration
  with open(config_path, "r") as f:
    config = yaml.safe_load(f)
  output_path = config["output"]["output_path"]
  hg_name = config["output"]["hg_name"]
  data_path = config["data"]["data_path"]
  batch_size = config["data"]["batch_size"]
  model_id = config["model"]["model_id"]
  lora_rank = config["model"]["lora_rank"]
  lora_dropout = config["model"]["lora_dropout"]
  epoch = config["model"]["epoch"]
  lr = config["model"]["lr"]
  weight_decay = config["model"]["weight_decay"]
  os.makedirs(output_path)

  # Dataset
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"
  cols_removed = ["id", "label", "label_text", "text", "lang"]
  data_train = load_from_disk(os.path.join(data_path, "train"))
  data_val = load_from_disk(os.path.join(data_path, "validation"))
  unique_labels = sorted(list(set(data_train["label_text"])))
  label_to_id = {label: i for i, label in enumerate(unique_labels)}
  tokenized_train = data_train.map(preprocess, 
                                  fn_kwargs={"tokenizer": tokenizer, "label_to_id": label_to_id},
                                  batched=True, 
                                  remove_columns=cols_removed)
  tokenized_val = data_val.map(preprocess, 
                              fn_kwargs={"tokenizer": tokenizer, "label_to_id": label_to_id},
                              batched=True, 
                              remove_columns=cols_removed)
  logging.info("Finished preprocessed dataset")
  logging.info(f"Train: {len(tokenized_train)}, Val: {len(tokenized_val)}")

  # Model
  model = create_model(model_id, tokenizer, lora_rank, lora_dropout)
  logging.info(model.print_trainable_parameters())

  # Training
  training_args = TrainingArguments(
    output_dir = output_path,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    learning_rate = lr,
    num_train_epochs = epoch,
    eval_strategy = "steps",
    save_strategy = "epoch",
    optim = "adamw_torch",
    weight_decay = weight_decay,
    logging_steps = 50,
    eval_steps = 50,
    bf16 = True,
    report_to = "none"
  )
  trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_train,
    eval_dataset = tokenized_val,
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics = compute_metrics,
  )
  trainer.train()
  model.push_to_hub(hg_name) # save model to huggingface
  logging.info("Completed trained model and save model to HuggingFace")

  # Save some results
  with open(os.path.join(output_path, "model_params.txt"), 'w') as f:
    f.write("Model hyperparameters: \n")
    f.write(f"LoRA rank: {lora_rank}, LoRA dropout: {lora_dropout},\n")
    f.write(f"Epoch: {epoch}, lr: {lr}, weight_decay: {weight_decay}\n")
    f.write("Model parameters: \n")
    f.write(model.print_trainable_parameters())
  logging.info("Completed save model and training results")