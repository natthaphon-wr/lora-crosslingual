import argparse
import logging
import yaml

from datasets import load_dataset
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
  batch_size = config["data"]["batch_size"]
  model_id = config["model"]["model_id"]
  lora_rank = config["model"]["lora_rank"]
  lora_dropout = config["model"]["lora_dropout"]
  epoch = config["model"]["epoch"]
  lr = config["model"]["lr"]
  weight_decay = config["model"]["weight_decay"]

  # Dataset
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"
  dataset = load_dataset("mteb/amazon_massive_intent", "en")
  cols_removed = ["id", "label", "label_text", "text", "lang"]
  tokenized_train = dataset["train"].map(preprocess, batched=True, remove_columns=cols_removed)
  tokenized_val = dataset["validation"].map(preprocess, batched=True, remove_columns=cols_removed)
  tokenized_test = dataset["test"].map(preprocess, batched=True, remove_columns=cols_removed)
  logging.info("Finished preprocessed dataset")
  logging.info(f"Train: {len(tokenized_train)}, Val: {len(tokenized_val)}, Test: {len(tokenized_test)}")

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
    weight_decay = 0.01,
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

  # Evaluation on English
  # logging.info("Evalutation on English test set")
  # eval_result = trainer.evaluate(tokenized_test)
  # results_path = output_path + "/test_results.txt"
  # with open(results_path, 'w') as f:
  #   f.write("Model hyperparameters: \n")
  #   f.write(f"LoRA rank: {lora_rank}, LoRA dropout: {lora_dropout},\n")
  #   f.write(f"Epoch: {epoch}, lr: {lr}, weight_decay: {weight_decay}\n")
  #   f.write("-------------------------------------------------------\n")
  #   f.write("Evalutation on English test set: \n")
  #   f.write(f"Loss: {eval_result["eval_loss"]}, Acc: {eval_result["eval_accuracy"]}, F1: {eval_result["eval_f1"]}")
  # logging.info("Completed save model and training results")