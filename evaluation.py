import argparse
import logging
import yaml
import os
import pandas as pd

from datasets import load_from_disk
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding

from dataset import preprocess
from model import load_saved_model
from eval_metrics import compute_metrics

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s'
)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Arguments for evaluation")
  parser.add_argument("--config_path", type=str, help="YAML configuration")

  args = parser.parse_args()
  config_path = args.config_path

  # Read YAML configuration
  with open(config_path, "r") as f:
    config = yaml.safe_load(f)

  output_path = config["output_path"]
  pretrained_model_name = config["model"]["pretrained_model_name"]
  peft_model_name = config["model"]["peft_model_name"]
  data_root = config["data"]["data_root"]
  batch_size = config["data"]["batch_size"]
  languages = config["data"].get("languages", [])

  # Model
  tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"
  model = load_saved_model(peft_model_name, tokenizer)

  # Evaluate in each language
  cols_removed = ["id", "label", "label_text", "text", "lang"]
  results_list = []
  for lang in languages:
    # dataset
    if lang == "en":
      data_path = os.path.join(data_root, lang, "test")
    else:
      data_path = os.path.join(data_root, lang)
    dataset = load_from_disk(data_path)
    unique_labels = sorted(list(set(dataset["label_text"])))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    tokenized_dataset = dataset.map(preprocess, 
                                    fn_kwargs={"tokenizer": tokenizer, "label_to_id": label_to_id},
                                    batched=True, 
                                    remove_columns=cols_removed)
    # evaluator
    eval_training_args = TrainingArguments(
      output_dir = output_path, 
      per_device_eval_batch_size = batch_size,
      report_to = "none",
    )
    evaluator = Trainer(
      model = model,
      args = eval_training_args,
      eval_dataset = tokenized_dataset,
      data_collator = DataCollatorWithPadding(tokenizer=tokenizer),
      compute_metrics = compute_metrics,
    )
    # evaluate
    eval_result = evaluator.evaluate(tokenized_dataset)
    result = {"language": lang, **eval_result}
    results_list.append(result)
  results_df = pd.DataFrame(results_list)
  results_df.to_csv(os.path.join(output_path, "eval_results.csv"))

