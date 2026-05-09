import argparse
import logging
import yaml
import os
import pandas as pd

from datasets import load_from_disk
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding

from dataset import preprocess, get_label2id
from model import load_peft_model
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
  peft_model_id = config["model"]["peft_model_id"]
  data_root = config["data"]["data_root"]
  batch_size = config["data"]["batch_size"]
  languages = config["data"].get("languages", [])
  logging.info(f"Evaluate on total {len(languages)} languages: ")
  logging.info(languages)

  # Model
  model, tokenizer, pretrained_model_id = load_peft_model(peft_model_id)
  logging.info(f"Completed load peft model, {peft_model_id}, using pretrained {pretrained_model_id}")

  # Evaluate in each language
  label_to_id = get_label2id(data_root)
  logging.info(f"Number of unique labels: {len(label_to_id)}")
  cols_removed = ["id", "label", "label_text", "text", "lang"]
  results_list = []
  for i, lang in enumerate(languages):
    logging.info(f"Evaluating on {lang} {i+1}/{len(languages)} ....")
    # dataset
    if lang == "en":
      data_path = os.path.join(data_root, lang, "test")
    else:
      data_path = os.path.join(data_root, lang)
    dataset = load_from_disk(data_path)
    tokenized_dataset = dataset.map(preprocess, 
                                    fn_kwargs={"tokenizer": tokenizer, "label_to_id": label_to_id},
                                    batched=True, 
                                    remove_columns=cols_removed)
    # evaluator
    eval_training_args = TrainingArguments(
      output_dir = output_path, 
      per_device_eval_batch_size = batch_size,
      do_train=False,
      do_eval=True,
    )
    evaluator = Trainer(
      model = model,
      args = eval_training_args,
      eval_dataset = tokenized_dataset,
      data_collator = DataCollatorWithPadding(tokenizer=tokenizer),
      compute_metrics = compute_metrics,
    )
    # evaluate
    eval_result = evaluator.evaluate()
    result = {"language": lang, **eval_result}
    logging.info(result)
    results_list.append(result)
  logging.info(f"Completed evaluate model on all {len(languages)} languages")
  results_df = pd.DataFrame(results_list)
  fname = "eval_results_" + peft_model_id.split("/")[1] + ".csv"
  results_df.to_csv(os.path.join(output_path, fname))
  logging.info(f"Completed saved performance results to {os.path.join(output_path, fname)}")