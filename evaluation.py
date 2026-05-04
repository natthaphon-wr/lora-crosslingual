import argparse
import logging
import yaml
import pandas as pd

from transformers import AutoTokenizer

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
  pretrained_model = config["model"]["pretrained_model"]
  lora_model = config["model"]["lora_model"]
  languages = config.get("languages", [])

  # Dataset
  tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"

