import argparse
import logging
import yaml
import os
import pandas as pd

from datasets import load_dataset


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Arguments for download data")
  parser.add_argument("--config_path", type=str, help="YAML configuration")

  args = parser.parse_args()
  config_path = args.config_path

  # Read YAML configuration
  with open(config_path, "r") as f:
    config = yaml.safe_load(f)

  dataset_repo = config["dataset_repo"]
  save_root = config["save_root"]
  languages = config["languages"]

  # Download English data (train/val/test)
  dataset_eng = load_dataset(dataset_repo, "en")
  output_path = os.path.join(save_root, "en")
  dataset_eng.save_to_disk(output_path)
  logging.info(f"Completed saved train/val/test English data to {output_path}")

  # Download other languages
  for lang in languages:
    try:
      dataset = load_dataset(dataset_repo, lang, split="test")
      output_path = os.path.join(save_root, lang)
      dataset.save_to_disk(output_path)
      logging.info(f"Completed saved {lang} data to {output_path}")
    except Exception as e:
      logging.info(f"Error downloading {lang}: {e}")

