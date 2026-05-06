import os
from datasets import load_from_disk

def get_label2id(data_root):
  data_path = os.path.join(data_root, "en", "train") #from eng train set
  dataset = load_from_disk(data_path)
  unique_labels = sorted(list(set(dataset["label_text"])))
  label_to_id = {label: i for i, label in enumerate(unique_labels)}
  return label_to_id

def preprocess(dataset, tokenizer, label_to_id):
  result = tokenizer(dataset["text"], truncation=True, max_length=128)
  result["labels"] = [label_to_id[label] for label in dataset["label_text"]]
  return result