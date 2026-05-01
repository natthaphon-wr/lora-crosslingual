def preprocess(tokenizer, label_to_id, dataset):
  result = tokenizer(dataset["text"], truncation=True, max_length=128)
  result["labels"] = [label_to_id[label] for label in dataset["label_text"]]
  return result