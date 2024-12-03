from datasets import load_from_disk

dataset = load_from_disk("processed_dataset")
print(f"Number of rows: {len(dataset)}")
