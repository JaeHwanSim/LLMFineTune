from datasets import load_dataset

dataset = load_dataset("beomi/KoAlpaca-v1.1a", split="train[:1]")
print("Column names:", dataset.column_names)
print("Sample:", dataset[0])
