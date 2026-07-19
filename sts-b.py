!pip -q install transformers datasets accelerate scikit-learn pandas scipy

import time
import numpy as np
from scipy.stats import pearsonr, spearmanr
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

MODEL_NAME = "distilbert-base-uncased"
TASK = "stsb"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dataset = load_dataset("nyu-mll/glue", TASK)

def preprocess(examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True,
        max_length=128
    )

dataset = dataset.map(preprocess, batched=True)

data_collator = DataCollatorWithPadding(tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    problem_type="regression"
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.squeeze(predictions)

    return {
        "pearson": pearsonr(predictions, labels)[0],
        "spearman": spearmanr(predictions, labels)[0],
    }

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,
    logging_steps=100,
    save_strategy="no",
    eval_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

start_train = time.time()
trainer.train()
train_time = time.time() - start_train

start_eval = time.time()
metrics = trainer.evaluate()
eval_time = time.time() - start_eval

memory_size_mb = sum(
    p.numel() * p.element_size()
    for p in model.parameters()
) / (1024**2)

throughput = len(dataset["validation"]) / eval_time
latency_ms = (eval_time / len(dataset["validation"])) * 1000
energy_wh = (70 * (train_time + eval_time)) / 3600

print("\n===== RESULTS STS-B =====")
print("Memory Size:", round(memory_size_mb,2), "MB")
print("Latency:", round(latency_ms,4), "ms/sample")
print("Pearson:", round(metrics["eval_pearson"],4))
print("Spearman:", round(metrics["eval_spearman"],4))
print("Throughput:", round(throughput,2))
print("Energy:", round(energy_wh,4), "Wh")
print("Bits:", 32)
