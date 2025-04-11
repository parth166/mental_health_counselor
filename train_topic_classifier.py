import json
import os
import pandas as pd
import numpy as np
import joblib
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoConfig
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from transformers.modeling_outputs import SequenceClassifierOutput

class WeightedDistilBert(DistilBertForSequenceClassification):
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.class_weights = class_weights
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        # Remove unknown extras
        kwargs.pop("num_items_in_batch", None)

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            **kwargs
        )

        logits = outputs.logits
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )

def main():
    # ---------------------------
    # 2) Load + Prep Data
    # ---------------------------
    with open("data/data.json", "r") as f:
        data = json.load(f)

    rows = []
    for v in data.values():
        if v.get('question_text') is not None and v.get('topic') is not None:
            rows.append({"text": v["question_text"], "label": v["topic"]})

    df = pd.DataFrame(rows)
    print("Number of training rows:", len(df))

    # Encode labels
    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df["label"])

    # Class weights for imbalance
    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(df["label_id"]),
        y=df["label_id"]
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(device)

    # Save label encoder
    if not os.path.isdir("./models"):
        os.makedirs("./models")
    joblib.dump(label_encoder, "models/label_encoder.pkl")

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df[["text", "label_id"]])

    # Train/test split
    dataset = dataset.train_test_split(test_size=0.1)
    train_ds = dataset["train"]
    eval_ds = dataset["test"]

    # ---------------------------
    # 3) Tokenize
    # ---------------------------
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    eval_ds = eval_ds.map(tokenize_fn, batched=True)

    train_ds = train_ds.rename_column("label_id", "labels")
    eval_ds = eval_ds.rename_column("label_id", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    num_labels = len(label_encoder.classes_)

    # ---------------------------
    # 4) Build WeightedDistilBert
    # ---------------------------
    config = AutoConfig.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels
    )

    # Start with a standard DistilBert
    base_model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        config=config
    )

    # Create your custom model, load weights
    model = WeightedDistilBert(config=config, class_weights=class_weights)
    model.load_state_dict(base_model.state_dict(), strict=False)
    model.to(device)

    # ---------------------------
    # 5) Define Metrics
    # ---------------------------
    global_eval_labels = []
    global_eval_preds = []
    global_label_encoder = label_encoder

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        nonlocal global_eval_labels, global_eval_preds
        global_eval_labels = labels
        global_eval_preds = preds
        
        y_true = global_eval_labels
        y_pred = global_eval_preds

        unique_label_ids = sorted(list(set(y_true) | set(y_pred)))  
        # e.g. [0, 1, 2, 5, 7, ...]

        # 2) Map each ID to the correct label string from label_encoder
        target_names = [global_label_encoder.classes_[idx] for idx in unique_label_ids]

        accuracy_score = (preds == labels).mean()

        report = classification_report(
            y_true,
            y_pred,
            labels=unique_label_ids,
            target_names=target_names,
            output_dict=True
        )

        # Extract f1 for each class
        class_metrics = {
            f"f1_{label}": val["f1-score"]
            for label, val in report.items()
            if label in global_label_encoder.classes_
        }
        return {
            "accuracy": accuracy_score,
            **class_metrics
        }

    training_args = TrainingArguments(
        output_dir="models/bert_topic_model",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="logs",
        num_train_epochs=5,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Full final classification report
    print("\n\nFinal Classification Report:\n")
    print(classification_report(
        global_eval_labels,
        global_eval_preds,
        target_names=global_label_encoder.classes_
    ))

    trainer.save_model("models/bert_topic_model")  # Saves entire model
    tokenizer.save_pretrained("models/bert_topic_model")

if __name__ == "__main__":
    main()
