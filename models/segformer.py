import numpy as np
import torch
import wandb
from sklearn.metrics import f1_score
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

from models.configs_model.best_segformer import (
    MODEL_NAME,
    CLASSES,
    OUTPUT_SEGFORMER,
    PATIENCE,
    LOG_STEPS,
    WEIGHT_DECAY,
    EPOCHS,
    BATCH_SIZE,
    LR,
)


class Segformer:

    def __init__(self, model_name=MODEL_NAME, num_labels=CLASSES):
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        self.trainer = None

    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1).flatten()
        labels = labels.flatten()
        f1_macro = f1_score(labels, preds, average="macro")
        return {"f1_macro": f1_macro}

    def my_data_collator(self, features):
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        return {"pixel_values": pixel_values, "labels": labels}

    def train(
            self,
            train_dataset,
            val_dataset,
            output_dir=OUTPUT_SEGFORMER,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
    ):

        args = TrainingArguments(
            output_dir=output_dir,
            save_strategy="epoch",
            eval_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=WEIGHT_DECAY,
            logging_dir=output_dir,
            logging_steps=LOG_STEPS,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            fp16=True,
            load_best_model_at_end=True
        )

        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            data_collator=self.my_data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
        )
        self.trainer.train()

    def predict(self, dataset):
        predictions = self.trainer.predict(dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        return preds
