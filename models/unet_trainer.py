import os
import torch
from matplotlib import pyplot as plt
from segmentation_models_pytorch.utils.metrics import IoU, Fscore
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from tqdm import trange
from losses.dice import NamedDiceLoss


class UNetTrainer:
    def __init__(
            self,
            model,
            train_loader,
            valid_loader,
            optimizer=None,
            epochs=30,
            device=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = [IoU(threshold=0.5), Fscore(threshold=0.5, beta=2)]
        self.optimizer = optimizer if optimizer else torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self.train_epoch = TrainEpoch(
            self.model,
            loss=NamedDiceLoss(mode="binary"),
            metrics=self.metrics,
            optimizer=self.optimizer,
            device=self.device,
            verbose=True
        )

        self.valid_epoch = ValidEpoch(
            self.model,
            loss=NamedDiceLoss(mode="binary"),
            metrics=self.metrics,
            device=self.device,
            verbose=True
        )

        self.loss_logs = {"train": [], "val": []}
        self.metric_logs = {"train": {"iou": [], "fscore": []},
                            "val": {"iou": [], "fscore": []}}
        self.max_score = 0

    def train(self, save_path="models/best_model.pth"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        for epoch in trange(self.epochs, desc="Epochs"):
            print(f"\nEpoch {epoch+1}/{self.epochs}")

            train_logs = self.train_epoch.run(self.train_loader)
            train_loss = train_logs.get("dice_loss", 0)
            train_iou = train_logs.get("iou_score", 0)
            train_fscore = train_logs.get("fscore", 0)
            self.loss_logs["train"].append(train_loss)
            self.metric_logs["train"]["iou"].append(train_iou)
            self.metric_logs["train"]["fscore"].append(train_fscore)

            valid_logs = self.valid_epoch.run(self.valid_loader)
            val_loss = valid_logs.get("dice_loss", 0)
            val_iou = valid_logs.get("iou_score", 0)
            val_fscore = valid_logs.get("fscore", 0)
            self.loss_logs["val"].append(val_loss)
            self.metric_logs["val"]["iou"].append(val_iou)
            self.metric_logs["val"]["fscore"].append(val_fscore)
            print(f"Train Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Fscore: {train_fscore:.4f}")
            print(f"Valid Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Fscore: {val_fscore:.4f}")

            if val_fscore > self.max_score:
                self.max_score = val_fscore
                torch.save(self.model, save_path)
                print(f"Best model saved (Fscore = {val_fscore:.4f})")

    def plot_metrics(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        axes[0].plot(self.loss_logs["train"], label="train")
        axes[0].plot(self.loss_logs["val"], label="val")
        axes[0].set_title("Dice Loss")
        axes[0].legend()

        axes[1].plot(self.metric_logs["train"]["iou"], label="train")
        axes[1].plot(self.metric_logs["val"]["iou"], label="val")
        axes[1].set_title("IoU")
        axes[1].legend()

        axes[2].plot(self.metric_logs["train"]["fscore"], label="train")
        axes[2].plot(self.metric_logs["val"]["fscore"], label="val")
        axes[2].set_title("Fscore")
        axes[2].legend()

        plt.show()