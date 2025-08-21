import os
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import trange
import matplotlib.pyplot as plt
from global_config import DEVICE
from models.configs_model.unet import BATCH_SIZE, LR, LR_STEP, LR_GAMMA


class UNetTrainer:
    def __init__(
            self,
            model,
            train_dataset,
            val_dataset,
            batch_size=BATCH_SIZE,
            lr=LR,
            lr_step=LR_STEP,
            lr_gamma=LR_GAMMA,
    ):
        self.device = DEVICE
        self.model = model.to(self.device)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.loss = smp.utils.losses.DiceLoss()
        self.metrics = [smp.utils.metrics.IoU(threshold=0.5),
                        smp.utils.metrics.Fscore(threshold=0.5, beta=2)]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_step, gamma=lr_gamma)
        self.train_epoch = smp.utils.train.TrainEpoch(
            self.model, loss=self.loss, metrics=self.metrics, optimizer=self.optimizer,
            device=self.device, verbose=True
        )
        self.valid_epoch = smp.utils.train.ValidEpoch(
            self.model, loss=self.loss, metrics=self.metrics, device=self.device, verbose=True
        )
        self.loss_logs = {"train": [], "val": []}
        self.metric_logs = {"train": {"iou": [], "f2score": []},
                            "val": {"iou": [], "f2score": []}}
        self.max_f2score = 0
    def train(self, epochs=10, save_path="models/best_model.pth"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        for epoch in trange(epochs, desc="Epochs"):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            train_logs = self.train_epoch.run(self.train_loader)
            val_logs = self.valid_epoch.run(self.valid_loader)
            self.loss_logs["train"].append(train_logs["dice_loss"])
            self.loss_logs["val"].append(val_logs["dice_loss"])
            self.metric_logs["train"]["iou"].append(train_logs["iou_score"])
            self.metric_logs["train"]["f2score"].append(train_logs["fscore"])
            self.metric_logs["val"]["iou"].append(val_logs["iou_score"])
            self.metric_logs["val"]["f2score"].append(val_logs["fscore"])
            print(
                f"Train Loss: {train_logs['dice_loss']:.4f}, IoU: {train_logs['iou_score']:.4f}, F2: {train_logs['fscore']:.4f}")
            print(
                f"Valid Loss: {val_logs['dice_loss']:.4f}, IoU: {val_logs['iou_score']:.4f}, F2: {val_logs['fscore']:.4f}")
            if val_logs["fscore"] > self.max_f2score:
                self.max_f2score = val_logs["fscore"]
                torch.save(self.model.state_dict(), save_path)
                print(f"✅ Best model saved (F2score = {self.max_f2score:.4f})")
            self.scheduler.step()

    @torch.no_grad()
    def predict(self, images):
        self.model.eval()
        images = images.to(self.device)
        preds = self.model(images)
        return preds

    def plot_metrics(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        axes[0].plot(self.loss_logs["train"], label="train")
        axes[0].plot(self.loss_logs["val"], label="val")
        axes[0].set_title("Loss (DiceLoss)")
        axes[0].legend()

        # IoU
        axes[1].plot(self.metric_logs["train"]["iou"], label="train")
        axes[1].plot(self.metric_logs["val"]["iou"], label="val")
        axes[1].set_title("IoU")
        axes[1].legend()

        # F2score
        axes[2].plot(self.metric_logs["train"]["f2score"], label="train")
        axes[2].plot(self.metric_logs["val"]["f2score"], label="val")
        axes[2].set_title("F2score")
        axes[2].legend()

        plt.show()