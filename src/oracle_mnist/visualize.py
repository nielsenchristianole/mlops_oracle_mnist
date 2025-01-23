import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from sklearn.manifold import TSNE


class VisualizeCallback(Callback):
    def __init__(self, log_dir, tsne_perplexity=30, tsne_n_iter=300):
        super().__init__()
        self.log_dir = log_dir
        self.tsne_perplexity = tsne_perplexity
        self.tsne_n_iter = tsne_n_iter

    def on_train_epoch_end(self, trainer, pl_module):
        # Visualize training metrics
        metrics = trainer.logged_metrics
        train_loss = metrics.get("train_loss", None)
        val_loss = metrics.get("val_loss", None)
        train_acc = metrics.get("train_acc", None)
        val_acc = metrics.get("val_acc", None)

        self.plot_metrics(train_loss, val_loss, train_acc, val_acc)

    def on_fit_end(self, trainer, pl_module):
        # Extract data and visualize
        data_module = trainer.datamodule

        train_embeddings, train_labels = self.generate_embeddings(pl_module, data_module.train_dataloader())
        test_embeddings, test_labels = self.generate_embeddings(pl_module, data_module.test_dataloader())

        self.plot_tsne(train_embeddings, train_labels, title="t-SNE Visualization (Training Data)")
        self.plot_tsne(test_embeddings, test_labels, title="t-SNE Visualization (Test Data)")

    def generate_embeddings(self, model, dataloader):
        model.eval()
        embeddings = []
        labels = []

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(model.device)
                y = y.to(model.device)
                outputs = model(x)
                embeddings.append(outputs.cpu().numpy())
                labels.append(y.cpu().numpy())

        embeddings = np.vstack(embeddings)
        labels = np.concatenate(labels)
        return embeddings, labels

    def plot_tsne(self, embeddings, labels, title):
        tsne = TSNE(n_components=2, perplexity=self.tsne_perplexity, n_iter=self.tsne_n_iter, random_state=42)
        tsne_results = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=2, alpha=0.6)
        plt.title(title)
        plt.xlabel("t-SNE Dim 1")
        plt.ylabel("t-SNE Dim 2")
        plt.savefig(f"{self.log_dir}/{title.replace(' ', '_')}.png")
        plt.close()

    def plot_metrics(self, train_loss, val_loss, train_acc, val_acc):
        epochs = range(1, len(train_loss) + 1)

        plt.figure(figsize=(10, 5))
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Over Epochs")

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc, label="Train Accuracy")
        plt.plot(epochs, val_acc, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy Over Epochs")

        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/metrics.png")
        plt.close()
