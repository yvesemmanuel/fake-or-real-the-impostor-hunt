import os
import json
import random
from datetime import datetime
from typing import Optional, List, Tuple, Literal, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel, AutoTokenizer

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import seaborn as sns


MODEL_NAME = "omykhailiv/bert-fake-news-recognition"


class TextTruthnessDataset(Dataset):
    def __init__(self, text_pairs, labels, tokenizer, max_length=512, augmenter=None):
        self.text_pairs = text_pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augmenter = augmenter

    def __len__(self):
        return len(self.text_pairs)

    def __getitem__(self, idx):
        text_1, text_2 = self.text_pairs[idx]
        label = self.labels[idx]

        if self.augmenter:
            original_text_1, original_text_2 = text_1, text_2
            text_1 = self.augmenter.augment(text_1)
            text_2 = self.augmenter.augment(text_2)

            if isinstance(text_1, list):
                text_1 = text_1[0] if text_1 else original_text_1
            if isinstance(text_2, list):
                text_2 = text_2[0] if text_2 else original_text_2

        encoding_1 = self.tokenizer(
            text_1,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        encoding_2 = self.tokenizer(
            text_2,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        binary_label = 1 if label == 1 else 0

        return {
            "text_1_input_ids": encoding_1["input_ids"].squeeze(),
            "text_1_attention_mask": encoding_1["attention_mask"].squeeze(),
            "text_2_input_ids": encoding_2["input_ids"].squeeze(),
            "text_2_attention_mask": encoding_2["attention_mask"].squeeze(),
            "label": torch.tensor(binary_label, dtype=torch.float),
        }


class SiameseTextClassifier(nn.Module):
    def __init__(
        self, model_name=MODEL_NAME, dropout_rate=0.2, pooling_strategy="cls"
    ):
        super(SiameseTextClassifier, self).__init__()

        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.text_encoder.config.hidden_size
        self.pooling_strategy = pooling_strategy

        self.representation_layer = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.comparison_layer = nn.Sequential(
            nn.Linear(256 * 4, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def encode_text(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        if self.pooling_strategy == "cls":
            pooled_output = outputs.last_hidden_state[:, 0, :]
        elif self.pooling_strategy == "mean":
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )

            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        elif self.pooling_strategy == "max":
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )

            token_embeddings[input_mask_expanded == 0] = -1e9
            pooled_output = torch.max(token_embeddings, 1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        representation = self.representation_layer(pooled_output)
        return representation

    def forward(
        self,
        text_1_input_ids,
        text_1_attention_mask,
        text_2_input_ids,
        text_2_attention_mask,
    ):
        repr_1 = self.encode_text(text_1_input_ids, text_1_attention_mask)
        repr_2 = self.encode_text(text_2_input_ids, text_2_attention_mask)

        diff = repr_1 - repr_2

        product = repr_1 * repr_2

        combined = torch.cat([repr_1, repr_2, diff, product], dim=1)

        prediction = self.comparison_layer(combined)

        return prediction.squeeze()

    def _init_weights(self):
        """Initialize weights for better convergence."""
        for module in [self.representation_layer, self.comparison_layer]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)


def load_dataset(data_dir, labels_file) -> tuple[List[Tuple[str, str]], List[str]]:
    """Load the text pairs and labels from the dataset structure."""
    labels_df = pd.read_csv(labels_file)

    text_pairs: List[Tuple[str, str]] = []
    labels: List[str] = []

    for _, row in labels_df.iterrows():
        article_id = row["id"]
        real_text_id = row["real_text_id"]

        article_dir: str = os.path.join(data_dir, f"article_{article_id:04d}")
        file_1_path: str = os.path.join(article_dir, "file_1.txt")
        file_2_path: str = os.path.join(article_dir, "file_2.txt")

        try:
            with open(file_1_path, "r", encoding="utf-8") as f:
                text_1: str = f.read()
            with open(file_2_path, "r", encoding="utf-8") as f:
                text_2: str = f.read()

            text_pairs.append((text_1, text_2))
            labels.append(real_text_id)

        except FileNotFoundError:
            print(f"Warning: Could not find files for article {article_id}")
            continue

    return text_pairs, labels


def plot_training_history(history, save_dir) -> None:
    """Plot and save training history."""
    plt.style.use("seaborn-v0_8")
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "b-", label="Training Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2)
    ax1.set_title("Model Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-", label="Training Accuracy", linewidth=2)
    ax2.plot(epochs, history["val_acc"], "r-", label="Validation Accuracy", linewidth=2)
    ax2.set_title("Model Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()

    plot_path = os.path.join(save_dir, "training_history.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Training history plot saved to {plot_path}")

    plt.close()


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    eps=1e-6,
    save_dir="transformers/models",
) -> dict[str, List[Any]]:
    """Training function with proper loss calculation and validation."""
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=eps
    )

    max_grad_norm = 1.0

    total_steps: int = len(train_loader) * num_epochs
    warmup_steps: int = total_steps // 10

    def lr_lambda(current_step) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step)
            / float(max(1, total_steps - warmup_steps)),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    os.makedirs(save_dir, exist_ok=True)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_acc = 0
    step_count = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch in train_loader:
            optimizer.zero_grad()

            text_1_input_ids = batch["text_1_input_ids"].to(device)
            text_1_attention_mask = batch["text_1_attention_mask"].to(device)
            text_2_input_ids = batch["text_2_input_ids"].to(device)
            text_2_attention_mask = batch["text_2_attention_mask"].to(device)
            labels = batch["label"].to(device)

            predictions = model(
                text_1_input_ids,
                text_1_attention_mask,
                text_2_input_ids,
                text_2_attention_mask,
            )

            predictions = predictions.view(-1)
            labels = labels.view(-1)
            loss = criterion(predictions, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()
            step_count += 1

            total_train_loss += loss.item()

            predicted_labels = (predictions > 0.5).float()
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)

        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                text_1_input_ids = batch["text_1_input_ids"].to(device)
                text_1_attention_mask = batch["text_1_attention_mask"].to(device)
                text_2_input_ids = batch["text_2_input_ids"].to(device)
                text_2_attention_mask = batch["text_2_attention_mask"].to(device)
                labels = batch["label"].to(device)

                predictions = model(
                    text_1_input_ids,
                    text_1_attention_mask,
                    text_2_input_ids,
                    text_2_attention_mask,
                )

                predictions = predictions.view(-1)
                labels = labels.view(-1)
                loss = criterion(predictions, labels)
                total_val_loss += loss.item()

                predicted_labels = (predictions > 0.5).float()
                val_correct += (predicted_labels == labels).sum().item()
                val_total += labels.size(0)

        avg_train_loss: float = total_train_loss / len(train_loader)
        avg_val_loss: float = total_val_loss / len(val_loader)
        train_accuracy: float = correct_predictions / total_predictions
        val_accuracy: float = val_correct / val_total

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_accuracy)
        history["val_acc"].append(val_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                    "history": history,
                },
                os.path.join(save_dir, "best_model.pt"),
            )
            print(f"  New best model saved! Val Acc: {val_accuracy:.4f}")

        plateau_scheduler.step(avg_val_loss)

    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": val_accuracy,
            "history": history,
        },
        os.path.join(save_dir, "final_model.pt"),
    )

    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    plot_training_history(history, save_dir)

    return history


def k_fold_training(
    text_pairs,
    labels,
    tokenizer,
    model_name=MODEL_NAME,
    pooling_strategy="cls",
    num_epochs=10,
    learning_rate=2e-5,
    k=5,
    augmenter=None,
    save_dir="transformers/models",
):
    """K-fold cross-validation training loop for robust model evaluation."""
    from sklearn.model_selection import KFold
    from torch.utils.data import Subset

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_results = []
    best_model_state = None
    best_val_acc = 0

    print(f"=== Starting {k}-Fold Cross-Validation ===")

    for fold, (train_idx, val_idx) in enumerate(kf.split(text_pairs)):
        print(f"=== Starting Fold {fold + 1}/{k} ===")

        train_dataset = TextTruthnessDataset(
            [text_pairs[i] for i in train_idx],
            [labels[i] for i in train_idx],
            tokenizer,
            augmenter=augmenter,
        )

        val_dataset = TextTruthnessDataset(
            [text_pairs[i] for i in val_idx], [labels[i] for i in val_idx], tokenizer
        )

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

        model = SiameseTextClassifier(
            model_name=model_name, pooling_strategy=pooling_strategy
        )

        fold_save_dir = os.path.join(save_dir, f"fold_{fold + 1}")

        history = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs,
            learning_rate,
            save_dir=fold_save_dir,
        )

        final_val_acc = history["val_acc"][-1]
        if final_val_acc > best_val_acc:
            best_val_acc = final_val_acc
            best_model_state = model.state_dict().copy()

        fold_results.append(
            {
                "fold": fold + 1,
                "history": history,
                "val_acc": final_val_acc,
                "best_val_acc": max(history["val_acc"]),
            }
        )

        print(
            f"Fold {fold + 1} completed - Final Val Acc: {final_val_acc:.4f}, Best Val Acc: {max(history['val_acc']):.4f}"
        )

    val_accuracies = [result["val_acc"] for result in fold_results]
    best_val_accuracies = [result["best_val_acc"] for result in fold_results]

    cv_mean = np.mean(val_accuracies)
    cv_std = np.std(val_accuracies)

    print(f"\n=== Cross-Validation Results ===")
    print(f"Final Validation Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
    print(
        f"Best Validation Accuracy: {np.mean(best_val_accuracies):.4f} ± {np.std(best_val_accuracies):.4f}"
    )

    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_model_state,
            "fold_results": fold_results,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "best_overall_acc": best_val_acc,
        },
        os.path.join(save_dir, "best_kfold_model.pt"),
    )

    with open(os.path.join(save_dir, "kfold_results.json"), "w") as f:
        serializable_results = []
        for result in fold_results:
            serializable_result = {
                "fold": result["fold"],
                "val_acc": float(result["val_acc"]),
                "best_val_acc": float(result["best_val_acc"]),
                "history": {
                    "train_loss": [float(x) for x in result["history"]["train_loss"]],
                    "val_loss": [float(x) for x in result["history"]["val_loss"]],
                    "train_acc": [float(x) for x in result["history"]["train_acc"]],
                    "val_acc": [float(x) for x in result["history"]["val_acc"]],
                },
            }
            serializable_results.append(serializable_result)

        json.dump(
            {
                "fold_results": serializable_results,
                "cv_mean": float(cv_mean),
                "cv_std": float(cv_std),
                "best_overall_acc": float(best_val_acc),
            },
            f,
            indent=2,
        )

    return fold_results, best_model_state


def predict_on_test_set(
    model, test_data_dir, tokenizer, predictions_dir="transformers/predictions"
):
    """Generate predictions for the test set."""
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model.to(device)
    model.eval()

    os.makedirs(predictions_dir, exist_ok=True)

    test_articles = [d for d in os.listdir(test_data_dir) if d.startswith("article_")]
    test_articles.sort()

    predictions = []

    print(f"Processing {len(test_articles)} test articles...")

    for i, article_dir in enumerate(test_articles):
        article_id = int(article_dir.split("_")[1])

        file_1_path = os.path.join(test_data_dir, article_dir, "file_1.txt")
        file_2_path = os.path.join(test_data_dir, article_dir, "file_2.txt")

        try:
            with open(file_1_path, "r", encoding="utf-8") as f:
                text_1 = f.read().strip()
            with open(file_2_path, "r", encoding="utf-8") as f:
                text_2 = f.read().strip()
        except FileNotFoundError:
            print(f"Warning: Could not find files for test article {article_id}")
            continue

        encoding_1 = tokenizer(
            text_1,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        encoding_2 = tokenizer(
            text_2,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

        text_1_input_ids = encoding_1["input_ids"].to(device)
        text_1_attention_mask = encoding_1["attention_mask"].to(device)
        text_2_input_ids = encoding_2["input_ids"].to(device)
        text_2_attention_mask = encoding_2["attention_mask"].to(device)

        with torch.no_grad():
            probability_text_1_true = model(
                text_1_input_ids,
                text_1_attention_mask,
                text_2_input_ids,
                text_2_attention_mask,
            ).item()

        predicted_real_text_id: Literal[1, 2] = (
            1 if probability_text_1_true > 0.5 else 2
        )

        predictions.append(
            {
                "id": article_id,
                "real_text_id": predicted_real_text_id,
                "probability_text_1_true": probability_text_1_true,
            }
        )

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(test_articles)} articles...")

    predictions_df = pd.DataFrame(predictions)
    predictions_df = predictions_df.sort_values("id")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_file = os.path.join(
        predictions_dir, f"test_predictions_{timestamp}.csv"
    )

    predictions_df[["id", "real_text_id"]].to_csv(predictions_file, index=False)

    detailed_file = os.path.join(
        predictions_dir, f"detailed_predictions_{timestamp}.csv"
    )
    predictions_df.to_csv(detailed_file, index=False)

    print(f"Predictions saved to {predictions_file}")
    print(f"Detailed predictions saved to {detailed_file}")

    return predictions_df


if __name__ == "__main__":
    model_name = MODEL_NAME
    pooling_strategy = "cls"
    max_length = 512
    batch_size = 4
    num_epochs = 15
    learning_rate = 5e-5

    training_strategy = "kfold"
    k_folds = 5

    use_augmentation = True

    train_data_dir = "data/train"
    test_data_dir = "data/test"
    train_labels_file = "data/train.csv"

    print("Loading dataset...")
    text_pairs, labels = load_dataset(train_data_dir, train_labels_file)
    print(f"Loaded {len(text_pairs)} text pairs")

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    augmenter = None
    if use_augmentation:
        try:
            import nlpaug.augmenter.word as naw

            augmenter = naw.SynonymAug(aug_src="wordnet", aug_max=2)
            print("Data augmentation enabled with SynonymAug")
        except ImportError:
            print("Warning: nlpaug not available. Install with: pip install nlpaug")
            print("Continuing without data augmentation...")

    if training_strategy == "kfold":
        print(f"Using {k_folds}-fold cross-validation training...")

        fold_results, best_model_state = k_fold_training(
            text_pairs=text_pairs,
            labels=labels,
            tokenizer=tokenizer,
            model_name=model_name,
            pooling_strategy=pooling_strategy,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            k=k_folds,
            augmenter=augmenter,
        )

        model = SiameseTextClassifier(
            model_name=model_name, pooling_strategy=pooling_strategy
        )
        model.load_state_dict(best_model_state)

        best_checkpoint = torch.load(
            "transformers/models/best_kfold_model.pt", weights_only=False
        )
        best_val_accuracy = best_checkpoint["best_overall_acc"]

    else:
        print("Using standard train-validation split...")

        train_pairs, val_pairs, train_labels, val_labels = train_test_split(
            text_pairs, labels, test_size=0.2, random_state=42, stratify=labels
        )

        print(f"Training set: {len(train_pairs)} pairs")
        print(f"Validation set: {len(val_pairs)} pairs")

        train_dataset = TextTruthnessDataset(
            train_pairs, train_labels, tokenizer, max_length, augmenter=augmenter
        )
        val_dataset = TextTruthnessDataset(val_pairs, val_labels, tokenizer, max_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print(f"Initializing model: {model_name} with {pooling_strategy} pooling")
        model = SiameseTextClassifier(
            model_name=model_name, pooling_strategy=pooling_strategy
        )

        print("Starting training...")
        history = train_model(
            model, train_loader, val_loader, num_epochs, learning_rate
        )

        print("Loading best model for test predictions...")
        checkpoint = torch.load("transformers/models/best_model.pt", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_val_accuracy = checkpoint["val_accuracy"]

    print("Generating test predictions...")
    predictions = predict_on_test_set(model, test_data_dir, tokenizer)

    print("Training and prediction complete!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Generated predictions for {len(predictions)} test samples")
