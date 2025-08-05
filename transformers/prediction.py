from typing import Any

import torch
from transformers import AutoTokenizer
from siamese_transformer_trainer import SiameseTextClassifier, predict_on_test_set


def load_best_kfold_model(
    model_path="transformers/models/best_kfold_model.pt",
    model_name="bert-base-uncased",
    pooling_strategy="cls",
) -> tuple[SiameseTextClassifier, Any]:
    """Load the best k-fold model for predictions."""
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Loading model from {model_path}")
    print(f"Using device: {device}")

    checkpoint = torch.load(model_path, weights_only=False, map_location=device)

    model = SiameseTextClassifier(
        model_name=model_name, pooling_strategy=pooling_strategy
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    best_acc = checkpoint.get("best_overall_acc", "Unknown")
    cv_mean = checkpoint.get("cv_mean", "Unknown")
    cv_std = checkpoint.get("cv_std", "Unknown")

    print(f"Model loaded successfully!")
    print(f"Best overall accuracy: {best_acc}")
    print(f"Cross-validation mean: {cv_mean}")
    print(f"Cross-validation std: {cv_std}")

    return model, checkpoint


def main():
    """Main prediction function."""

    model_name = "bert-base-uncased"
    pooling_strategy = "cls"

    test_data_dir = "data/test"
    model_path = "transformers/models/best_kfold_model.pt"

    print("=== Starting Test Predictions ===")

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model, checkpoint = load_best_kfold_model(
        model_path=model_path, model_name=model_name, pooling_strategy=pooling_strategy
    )

    print("Generating predictions on test set...")
    predictions_df = predict_on_test_set(
        model=model,
        test_data_dir=test_data_dir,
        tokenizer=tokenizer,
        predictions_dir="transformers/predictions",
    )

    print("\n=== Prediction Summary ===")
    print(f"Total test samples processed: {len(predictions_df)}")
    print(f"Predictions saved to transformers/predictions/")

    real_text_1_count = (predictions_df["real_text_id"] == 1).sum()
    real_text_2_count = (predictions_df["real_text_id"] == 2).sum()

    print(
        f"Predicted as real_text_id=1: {real_text_1_count} ({real_text_1_count / len(predictions_df) * 100:.1f}%)"
    )
    print(
        f"Predicted as real_text_id=2: {real_text_2_count} ({real_text_2_count / len(predictions_df) * 100:.1f}%)"
    )

    avg_confidence = predictions_df["probability_text_1_true"].mean()
    median_confidence = predictions_df["probability_text_1_true"].median()
    print(
        f"Average confidence (prob_text_1_true): {avg_confidence:.4f}\nMedian confidence: {median_confidence:.4f}"
    )

    return predictions_df


if __name__ == "__main__":
    predictions = main()
