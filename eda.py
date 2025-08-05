import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings


warnings.filterwarnings("ignore")


class TextDatasetEDA:
    """Exploratory Data Analysis class for the text classification dataset."""

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, "train")
        self.test_dir = os.path.join(data_dir, "test")
        self.train_csv = os.path.join(data_dir, "train.csv")

        self.df_train = None
        self.texts_data = []
        self.stop_words = set(stopwords.words("english"))

        os.makedirs("eda_plots", exist_ok=True)

    def load_data(self):
        """Load training data and text files."""
        print("Loading dataset...")

        self.df_train = pd.read_csv(self.train_csv)
        print(f"Training labels loaded: {len(self.df_train)} articles")

        for idx, row in self.df_train.iterrows():
            article_id = row["id"]
            real_text_id = row["real_text_id"]

            article_dir = os.path.join(self.train_dir, f"article_{article_id:04d}")

            with open(
                os.path.join(article_dir, "file_1.txt"), "r", encoding="utf-8"
            ) as f:
                text_1 = f.read().strip()
            with open(
                os.path.join(article_dir, "file_2.txt"), "r", encoding="utf-8"
            ) as f:
                text_2 = f.read().strip()

            if real_text_id == 1:
                real_text, fake_text = text_1, text_2
            else:
                real_text, fake_text = text_2, text_1

            self.texts_data.append(
                {
                    "article_id": article_id,
                    "real_text": real_text,
                    "fake_text": fake_text,
                    "real_text_id": real_text_id,
                }
            )

        print(f"Loaded {len(self.texts_data)} article pairs")

    def basic_statistics(self):
        """Generate basic dataset statistics."""
        print("\n" + "=" * 60)
        print("BASIC DATASET STATISTICS")
        print("=" * 60)

        print(f"Total training articles: {len(self.texts_data)}")
        print(f"Real text distribution:")
        real_text_dist = self.df_train["real_text_id"].value_counts().sort_index()
        for file_id, count in real_text_dist.items():
            print(
                f"  File {file_id}: {count} articles ({count / len(self.df_train) * 100:.1f}%)"
            )

        real_lengths = [len(item["real_text"]) for item in self.texts_data]
        fake_lengths = [len(item["fake_text"]) for item in self.texts_data]

        print(f"\nText Length Statistics (characters):")
        print(
            f"Real texts - Mean: {np.mean(real_lengths):.1f}, Std: {np.std(real_lengths):.1f}"
        )
        print(f"Real texts - Min: {min(real_lengths)}, Max: {max(real_lengths)}")
        print(
            f"Fake texts - Mean: {np.mean(fake_lengths):.1f}, Std: {np.std(fake_lengths):.1f}"
        )
        print(f"Fake texts - Min: {min(fake_lengths)}, Max: {max(fake_lengths)}")

        real_word_counts = [len(item["real_text"].split()) for item in self.texts_data]
        fake_word_counts = [len(item["fake_text"].split()) for item in self.texts_data]

        print(f"\nWord Count Statistics:")
        print(
            f"Real texts - Mean: {np.mean(real_word_counts):.1f}, Std: {np.std(real_word_counts):.1f}"
        )
        print(
            f"Fake texts - Mean: {np.mean(fake_word_counts):.1f}, Std: {np.std(fake_word_counts):.1f}"
        )

        return {
            "real_lengths": real_lengths,
            "fake_lengths": fake_lengths,
            "real_word_counts": real_word_counts,
            "fake_word_counts": fake_word_counts,
        }

    def plot_length_distributions(self, stats):
        """Plot text length distributions."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].hist(
            stats["real_lengths"], bins=30, alpha=0.7, label="Real", color="blue"
        )
        axes[0, 0].hist(
            stats["fake_lengths"], bins=30, alpha=0.7, label="Fake", color="red"
        )
        axes[0, 0].set_xlabel("Character Length")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Character Length Distribution")
        axes[0, 0].legend()

        axes[0, 1].hist(
            stats["real_word_counts"], bins=30, alpha=0.7, label="Real", color="blue"
        )
        axes[0, 1].hist(
            stats["fake_word_counts"], bins=30, alpha=0.7, label="Fake", color="red"
        )
        axes[0, 1].set_xlabel("Word Count")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Word Count Distribution")
        axes[0, 1].legend()

        axes[1, 0].boxplot(
            [stats["real_lengths"], stats["fake_lengths"]], labels=["Real", "Fake"]
        )
        axes[1, 0].set_ylabel("Character Length")
        axes[1, 0].set_title("Character Length Box Plot")

        axes[1, 1].boxplot(
            [stats["real_word_counts"], stats["fake_word_counts"]],
            labels=["Real", "Fake"],
        )
        axes[1, 1].set_ylabel("Word Count")
        axes[1, 1].set_title("Word Count Box Plot")

        plt.tight_layout()
        plt.savefig("eda_plots/length_distributions.png", dpi=300, bbox_inches="tight")
        plt.close()

    def analyze_vocabulary(self):
        """Analyze vocabulary characteristics."""
        print("\n" + "=" * 60)
        print("VOCABULARY ANALYSIS")
        print("=" * 60)

        real_texts = [item["real_text"] for item in self.texts_data]
        fake_texts = [item["fake_text"] for item in self.texts_data]

        real_words = []
        fake_words = []

        for text in real_texts:
            words = word_tokenize(text.lower())
            words = [
                word for word in words if word.isalpha() and word not in self.stop_words
            ]
            real_words.extend(words)

        for text in fake_texts:
            words = word_tokenize(text.lower())
            words = [
                word for word in words if word.isalpha() and word not in self.stop_words
            ]
            fake_words.extend(words)

        real_vocab = set(real_words)
        fake_vocab = set(fake_words)

        print(f"Real texts vocabulary size: {len(real_vocab)}")
        print(f"Fake texts vocabulary size: {len(fake_vocab)}")
        print(f"Overlap: {len(real_vocab.intersection(fake_vocab))} words")
        print(f"Real-only words: {len(real_vocab - fake_vocab)}")
        print(f"Fake-only words: {len(fake_vocab - real_vocab)}")

        real_word_freq = Counter(real_words)
        fake_word_freq = Counter(fake_words)

        print(f"\nTop 20 words in real texts:")
        for word, freq in real_word_freq.most_common(20):
            print(f"  {word}: {freq}")

        print(f"\nTop 20 words in fake texts:")
        for word, freq in fake_word_freq.most_common(20):
            print(f"  {word}: {freq}")

        return real_word_freq, fake_word_freq

    def generate_wordclouds(self, real_word_freq, fake_word_freq):
        """Generate word clouds for real and fake texts."""
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        real_wordcloud = WordCloud(
            width=800, height=400, background_color="white"
        ).generate_from_frequencies(real_word_freq)
        axes[0].imshow(real_wordcloud, interpolation="bilinear")
        axes[0].set_title("Real Texts Word Cloud", fontsize=16)
        axes[0].axis("off")

        fake_wordcloud = WordCloud(
            width=800, height=400, background_color="white"
        ).generate_from_frequencies(fake_word_freq)
        axes[1].imshow(fake_wordcloud, interpolation="bilinear")
        axes[1].set_title("Fake Texts Word Cloud", fontsize=16)
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig("eda_plots/wordclouds.png", dpi=300, bbox_inches="tight")
        plt.close()

    def analyze_ngrams(self, n=2):
        """Analyze n-grams in the texts."""
        print(f"\n" + "=" * 60)
        print(f"{n}-GRAM ANALYSIS")
        print("=" * 60)

        real_texts = " ".join([item["real_text"] for item in self.texts_data])
        fake_texts = " ".join([item["fake_text"] for item in self.texts_data])

        real_words = [
            word
            for word in word_tokenize(real_texts.lower())
            if word.isalpha() and word not in self.stop_words
        ]
        fake_words = [
            word
            for word in word_tokenize(fake_texts.lower())
            if word.isalpha() and word not in self.stop_words
        ]

        real_ngrams = list(ngrams(real_words, n))
        fake_ngrams = list(ngrams(fake_words, n))

        real_ngram_freq = Counter(real_ngrams)
        fake_ngram_freq = Counter(fake_ngrams)

        print(f"Top 15 {n}-grams in real texts:")
        for ngram, freq in real_ngram_freq.most_common(15):
            print(f"  {' '.join(ngram)}: {freq}")

        print(f"\nTop 15 {n}-grams in fake texts:")
        for ngram, freq in fake_ngram_freq.most_common(15):
            print(f"  {' '.join(ngram)}: {freq}")

    def analyze_sentence_structure(self):
        """Analyze sentence structure characteristics."""
        print("\n" + "=" * 60)
        print("SENTENCE STRUCTURE ANALYSIS")
        print("=" * 60)

        real_sent_lengths = []
        fake_sent_lengths = []
        real_sent_counts = []
        fake_sent_counts = []

        for item in self.texts_data:
            real_sentences = sent_tokenize(item["real_text"])
            real_sent_counts.append(len(real_sentences))
            for sent in real_sentences:
                real_sent_lengths.append(len(sent.split()))

            fake_sentences = sent_tokenize(item["fake_text"])
            fake_sent_counts.append(len(fake_sentences))
            for sent in fake_sentences:
                fake_sent_lengths.append(len(sent.split()))

        print(f"Sentence count per text:")
        print(
            f"Real texts - Mean: {np.mean(real_sent_counts):.1f}, Std: {np.std(real_sent_counts):.1f}"
        )
        print(
            f"Fake texts - Mean: {np.mean(fake_sent_counts):.1f}, Std: {np.std(fake_sent_counts):.1f}"
        )

        print(f"\nSentence length (words):")
        print(
            f"Real texts - Mean: {np.mean(real_sent_lengths):.1f}, Std: {np.std(real_sent_lengths):.1f}"
        )
        print(
            f"Fake texts - Mean: {np.mean(fake_sent_lengths):.1f}, Std: {np.std(fake_sent_lengths):.1f}"
        )

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].hist(real_sent_counts, bins=20, alpha=0.7, label="Real", color="blue")
        axes[0].hist(fake_sent_counts, bins=20, alpha=0.7, label="Fake", color="red")
        axes[0].set_xlabel("Sentences per Text")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Sentence Count Distribution")
        axes[0].legend()

        axes[1].hist(real_sent_lengths, bins=30, alpha=0.7, label="Real", color="blue")
        axes[1].hist(fake_sent_lengths, bins=30, alpha=0.7, label="Fake", color="red")
        axes[1].set_xlabel("Words per Sentence")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Sentence Length Distribution")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("eda_plots/sentence_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def tfidf_analysis(self):
        """Perform TF-IDF analysis and visualization."""
        print("\n" + "=" * 60)
        print("TF-IDF ANALYSIS")
        print("=" * 60)

        all_texts = []
        labels = []

        for item in self.texts_data:
            all_texts.append(item["real_text"])
            labels.append("real")
            all_texts.append(item["fake_text"])
            labels.append("fake")

        vectorizer = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 2), min_df=2
        )
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

        feature_names = vectorizer.get_feature_names_out()

        real_indices = [i for i, label in enumerate(labels) if label == "real"]
        fake_indices = [i for i, label in enumerate(labels) if label == "fake"]

        real_tfidf = tfidf_matrix[real_indices]
        fake_tfidf = tfidf_matrix[fake_indices]

        real_mean_tfidf = np.array(real_tfidf.mean(axis=0)).flatten()
        fake_mean_tfidf = np.array(fake_tfidf.mean(axis=0)).flatten()

        tfidf_diff = real_mean_tfidf - fake_mean_tfidf

        real_top_indices = np.argsort(tfidf_diff)[-20:][::-1]
        print("\nTop 20 features for REAL texts:")
        for idx in real_top_indices:
            print(f"  {feature_names[idx]}: {tfidf_diff[idx]:.4f}")

        fake_top_indices = np.argsort(tfidf_diff)[:20]
        print("\nTop 20 features for FAKE texts:")
        for idx in fake_top_indices:
            print(f"  {feature_names[idx]}: {tfidf_diff[idx]:.4f}")

        return tfidf_matrix, labels, feature_names

    def visualize_embeddings(self, tfidf_matrix, labels):
        """Visualize text embeddings using PCA and t-SNE."""
        print("\n" + "=" * 60)
        print("EMBEDDING VISUALIZATION")
        print("=" * 60)

        pca = PCA(n_components=2, random_state=42)
        pca_embeddings = pca.fit_transform(tfidf_matrix.toarray())

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_embeddings = tsne.fit_transform(tfidf_matrix.toarray())

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        real_mask = np.array(labels) == "real"
        fake_mask = np.array(labels) == "fake"

        axes[0].scatter(
            pca_embeddings[real_mask, 0],
            pca_embeddings[real_mask, 1],
            c="blue",
            alpha=0.6,
            label="Real",
            s=30,
        )
        axes[0].scatter(
            pca_embeddings[fake_mask, 0],
            pca_embeddings[fake_mask, 1],
            c="red",
            alpha=0.6,
            label="Fake",
            s=30,
        )
        axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        axes[0].set_title("PCA Visualization")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(
            tsne_embeddings[real_mask, 0],
            tsne_embeddings[real_mask, 1],
            c="blue",
            alpha=0.6,
            label="Real",
            s=30,
        )
        axes[1].scatter(
            tsne_embeddings[fake_mask, 0],
            tsne_embeddings[fake_mask, 1],
            c="red",
            alpha=0.6,
            label="Fake",
            s=30,
        )
        axes[1].set_xlabel("t-SNE Dimension 1")
        axes[1].set_ylabel("t-SNE Dimension 2")
        axes[1].set_title("t-SNE Visualization")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            "eda_plots/embeddings_visualization.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def run_complete_analysis(self):
        """Run the complete EDA pipeline."""
        print("EXPLORATORY DATA ANALYSIS")
        print("The Impostor Hunt in Texts Dataset")
        print("=" * 60)

        self.load_data()

        stats = self.basic_statistics()
        self.plot_length_distributions(stats)

        real_word_freq, fake_word_freq = self.analyze_vocabulary()
        self.generate_wordclouds(real_word_freq, fake_word_freq)

        self.analyze_ngrams(n=2)
        self.analyze_ngrams(n=3)

        self.analyze_sentence_structure()

        tfidf_matrix, labels, feature_names = self.tfidf_analysis()

        self.visualize_embeddings(tfidf_matrix, labels)

        print("\n" + "=" * 60)
        print("EDA COMPLETE!")
        print("Plots saved in 'eda_plots/' directory")
        print("=" * 60)


if __name__ == "__main__":
    eda = TextDatasetEDA("data/")
    eda.run_complete_analysis()
