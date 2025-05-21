import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


class LexicalSyntacticTransformer(BaseEstimator, TransformerMixin):
    _superlatives = {
        "best", "worst", "greatest", "biggest", "smallest",
        "most", "least", "cutest", "craziest", "amazing",
        "ultimate", "shocking", "unbelievable", "incredible",
    }
    _stop_punct = set("?!")

    def fit(self, X, y=None):
        self._dv = DictVectorizer(sparse=True)
        dummy = self._transform_row("sample headline")
        self._dv.fit([dummy])
        return self

    def transform(self, X):
        data = [self._transform_row(text) for text in X]
        return self._dv.transform(data)

    def _transform_row(self, text: str) -> dict:
        tokens = text.lower().split()
        num_tokens = len(tokens)
        features = {
            "length_chars": len(text),
            "length_words": num_tokens,
            "num_exclaim": text.count("!"),
            "num_question": text.count("?"),
            "num_quotes": text.count("\""),
            "starts_with_number": int(tokens[0].isdigit()) if tokens else 0,
            "all_caps_words": sum(1 for t in tokens if t.isupper() and len(t) > 1),
            "superlative_count": sum(1 for t in tokens if t in self._superlatives),
        }

        features["exclaim_ratio"] = features["num_exclaim"] / max(1, len(text))
        features["question_ratio"] = features["num_question"] / max(1, len(text))
        return features


def train_svm(headlines: pd.Series, labels: pd.Series, output_dir: Path):
    print("→ Training TF‑IDF + lexical SVM…")
    tfidf = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    lexical = LexicalSyntacticTransformer()
    features = FeatureUnion([("tfidf", tfidf), ("lex", lexical)])

    clf = LinearSVC(class_weight="balanced", C=1.0)

    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("features", features), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        headlines, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print(classification_report(y_test, preds, digits=3))

    output_dir.mkdir(exist_ok=True)
    joblib.dump(pipe, output_dir / "svm_clickbait.joblib")
    print(f"✔ Saved model to {output_dir / 'svm_clickbait.joblib'}")


def train_bert(headlines: pd.Series, labels: pd.Series, output_dir: Path, epochs=3):
    try:
        from datasets import Dataset, ClassLabel
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    except ImportError:
        raise SystemExit("Install 'datasets' and 'transformers' to use --model bert")

    print("→ Fine‑tuning BERT… (this may take a while)")

    label_list = sorted(labels.unique())
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    labels_num = labels.map(label2id)

    ds = Dataset.from_dict({"text": headlines, "label": labels_num})
    ds = ds.train_test_split(test_size=0.2, stratify_by_column="label", seed=42)

    checkpoint = "bert-base-uncased"
    tok = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=64)

    ds = ds.map(tokenize, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=len(label_list), id2label=id2label, label2id=label2id
    )

    args = TrainingArguments(
        output_dir=str(output_dir / "bert_runs"),
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        weight_decay=0.01,
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
    )

    def compute_metrics(eval_pred):
        import evaluate
        metric = evaluate.load("f1")
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return metric.compute(predictions=preds, references=labels, average="binary")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "bert_clickbait"))

    print("✔ BERT model saved to", output_dir / "bert_clickbait")


# ----------------------------- CLI ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Clickbait headline detector")
    ap.add_argument("--data", default="clickbait_data.csv", help="Path to CSV dataset")
    ap.add_argument("--model", choices=["svm", "bert"], default="svm")
    ap.add_argument("--epochs", type=int, default=3, help="BERT training epochs")
    ap.add_argument("--out", default="models", help="Output directory")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    text_col = "headline" if "headline" in df.columns else "text"
    label_col = "clickbait" if "clickbait" in df.columns else "label"

    headlines = df[text_col].astype(str)
    labels = df[label_col].astype(int)

    output_dir = Path(args.out)
    if args.model == "svm":
        train_svm(headlines, labels, output_dir)
    else:
        train_bert(headlines, labels, output_dir, args.epochs)


if __name__ == "__main__":
    main()
