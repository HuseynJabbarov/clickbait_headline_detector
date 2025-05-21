from __future__ import annotations

import math
import sys
from pathlib import Path

import joblib

try:
    from train import LexicalSyntacticTransformer

    sys.modules["__main__"].LexicalSyntacticTransformer = LexicalSyntacticTransformer
except ModuleNotFoundError:
    pass


def sigmoid(x: float) -> float:
    x = max(min(x, 20), -20)
    return 1.0 / (1.0 + math.exp(-x))


def load_model(path: Path):
    """Load a joblib dump. Return (pipeline, held‑out accuracy|None)."""
    bundle = joblib.load(path)
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle["model"], bundle.get("accuracy")
    return bundle, None


def main():
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/svm_clickbait.joblib")
    if not model_path.exists():
        sys.exit(f"Model file not found: {model_path}")

    model, acc = load_model(model_path)

    print("\n🔮 Model loaded from", model_path)
    if acc is not None:
        print(f"Held‑out accuracy: {acc:.2%}\n")
    print("Type a headline and press Enter (write 'exit' to quit).\n")

    while True:
        try:
            headline = input("» ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if headline.lower() == "exit":
            print("Goodbye!")
            break
        if not headline:
            continue

        pred_int = int(model.predict([headline])[0])
        label = "CLICKBAIT" if pred_int == 1 else "not clickbait"
        print(f"{label}")


if __name__ == "__main__":
    main()
