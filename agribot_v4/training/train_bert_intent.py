"""
training/train_bert_intent.py
Full BERT Intent Classifier — NEW (bert-base-uncased, for comparison with DistilBERT)

The existing train_intent_classifier.py uses DistilBERT (66M params, fast).
This file trains full BERT (110M params, more accurate) on the SAME data
for a fair comparison.

Saves (separate — never overwrites existing DistilBERT):
  ../models/bert_intent_model/     ← full BERT weights
  ../models/comparison_nlp_distilbert_vs_bert.json

Also produces: per-intent precision, recall, F1, confusion matrix.
"""

import os, json, warnings, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score)
warnings.filterwarnings("ignore")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
BERT_DIR  = os.path.join(MODEL_DIR, "bert_intent_model")
DBERT_DIR = os.path.join(MODEL_DIR, "intent_model")  # existing DistilBERT
os.makedirs(BERT_DIR, exist_ok=True)

# ── Same training data as existing DistilBERT ─────────────────────────────────
INTENT_LABELS = {
    "crop_recommendation": 0,
    "fertilizer_advice":   1,
    "crop_disease":        2,
    "weather_planting":    3,
}
ID2LABEL = {v: k for k, v in INTENT_LABELS.items()}

# Identical training data to train_intent_classifier.py for fair comparison
TRAINING_DATA = [
    # crop_recommendation (0)
    ("What crop should I grow with nitrogen 90, phosphorus 42, potassium 43?", 0),
    ("Recommend a crop for my soil with pH 6.5 and rainfall 200mm.", 0),
    ("Which crop is best for sandy loam soil with high humidity?", 0),
    ("My soil has N=70, P=30, K=50, temperature 25°C. What should I plant?", 0),
    ("Suggest a suitable crop for my farm.", 0),
    ("What crop grows best in high humidity and moderate rainfall?", 0),
    ("Can I grow potatoes in sandy soil?", 0),
    ("What crops do well in dry areas with low rainfall?", 0),
    ("I have a small farm with clay soil. What should I plant?", 0),
    ("Which crop is best for black cotton soil?", 0),
    ("What should I plant in my field this season?", 0),
    ("Which crops are suitable for loamy soil with high potassium?", 0),
    ("What is a good crop for acidic soil?", 0),
    ("My field has red soil, what can I grow there?", 0),
    ("Can I grow rice in my area where it rains a lot?", 0),
    ("What crops can grow with very little water?", 0),
    ("Which crop is profitable for a small farm?", 0),
    ("Can maize grow in my soil with pH 6.2?", 0),
    ("Can I grow banana on my farm?", 0),
    ("What crop should I plant after harvesting rice?", 0),
    ("What can I grow in hot weather with sandy soil?", 0),
    ("Best crop for waterlogged fields?", 0),
    ("What grows well in low nitrogen red soil?", 0),
    ("Suggest drought-resistant crops for my field.", 0),
    ("My land is near a river with good water supply. What to plant?", 0),
    ("What crop is suitable for new cultivated land?", 0),
    ("N=80, P=40, K=60, temp=28, humidity=75, pH=6.5 – recommend a crop", 0),
    ("Which vegetable grows in hot weather?", 0),
    ("I want to grow a cash crop. What do you recommend?", 0),
    ("What is the best crop for tropical weather?", 0),

    # fertilizer_advice (1)
    ("What fertilizer should I use for my wheat crop on sandy soil?", 1),
    ("My maize has low nitrogen, which fertilizer do you recommend?", 1),
    ("Recommend a fertilizer for rice in clay soil.", 1),
    ("Which fertilizer is best for ground nuts with phosphorus deficiency?", 1),
    ("How much urea should I apply for my sugarcane field?", 1),
    ("What NPK ratio fertilizer fits cotton on red soil?", 1),
    ("Which fertilizer is best for flowering plants?", 1),
    ("What fertilizer helps plants grow faster?", 1),
    ("My plants are not growing well. Should I add more fertilizer?", 1),
    ("What fertilizer do I use for vegetable garden?", 1),
    ("My crop leaves are pale yellow. Do I need nitrogen fertilizer?", 1),
    ("What fertilizer is good for fruit trees?", 1),
    ("Which fertilizer increases yield for barley?", 1),
    ("Is DAP good for my wheat crop?", 1),
    ("Should I use urea or DAP for my rice crop?", 1),
    ("My soil has low potassium. What fertilizer should I add?", 1),
    ("I want to use organic fertilizer. What is best for my farm?", 1),
    ("What fertilizer helps root growth?", 1),
    ("My plants look weak and pale. Which fertilizer can help?", 1),
    ("What is the right time to apply fertilizer to my crops?", 1),
    ("How many kg of fertilizer per acre for wheat?", 1),
    ("Should I fertilize before or after rain?", 1),
    ("What fertilizer dosage for rice?", 1),
    ("How much urea per acre for maize?", 1),
    ("What is urea and when to use it?", 1),
    ("My cotton crop is yellowing. Which fertilizer should I use?", 1),
    ("What fertilizer for sugarcane to increase sugar content?", 1),
    ("How to improve soil fertility before planting?", 1),
    ("Fertilizer recommendation for black soil with maize.", 1),
    ("What is the dose of DAP for cotton per acre?", 1),

    # crop_disease (2)
    ("My rice leaves are turning yellow, what disease could this be?", 2),
    ("There are brown spots on my wheat leaves, how do I treat them?", 2),
    ("My tomato plants have black lesions on the stem.", 2),
    ("The maize leaves have white powdery coating, what is wrong?", 2),
    ("My potato plants are wilting and the roots look rotten.", 2),
    ("Fungal infection is spreading on my sugarcane.", 2),
    ("My mango leaves have spots and are falling off early.", 2),
    ("The cotton plants show mosaic pattern on leaves.", 2),
    ("Leaf blight is affecting my rice.", 2),
    ("How do I control aphid infestation on my cabbage?", 2),
    ("My wheat shows rust symptoms.", 2),
    ("There are insects eating my crop leaves, how to control?", 2),
    ("My banana plants have black spots on the leaves.", 2),
    ("My plants look sick. The leaves are drooping and yellow.", 2),
    ("Something is eating my crops at night. What pest is it?", 2),
    ("White insects on the bottom of my leaves. What are they?", 2),
    ("Small holes in leaves and the crop is not growing. What disease?", 2),
    ("My rice crop has orange colored dust on the leaves.", 2),
    ("The tip of my maize leaves are drying up. Is it a disease?", 2),
    ("There is a sticky substance on my fruit tree leaves.", 2),
    ("Worms are destroying my crop from the roots.", 2),
    ("I see small black bugs on my crop. What are they?", 2),
    ("My tomato fruits have black rot on them.", 2),
    ("Strange marks on my crop leaves. Is it a disease or pest?", 2),
    ("After rain, my crop started getting spots. What is it?", 2),
    ("My sugarcane is rotting at the base.", 2),
    ("I see white mold on my vegetable crop after rain.", 2),
    ("Identify and treat the yellow streaks on my maize.", 2),
    ("My crop has leaf curl virus symptoms.", 2),
    ("There is a fungus spreading on my wheat field.", 2),

    # weather_planting (3)
    ("Is it a good time to plant maize given the current weather?", 3),
    ("The temperature is 30°C and humidity is high, can I sow rice now?", 3),
    ("When is the best planting season for wheat in dry climate?", 3),
    ("Should I delay planting because of expected heavy rainfall?", 3),
    ("Is frost risk high this season?", 3),
    ("Is this a good week to transplant seedlings?", 3),
    ("The rainy season has started, which crops should I sow first?", 3),
    ("Is there drought risk this season?", 3),
    ("It has been very hot lately. Is it safe to plant now?", 3),
    ("It rained heavily yesterday. Should I wait before planting?", 3),
    ("The weather has been cloudy and cold. Can I plant rice?", 3),
    ("When should I start sowing seeds this year?", 3),
    ("The monsoon is late. Should I change my planting schedule?", 3),
    ("Is it too hot to plant right now?", 3),
    ("The weather is unstable this season. Should I wait to plant?", 3),
    ("My area has not received rain in weeks. Can I still plant?", 3),
    ("The nights are getting cold. Will my seedlings survive?", 3),
    ("We had too much rain this month. Is planting still possible?", 3),
    ("Is this the right season to grow watermelon?", 3),
    ("Should I plant before or after the rainy season?", 3),
    ("When is the best time to sow maize?", 3),
    ("The weather forecast says heavy rain next week. Should I plant now?", 3),
    ("Is the current humidity level good for planting paddy?", 3),
    ("The temperature has dropped suddenly. Will my crops be affected?", 3),
    ("After the dry season, when should I plant rice?", 3),
    ("Is it good weather today to plant?", 3),
    ("Should I plant now or wait for the weather to improve?", 3),
    ("What crops survive in a heatwave?", 3),
    ("Frost last night — should I protect my seedlings?", 3),
    ("30°C and 75% humidity — can I grow maize?", 3),
]


import random
def augment(sentences, labels, factor=4):
    out_s, out_l = list(sentences), list(labels)
    for s, l in zip(sentences, labels):
        words = s.split()
        if len(words) > 4:
            for _ in range(factor):
                w2 = words[:]
                random.shuffle(w2)
                out_s.append(" ".join(w2))
                out_l.append(l)
    return out_s, out_l


class IntentDataset(Dataset):
    def __init__(self, enc, labels):
        self.enc = enc
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch   = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds   = outputs.logits.argmax(-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    return np.array(all_preds), np.array(all_labels)


def compute_full_metrics(y_true, y_pred, label_names):
    return {
        "accuracy":           round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro":    round(float(precision_score(y_true, y_pred, average="macro",    zero_division=0)), 4),
        "recall_macro":       round(float(recall_score(y_true,    y_pred, average="macro",    zero_division=0)), 4),
        "f1_macro":           round(float(f1_score(y_true,        y_pred, average="macro",    zero_division=0)), 4),
        "precision_weighted": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall_weighted":    round(float(recall_score(y_true,    y_pred, average="weighted", zero_division=0)), 4),
        "f1_weighted":        round(float(f1_score(y_true,        y_pred, average="weighted", zero_division=0)), 4),
        "confusion_matrix":   confusion_matrix(y_true, y_pred).tolist(),
    }


def train_model(model_name: str, model_checkpoint: str, sents, labels,
                device, epochs=6, batch_size=16, lr=2e-5):
    """Generic train loop for BERT / DistilBERT."""
    from transformers import (AutoTokenizer,
                              AutoModelForSequenceClassification,
                              get_linear_schedule_with_warmup)
    from torch.optim import AdamW

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=4,
        id2label=ID2LABEL, label2id=INTENT_LABELS)
    model.to(device)

    tr_s, vl_s, tr_l, vl_l = train_test_split(
        sents, labels, test_size=0.15, random_state=42, stratify=labels)

    def tok(s): return tokenizer(s, truncation=True, padding=True, max_length=128)
    tr_dl = DataLoader(IntentDataset(tok(tr_s), tr_l), batch_size=batch_size, shuffle=True)
    vl_dl = DataLoader(IntentDataset(tok(vl_s), vl_l), batch_size=32)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total     = len(tr_dl) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                    num_warmup_steps=len(tr_dl),
                    num_training_steps=total)

    best_acc = 0.0
    best_state = None
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in tr_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += out.loss.item()

        preds, true = evaluate_model(model, vl_dl, device)
        val_acc = accuracy_score(true, preds)
        print(f"  [{model_name}] Epoch {epoch}/{epochs} | "
              f"loss={total_loss/len(tr_dl):.4f} | val_acc={val_acc:.4f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = round(time.time() - t_start, 1)
    model.load_state_dict(best_state)
    final_preds, final_true = evaluate_model(model, vl_dl, device)
    metrics = compute_full_metrics(final_true, final_preds, list(INTENT_LABELS.keys()))
    metrics["train_time_sec"] = train_time
    metrics["best_val_acc"]   = best_acc

    return model, tokenizer, metrics


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device: {device}")

    # Prepare data
    sents  = [x[0] for x in TRAINING_DATA]
    labels = [x[1] for x in TRAINING_DATA]
    sents, labels = augment(sents, labels, factor=4)
    print(f"[INFO] Total samples after augmentation: {len(sents)}")

    label_names = list(INTENT_LABELS.keys())

    # ── Train full BERT ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training: bert-base-uncased (Full BERT, 110M params)")
    print("=" * 60)
    bert_model, bert_tok, bert_metrics = train_model(
        "BERT", "bert-base-uncased",
        sents, labels, device, epochs=6, lr=2e-5)

    # Save BERT
    bert_model.save_pretrained(BERT_DIR)
    bert_tok.save_pretrained(BERT_DIR)
    with open(os.path.join(BERT_DIR, "intent_labels.json"), "w") as f:
        json.dump({"id2label": ID2LABEL, "label2id": INTENT_LABELS}, f, indent=2)
    print(f"[BERT] Saved to {BERT_DIR}")

    # ── Train DistilBERT for comparison on same data ──────────────────────────
    print("\n" + "=" * 60)
    print("  Training: distilbert-base-uncased (DistilBERT, 66M params)")
    print("=" * 60)
    _, _, dbert_metrics = train_model(
        "DistilBERT", "distilbert-base-uncased",
        sents, labels, device, epochs=6, lr=2e-5)

    # ── Build comparison ──────────────────────────────────────────────────────
    winner = "BERT" if bert_metrics["accuracy"] >= dbert_metrics["accuracy"] else "DistilBERT"

    comparison = {
        "task":        "Intent Classification (NLP)",
        "dataset":     f"{len(TRAINING_DATA)} base samples × 5x augmentation = {len(sents)} total",
        "intents":     label_names,
        "val_split":   "85/15 stratified",
        "winner":      winner,
        "models": {
            "DistilBERT": {
                **dbert_metrics,
                "params":    "66M",
                "checkpoint":"distilbert-base-uncased",
                "notes":     "Existing production model — 40% smaller than BERT",
            },
            "BERT": {
                **bert_metrics,
                "params":    "110M",
                "checkpoint":"bert-base-uncased",
                "notes":     "Full BERT — higher accuracy, slower inference",
            },
        },
        "slide_table": [
            ["Metric",           "DistilBERT", "BERT (Full)", "Winner"],
            ["Accuracy",
             f"{dbert_metrics['accuracy']:.4f}",
             f"{bert_metrics['accuracy']:.4f}",
             "BERT" if bert_metrics["accuracy"] > dbert_metrics["accuracy"] else "DistilBERT"],
            ["Precision (macro)",
             f"{dbert_metrics['precision_macro']:.4f}",
             f"{bert_metrics['precision_macro']:.4f}", "—"],
            ["Recall (macro)",
             f"{dbert_metrics['recall_macro']:.4f}",
             f"{bert_metrics['recall_macro']:.4f}", "—"],
            ["F1-Score (macro)",
             f"{dbert_metrics['f1_macro']:.4f}",
             f"{bert_metrics['f1_macro']:.4f}",
             "BERT" if bert_metrics["f1_macro"] > dbert_metrics["f1_macro"] else "DistilBERT"],
            ["Train Time (s)",
             f"{dbert_metrics['train_time_sec']}",
             f"{bert_metrics['train_time_sec']}",
             "DistilBERT (faster)"],
            ["Model Size",  "66M params", "110M params", "DistilBERT (smaller)"],
        ],
    }

    comp_path = os.path.join(MODEL_DIR, "comparison_nlp_distilbert_vs_bert.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  NLP MODEL COMPARISON — DistilBERT vs Full BERT")
    print("=" * 60)
    print(f"{'Metric':<22} {'DistilBERT':>12} {'BERT':>10} {'Winner':>14}")
    print("-" * 60)
    for row in comparison["slide_table"][1:]:
        print(f"{row[0]:<22} {row[1]:>12} {row[2]:>10} {row[3]:>14}")
    print(f"\n🏆 Overall Winner: {winner}")
    print(f"Confusion Matrix (BERT): {bert_metrics['confusion_matrix']}")
    print(f"Comparison saved → {comp_path}")

    return bert_metrics, dbert_metrics


if __name__ == "__main__":
    train()
