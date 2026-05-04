"""
training/train_intent_classifier.py  — v2 (Data-Driven, Natural Language)

Covers natural, informal farmer queries in all 4 intents.
Much larger and more diverse training set.
"""

import os, json, random, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from transformers import (DistilBertTokenizerFast,
                          DistilBertForSequenceClassification,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings; warnings.filterwarnings("ignore")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "intent_model")
os.makedirs(MODEL_DIR, exist_ok=True)

INTENT_LABELS = {"crop_recommendation":0, "fertilizer_advice":1, "crop_disease":2, "weather_planting":3}
ID2LABEL = {v: k for k, v in INTENT_LABELS.items()}

TRAINING_DATA = [
    # ── 0: crop_recommendation ──────────────────────────────────────────────
    ("What crop should I grow with nitrogen 90, phosphorus 42, potassium 43?", 0),
    ("Recommend a crop for my soil with pH 6.5 and rainfall 200mm.", 0),
    ("Which crop is best for sandy loam soil with high humidity?", 0),
    ("My soil has N=70, P=30, K=50, temperature 25°C. What should I plant?", 0),
    ("Suggest a suitable crop for my farm.", 0),
    ("What crop grows best in high humidity and moderate rainfall?", 0),
    ("My soil pH is 7.2 with low nitrogen, which crop fits best?", 0),
    ("Can I grow potatoes in sandy soil?", 0),
    ("Tell me the best crop for tropical climate with clay soil.", 0),
    ("Which crops are suitable for loamy soil with high potassium?", 0),
    ("What should I plant in my field this season?", 0),
    ("What is a good crop for acidic soil?", 0),
    ("My field has red soil, what can I grow there?", 0),
    ("Can I grow rice in my area where it rains a lot?", 0),
    ("What crops do well in dry areas with low rainfall?", 0),
    ("I have a small farm with clay soil. What should I plant?", 0),
    ("Which crop is best for black cotton soil?", 0),
    ("My land gets flooded sometimes. What crop should I choose?", 0),
    ("I want to plant something in the rainy season. What do you suggest?", 0),
    ("What crops can grow with very little water?", 0),
    ("I have sandy soil and little rain. What can I grow?", 0),
    ("Which crop is profitable for a small farm?", 0),
    ("My soil test shows low phosphorus. What crop can still grow well?", 0),
    ("What crop is best suited for my loamy soil?", 0),
    ("I want to grow something that does not need much fertilizer.", 0),
    ("Which vegetables can grow in hot weather?", 0),
    ("I want to grow a cash crop. What do you recommend for my soil?", 0),
    ("My field stays wet most of the time. What crop is suitable?", 0),
    ("What crop should I try for the first time on new land?", 0),
    ("My farm is near a river and gets good water. What to plant?", 0),
    ("I want to grow something easy to sell. Suggest a crop.", 0),
    ("Can maize grow in my soil with pH 6.2?", 0),
    ("What crop gives the highest yield in tropical climate?", 0),
    ("I have 2 acres of land. What crop is best for me?", 0),
    ("Can I grow banana on my farm?", 0),
    ("Is mango good for my soil type?", 0),
    ("Which crop is best for high nitrogen soil?", 0),
    ("I want to do mixed farming. What crops go well together?", 0),
    ("My neighbor grows wheat. Can I also grow wheat here?", 0),
    ("What crop should I plant after harvesting rice?", 0),
    ("My soil is dark and heavy. What grows well?", 0),
    ("What can I grow in my garden at home?", 0),
    ("I want to grow food for my family. What is the easiest crop?", 0),
    ("The soil in my area is very hard. What crop can break it?", 0),
    ("What crop should I plant before the dry season?", 0),

    # ── 1: fertilizer_advice ────────────────────────────────────────────────
    ("What fertilizer should I use for my wheat crop on sandy soil?", 1),
    ("My maize has low nitrogen, which fertilizer do you recommend?", 1),
    ("Recommend a fertilizer for rice in clay soil.", 1),
    ("Which fertilizer is best for ground nuts with phosphorus deficiency?", 1),
    ("How much urea should I apply for my sugarcane field?", 1),
    ("What NPK ratio fertilizer fits cotton on red soil?", 1),
    ("I need fertilizer advice for my tomato crop.", 1),
    ("My soil potassium is very low, what fertilizer should I apply?", 1),
    ("Which fertilizer increases yield for barley?", 1),
    ("My crop needs phosphorus, what fertilizer should I buy?", 1),
    ("What is the best fertilizer for paddy rice in humid conditions?", 1),
    ("I want to maximize wheat yield, what fertilizer should I use?", 1),
    ("Which fertilizer should I apply before sowing cotton seeds?", 1),
    ("Which fertilizer is best for flowering plants?", 1),
    ("What fertilizer helps plants grow faster?", 1),
    ("My plants are not growing well. Should I add more fertilizer?", 1),
    ("What fertilizer do I use for vegetable garden?", 1),
    ("How do I improve soil fertility before planting?", 1),
    ("My crop leaves are pale yellow. Do I need nitrogen fertilizer?", 1),
    ("What fertilizer is good for fruit trees?", 1),
    ("I want to apply fertilizer to my paddy field. What do you suggest?", 1),
    ("My maize is not growing well. Is it a fertilizer problem?", 1),
    ("Which fertilizer is best for increasing grain size?", 1),
    ("What fertilizer should I apply during flowering stage?", 1),
    ("Is DAP good for my wheat crop?", 1),
    ("Should I use urea or DAP for my rice crop?", 1),
    ("What is the right time to apply fertilizer to my crops?", 1),
    ("My soil has low potassium. What fertilizer should I add?", 1),
    ("I want to use organic fertilizer. What is best for my farm?", 1),
    ("How many times should I apply fertilizer to maize?", 1),
    ("What fertilizer helps root growth?", 1),
    ("My plants look weak and pale. Which fertilizer can help?", 1),
    ("I have black soil with low nitrogen. What fertilizer to apply?", 1),
    ("What fertilizer is best for sugarcane in loamy soil?", 1),
    ("Should I fertilize before or after rain?", 1),
    ("My cotton crop is yellowing. Which fertilizer should I use?", 1),
    ("What is urea and when should I use it?", 1),
    ("How much fertilizer per acre for wheat?", 1),
    ("Which fertilizer is cheapest and most effective for small farms?", 1),
    ("I want to grow more tomatoes. What fertilizer should I give?", 1),
    ("My crop is not producing good fruit. Will fertilizer help?", 1),
    ("Which fertilizer is safe to use without harming the soil?", 1),
    ("What are the signs that my crop needs more fertilizer?", 1),
    ("I heard too much fertilizer can damage crops. What is the right amount?", 1),
    ("My paddy crop looks thin. What fertilizer will make it stronger?", 1),

    # ── 2: crop_disease ─────────────────────────────────────────────────────
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
    ("My crop is dying even after watering. What is wrong?", 2),
    ("White insects on the bottom of my leaves. What are they?", 2),
    ("Small holes in leaves and the crop is not growing. What disease?", 2),
    ("My rice crop has orange colored dust on the leaves.", 2),
    ("The tip of my maize leaves are drying up. Is it a disease?", 2),
    ("My crop looks healthy but then suddenly falls over and dies.", 2),
    ("There is a sticky substance on my fruit tree leaves.", 2),
    ("Worms are destroying my crop from the roots.", 2),
    ("My onion leaves are falling over and the bulbs are soft.", 2),
    ("I see small black bugs on my crop. What are they?", 2),
    ("My tomato fruits have black rot on them.", 2),
    ("The crop was fine last week but now looks dead. What happened?", 2),
    ("My paddy has a bad smell and the plants are dying.", 2),
    ("Strange marks on my crop leaves. Is it a disease or pest?", 2),
    ("My leaves have yellow and green patches. What is the problem?", 2),
    ("The flowers on my crop are falling off without fruit forming.", 2),
    ("After rain, my crop started getting spots. What is it?", 2),
    ("My sugarcane is rotting at the base.", 2),
    ("I sprayed pesticide but the insects are still there.", 2),
    ("My crop has small round holes in the leaves.", 2),
    ("The new leaves of my plant are twisted and small.", 2),
    ("My crops are getting damaged. What disease is spreading?", 2),
    ("I see white mold on my vegetable crop after rain.", 2),
    ("My crop is not producing fruits even though it has flowers.", 2),
    ("After the flood my crops started dying. What disease is this?", 2),
    ("My plant stems are turning black from the bottom.", 2),
    ("There are small insects jumping on my crop leaves.", 2),
    ("My rice grains are empty when harvested. What is the cause?", 2),
    ("The fruit from my trees is rotting before it ripens.", 2),
    ("My crop has a disease that spreads quickly to other plants.", 2),

    # ── 3: weather_planting ─────────────────────────────────────────────────
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
    ("Is it okay to plant after a big storm passes?", 3),
    ("The nights are getting cold. Will my seedlings survive?", 3),
    ("There was frost last night. Should I protect my seedlings?", 3),
    ("We had too much rain this month. Is planting still possible?", 3),
    ("The sky has been overcast for many days. Good time to plant?", 3),
    ("It is very dry this month. What should I plant?", 3),
    ("Is this the right season to grow watermelon?", 3),
    ("The ground is still wet from last week's rain. Can I plant now?", 3),
    ("Should I plant before or after the rainy season?", 3),
    ("It is getting cold at night but warm in the day. What can I grow?", 3),
    ("My region has strong winds now. Is it safe to plant?", 3),
    ("When is the best time to sow maize?", 3),
    ("The weather forecast says heavy rain next week. Should I plant now?", 3),
    ("My area has summer heat right now. Which crops survive this?", 3),
    ("Is the current humidity level good for planting paddy?", 3),
    ("We have had a dry spell. When will be the right time to plant?", 3),
    ("The temperature has dropped suddenly. Will my crops be affected?", 3),
    ("Is early morning planting better in hot weather?", 3),
    ("My region gets rain only 3 months a year. When should I plant?", 3),
    ("The harvest from last crop is done. Can I plant again immediately?", 3),
    ("It is good weather today. Should I start planting?", 3),
    ("After the dry season, when should I plant rice?", 3),
    ("How do I know if the weather is right for planting?", 3),
    ("Is it too late in the season to plant cotton?", 3),
    ("The sky looks like it will rain soon. Should I plant now or wait?", 3),
    ("My seedlings are ready. Is the weather suitable to transplant them?", 3),
    ("Is it good to plant during a heatwave?", 3),
    ("My area gets very cold in December. When should I plant wheat?", 3),
    ("The season is ending soon. Is it too late to plant anything?", 3),
]


def augment(sents, labels, factor=4):
    out_s, out_l = list(sents), list(labels)
    for s, l in zip(sents, labels):
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
        self.enc = enc; self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    sents  = [x[0] for x in TRAINING_DATA]
    labels = [x[1] for x in TRAINING_DATA]
    sents, labels = augment(sents, labels, factor=4)
    print(f"[INFO] Total samples: {len(sents)}")

    tr_s, vl_s, tr_l, vl_l = train_test_split(
        sents, labels, test_size=0.15, random_state=42, stratify=labels)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=4, id2label=ID2LABEL, label2id=INTENT_LABELS)
    model.to(device)

    def tok(s): return tokenizer(s, truncation=True, padding=True, max_length=128)
    tr_dl = DataLoader(IntentDataset(tok(tr_s), tr_l), batch_size=16, shuffle=True)
    vl_dl = DataLoader(IntentDataset(tok(vl_s), vl_l), batch_size=32)

    EPOCHS = 8
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=len(tr_dl),
        num_training_steps=len(tr_dl)*EPOCHS)

    best_acc = 0.0
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = correct = total = 0
        for batch in tr_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += out.loss.item()
            preds = out.logits.argmax(-1)
            correct += (preds==batch["labels"]).sum().item()
            total   += len(batch["labels"])

        model.eval(); vp, vl2 = [], []
        with torch.no_grad():
            for batch in vl_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                vp.extend(model(**batch).logits.argmax(-1).cpu().numpy())
                vl2.extend(batch["labels"].cpu().numpy())
        val_acc = np.mean(np.array(vp)==np.array(vl2))
        print(f"[Epoch {epoch}/{EPOCHS}] loss={total_loss/len(tr_dl):.4f} "
              f"train={correct/total:.4f} val={val_acc:.4f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"  ✓ Best saved (val={best_acc:.4f})")

    print("\n[FINAL REPORT]\n",
          classification_report(vl2, vp, target_names=list(INTENT_LABELS.keys()), zero_division=0))
    with open(os.path.join(MODEL_DIR, "intent_labels.json"), "w") as f:
        json.dump({"id2label": ID2LABEL, "label2id": INTENT_LABELS}, f, indent=2)
    print(f"\n[INFO] Intent model saved. Best val acc: {best_acc:.4f}")

if __name__ == "__main__":
    train()
