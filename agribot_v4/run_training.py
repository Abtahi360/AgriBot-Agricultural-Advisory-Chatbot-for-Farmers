"""
run_training.py  — AgriBot v4
Runs all training scripts in the correct order.

Steps:
  1. Crop model (RF baseline, 99.5% acc)           — ~2s
  2. Fertilizer model (RF agronomic, 97.6% acc)    — ~3s
  3. XGBoost crop model (comparison)               — ~12s
  4. BERT intent classifier (comparison, GPU rec.) — ~15min GPU / ~60min CPU
  5. EfficientNet disease model                    — requires PlantVillage dataset
  6. YOLO disease model                            — requires PlantDoc dataset
  7. DistilBERT intent classifier (existing)       — ~15min GPU / ~60min CPU
  8. Evaluation                                    — ~30s

Usage:
  python run_training.py                  # runs steps 1-3 + evaluation (fast, no GPU)
  python run_training.py --full           # runs all including BERT (needs GPU)
  python run_training.py --step crop      # single step
  python run_training.py --step xgboost
  python run_training.py --step bert
  python run_training.py --step disease
  python run_training.py --step eval
"""

import subprocess, sys, os, argparse, time

BASE = os.path.dirname(os.path.abspath(__file__))

def run(name, script, required=True):
    print("\n" + "=" * 60)
    print(f"  TRAINING: {name}")
    print("=" * 60)
    t0  = time.time()
    res = subprocess.run(
        [sys.executable, os.path.join(BASE, script)],
        cwd=BASE, check=False)
    elapsed = round(time.time() - t0, 1)
    if res.returncode != 0:
        print(f"\n[ERROR] {name} failed (exit {res.returncode}) after {elapsed}s")
        if required:
            sys.exit(res.returncode)
        print("  [WARN] Continuing despite error (optional step)")
    else:
        print(f"\n[OK] {name} completed in {elapsed}s")
    return res.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="AgriBot v4 Training")
    parser.add_argument("--full",  action="store_true",
                        help="Run all steps including BERT and disease models")
    parser.add_argument("--step",  type=str, default=None,
                        choices=["crop","fertilizer","xgboost","bert","disease","eval"],
                        help="Run a single step")
    args = parser.parse_args()

    if args.step:
        steps = {
            "crop":       ("Crop RF Model",           "training/train_crop_model.py",           True),
            "fertilizer": ("Fertilizer RF Model",     "training/train_fertilizer_model.py",     True),
            "xgboost":    ("XGBoost Crop Comparison", "training/train_xgboost_crop.py",         True),
            "bert":       ("BERT Intent Classifier",  "training/train_bert_intent.py",           False),
            "disease":    ("EfficientNet Disease",    "training/train_disease_efficientnet.py", False),
            "eval":       ("Evaluation",              "evaluation/evaluate_all.py",              False),
        }
        name, script, req = steps[args.step]
        run(name, script, req)
        return

    # Always run core models
    run("Crop Recommendation Model (RF baseline)",  "training/train_crop_model.py",      True)
    run("Fertilizer Prediction Model (RF)",         "training/train_fertilizer_model.py", True)
    run("XGBoost Crop Comparison",                  "training/train_xgboost_crop.py",     True)
    run("DistilBERT Intent Classifier",             "training/train_intent_classifier.py", False)

    if args.full:
        run("Full BERT Intent Classifier (comparison)", "training/train_bert_intent.py",          False)
        run("EfficientNet Disease Classifier",          "training/train_disease_efficientnet.py", False)
        run("YOLOv8 Disease Detector",                  "training/train_disease_yolo.py",         False)

    run("Model Evaluation",  "evaluation/evaluate_all.py",  False)

    print("\n" + "=" * 60)
    print("  ALL TRAINING COMPLETE!")
    print("  Models saved in: models/")
    print("  Evaluation in:   evaluation/")
    print()
    print("  Start server:")
    print("    uvicorn api.main:app --reload --port 8000")
    print()
    print("  For full model comparison (needs GPU):")
    print("    python run_training.py --full")
    print("=" * 60)


if __name__ == "__main__":
    main()
