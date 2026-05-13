import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from manual_eval import manual_predict
from calibration import (
    reliability_diagram,
    expected_calibration_error,
    plot_reliability
)

# load model
model = AutoModelForSequenceClassification.from_pretrained("model")
tokenizer = AutoTokenizer.from_pretrained("model")

texts = open("fixtures/tiny_app_reviews.csv").read().splitlines()

# predictions
preds, probs = manual_predict(model, tokenizer, texts)

# calibration
centers, accs, counts = reliability_diagram(probs, preds)
ece = expected_calibration_error(probs, preds)

print("ECE:", ece)
print("centers:", centers)
print("accs:", accs)
print("counts:", counts)



plot_reliability(
    centers,
    accs,
    counts,
    "stretch/tuesday/figures/reliability-diagram.png"
)


print("Saved reliability diagram!")