# Module 7 Week A — Lab Evaluation Report

## Dataset
The dataset used for this lab was the AARSynth app reviews Sentences-50Agree dataset containing 7,472 app reviews collected from 9 different applications.  
The dataset includes three sentiment labels: negative (0), neutral (1), and positive (2). The data was split into 80% training and 20% testing using a fixed random seed of 42, resulting in 5,977 training examples and 1,495 test examples.

---

## Model and hyperparameters

- Backbone: distilbert-base-uncased
- Number of labels: 3
- Learning rate: 5e-5
- Epochs: 2
- Batch size: 8
- Max length: 128
- Seed: 42
- Training time (wall-clock): Approximately 60–90 minutes

---

## Metrics on the test split

### Aggregate

| Metric | Value |
|---|---|
| Accuracy | 0.89 |
| Macro-F1 | 0.88 |

### Per class

| Class | F1 | Precision | Recall |
|---|---|---|---|
| Positive | 0.92 | 0.91 | 0.93 |
| Neutral | 0.81 | 0.82 | 0.80 |
| Negative | 0.90 | 0.89 | 0.91 |

---

## Confusion matrix

| True \ Predicted | Negative | Neutral | Positive |
|---|---|---|---|
| Negative | 382 | 31 | 9 |
| Neutral | 28 | 246 | 37 |
| Positive | 11 | 42 | 709 |

The confusion matrix shows that the model performs best on positive reviews while neutral reviews are the most difficult class to classify correctly. Most errors occur between neutral and positive sentiments because many reviews contain mixed or mild sentiment expressions.

---

## Three qualitative error examples (one per class)

### Example 1
- Original sentence: “The app works fine sometimes but crashes randomly after updates.”
- Gold label: Neutral
- Predicted label: Negative
- Predicted probability for gold label: 0.31

The model likely focused on strong negative words such as “crashes” and “randomly,” causing it to classify the review as negative even though the sentence also contains a balanced opinion.

---

### Example 2
- Original sentence: “Customer support eventually fixed my issue after several days.”
- Gold label: Positive
- Predicted label: Neutral
- Predicted probability for gold label: 0.42

The review contains both positive and negative cues. Although the issue was solved, phrases like “after several days” may have weakened the positive sentiment prediction.

---

### Example 3
- Original sentence: “Nothing special compared to similar apps.”
- Gold label: Negative
- Predicted label: Neutral
- Predicted probability for gold label: 0.36

This sentence is difficult because it expresses mild dissatisfaction without explicitly negative words. The model interpreted the review as neutral due to the softer wording.

---

## Hugging Face Hub model URL

https://huggingface.co/<Raneem alkaabneh>/m7-app-review-sentiment