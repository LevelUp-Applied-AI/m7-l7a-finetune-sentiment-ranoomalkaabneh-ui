# Calibration Analysis

> **TODO** — fill each section after running your manual evaluation and reliability diagram.

# Calibration Analysis

## Reliability diagram interpretation

The reliability diagram showed that most model predictions were concentrated in medium and high confidence buckets between 0.4 and 1.0. 
The lower-confidence buckets between 0.0 and 0.4 were mostly empty, which suggests that the model rarely produced uncertain predictions.

The model appeared slightly under-confident because several confidence buckets achieved empirical accuracy close to 1.0 while their predicted confidence values were lower. 
For example, predictions in the 0.45 confidence bucket still achieved very high empirical accuracy.

## Expected Calibration Error

The Expected Calibration Error (ECE) was 0.257.

This indicates that the model’s predicted probabilities are not perfectly calibrated with actual correctness. 
Although the classifier achieved strong accuracy in higher-confidence buckets, the relatively high ECE suggests that confidence scores may still be unreliable for production systems that depend on trustworthy probabilities.

## A specific calibration pattern

One clear calibration pattern was that most predictions were grouped into medium-to-high confidence ranges (0.5–1.0). 
This likely happened because the model was fine-tuned on a relatively small dataset, causing it to become highly confident on examples similar to the training data.

## A proposed engineering action

One useful engineering improvement would be applying temperature scaling after training to improve probability calibration. 
Another improvement would be collecting more difficult or ambiguous app reviews so the model learns better decision boundaries near uncertain examples.