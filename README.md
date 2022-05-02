# CS 4262 Project

**Hypothesis**: SVM classification on features learned from neural networks (specifically the deeper the better), performs better than SVM classification on raw input features (even when using an RBF gaussian that maps features to infinite dimensional space) and also performs better than end to end neural network classification.

### Classification Methods to Compare
1. Raw features with end to end NN classifier
2. Raw features with SVM classifier (linear kernel)
3. Raw features with SVM classifier (rbf kernel)
4. NN extracted features with SVM classifier (linear kernel- same dim feature space)
5. NN extracted features with SVM classifier (rbf kernel- infinite dim feature space)
6. PCA extracted features with SVM classifier (linear kernel)
7. PCA extracted featrues with SVM classifier (rbf kernel)

Variations in the neural network feature extractor we could try:
- use same NN for feature extraction and end to end classification. # neurons in final hidden layer = # features in original dataset (compares quality of learned features)
- Different depth neural networks (keep width consistent)
    - deeper networks will probably learn richer features that make classification perform better regardless of the task
    - will have to watch out for network just memorizing the dataset if we use the dates one with 1k training examples
### Notes
- Standardization important for improving performance of both SVM and neural networks 
- F1 Score for comparison metric is better than just plain accuracy (`sklearn.metrics.f1_score`)