# ML Summary

## Nearest Neighbour methods
* Computes distance between new point and all samples
* Pick k neighbours that are are nearest to x (majority vote to decide)
* *k << : High Variance, Low Bias* & *k >> : Low Variance, High Bias*
* **Pros**: Good performance, effective in low dimension data
* **Cons**: Memory requirement and costly to compute distances

## Decision Trees
* Branches with unique labels called leaves (stop tree)
    **else** Split recursively nodes by best information gain
* Entropy: **SUM -pi \* log2(pi)**
* Information gain: **Gain(D, A) = Ent(S) − SUM Ent(Sv) \* |Sv|/|S|**
    *where Sv is subset of dataset D with values ∈ A*
* Gini impurity: **1 - SUM pi^2** (replacement for Entropy)
* Prune tree by removing unnecessary branches
    *using a validation set to prevent overfitting*

## Challenges
* Model complexity >> Training Error-- Test Error++ (sweet spot!)
    Training set: Fitting the models
    Validation set: Determine hyperparameters
    Test set: General assesment of the model
* Curse of dimensionality: as dimensions grow, data points more sparse
* **Bias**-**Variance** tradeoff: Error = Variance + Bias^2
    * **Bias** measures average estimation with true function
    * **Variance** measures dependency of classifier on random sampling in training set
* **Conclusion**:
    Take home message: Match the model complexity to the data resources, not to the target complexity
