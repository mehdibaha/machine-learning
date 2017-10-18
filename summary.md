# ML Summary

## Nearest Neighbour methods
* Computes distance between new point and all samples
* Pick k neighbours that are are nearest to x (majority vote to decide)
* *k << High Variance, Low Bias* & *k >> Low Variance, High Bias*
* **Pros**: Good performance, effective in low dimension data
* **Cons**: Memory requirement and costly to compute distances

## Decision Trees
* Branches with unique labels called leaves (stop tree)
    **else** Split recursively nodes by best information gain
* **Entropy**: **∑-pi\*log2(pi)**
* **Information gain**: **Gain(D, A) = Ent(S)−∑Ent(Sv)\*|Sv|/|S|**
    *where Sv is subset of dataset D with values ∈ A*
* **Gini impurity**: **1-∑pi^2** (replacement for Entropy)
* **Prune tree** by removing unnecessary branches
    *using a validation set to prevent overfitting*
    
## Regression
* *Goal: Predict target associated to any arbitrary new input*
* **Least Squares**: Minimize squared error between target and input
* **RANSAC**: Robust fit of model to data set S which contains outliers
* **PARAM vs Non-PARAM**: Param better if close to the true form of *f* or high dimension
* **Ridge Regression/The Lasso**: Useful for reducing to almost zero/zero certain features

## Challenges
* *Model complexity>> :: Training Error-- Test Error++ (sweet spot!)*
    *Training set*: Fitting the models
    *Validation set*: Determine hyperparameters
    *Test set*: General assesment of the model
* **Curse of dimensionality**: as dimensions grow, data points more sparse
* **Bias**-**Variance** tradeoff: Error = Variance + Bias^2
    * **Bias** measures average estimation with true function
    * **Variance** measures dependency of classifier on random sampling in training set
* **Conclusion**:
    Match the model complexity to the data resources, not to the target complexity
    
## Probalistic Reasoning
* **Probility Methods** make results interpretable, and define a unified ML theory
* **Bayes decision theory**: yMAP = argmax P(x|y)P(y)
   where P(x|y) is likelihood distribution and P(y) is prior distribution

## Probalistic Learning Framework
* **Maximum Likelihood Estimate**: Maximizing likelihood of data
   Problem: More features => More difficult to model
   Solution: **Naive Bayes** => All features are regarded as independent.
* **Maximum a Posteriori**: Model parameters are probabilities
* We can try to fit a **mixture** of K distributions:
   * **K-Means** finds centroids with neighboring points
     ++ Guaranteed to converge
     -- Sensitive to initial conditions
     -- Euclidian distance favours spherical clusters
   * Expectation Maximization w/ **Mixture of Gaussians**
     ++ Can describe complex multi-modeal probability distributions
     -- Data points are smoothly used to update all parameters
   
## Classification with Separating Hyperplanes
* y = sign(∑ xi\*wi)
* Finding the best weights to separate data
* **Perceptron Learning**: Incremental learning
   wi ← wi + η(t − y)xi (only change when output is wrong!)
* **Delta Rule**: Minimize ∑||t −w⃗T⃗x||^2
   wi ← w⃗ − η gradw⃗||t − w⃗ T⃗x||^2 (for each sample)
   wi ← wi + η(t − w⃗T ⃗x)xi
* Minimization of structural risk = Maximization of the margin

## Support Vector Machines
* We can separate almost everything in higher dimensions
   1. Transform Input in high-dimension w/ function φ
   2. Choose unique separating hyperplane
   3. Classify new data using hyperplane
* ++ Work well with small data ++ Fast ++ Generalize well
* Kernel trick can solve inefficiency by avoiding calculating higher dimenions
* **Maximize ∑αi−0.5\*∑ αi αj ti tj φ(⃗xi)T φ(⃗xj) under 0≤αi≤C ∀i**
   0. C = +inf for NO slack, C € R for allowing slack
   1. Choose kernel function
   2. Compute αi
   3. Classify data point x⃗ via ∑ αi ti K(x⃗,x⃗i) > 0

## Artificial Neural Networks
* 1-Layer NN can implement **any linear function**, 2-Layer NN **any function**
* Perceptron/delta impossible as we lose info on weight of each neuron
* **Trick**: Using continuous threshold-like functions
   *Goal*: Minize error as function of *all* weights
   1. Compute direction where total error increases most
   2. Back-propragate weights in opposite direction wi ← wi − η\*∂E/∂wi
* In deep networks, gradients **vanish** as they become really small
* **Convolutional Networks**
   
