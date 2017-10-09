# Report

## Assignment 0
MONK-2 would be the hardest dataset to build a decision tree for because the underlying attribute relations is hard(est?) to describe in terms of "simple" decisions.

## Assignment 1
Entropies of each dataset:
* MONK-1: 1.000
* MONK-2: 0.957
* MONK-3: 0.999

## Assignment 2
In a uniform distribution, each class has an equal probability of appearing in the dataset, so:
    pk = 1/k (where c is number of classes)
So the entropy is equal to:
    E(uniform) = sum_k( -1/k * log_k(1/k) )
    E(uniform) = -log_k(1/k)
    E(uniform) = 1

On the other hand, in a non-uniform distribution, entropy is always lower than 1, and the more random it is, the lowest it is to 1.
Example distributions with high entropy: A fair coin toss.
Example distributions with low entropy: A biased coin toss (25% of getting heads, 75% tails).

## Assignment 3
* MONK-1: {A1: 0.07527255560831925, A2: 0.005838429962909286, A3: 0.00470756661729721, A4: 0.02631169650768228, A5: 0.28703074971578435, A6: 0.0007578557158638421} -> A5
* MONK-2: {A1: 0.0037561773775118823, A2: 0.0024584986660830532, A3: 0.0010561477158920196, A4: 0.015664247292643818, A5: 0.01727717693791797, A6: 0.006247622236881467} -> A5
* MONK-3: {A1: 0.007120868396071844, A2: 0.29373617350838865, A3: 0.0008311140445336207, A4: 0.002891817288654397, A5: 0.25591172461972755, A6: 0.007077026074097326} -> A2 and A5

A2 and A5 are strong contenders, but A5 seems the most convincing for all example datasets.

## Assignment 4
When the information gain is maximised, some of the subsets now have a null entropy which means we can completely determine them based on their attribute.
This is a useful heuristic because it allows us to reduce the entropy, i.e make the subsets more deterministic and less "random" when a certain attribute is picked out.

## Assignment 5
The ones with the most information gains for each subset, i.e:
* 0: -, 1: A4, 2: A6, 3: A1

Results:
* MONK-1: {Etrain: 1.0, Etest: 0.82}
* MONK-2: {Etrain: 1.0, Etest: 0.69}
* MONK-3: {Etrain: 1.0, Etest: 0.94}

Conclusion:
No, because the one with highest entropy is hard to train (2nd lowest score).

## Assignment 6
Pruning is very useful to limit the complexity of the model, because following Occahm's razor, the model should the simplest possible.
So while by pruning our decision tree, we can make our bias bigger (the model performs average in most datasets), it has the advantage of highly reducing the variance by limiting the complexity of the decision tree.

## Assignment 7
The test error is reduced with pruning. We determine that the best partition is

