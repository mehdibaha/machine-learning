import dtree as d
import matplotlib.pyplot as plt
import monkdata as m
import numpy as np
import random

from drawtree_qt5 import drawTree

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata) # DEACTIVATE THIS
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def pruneTree(tree, pruning_set):
    pruned = d.allPruned(tree)
    max_score = 0
    for prune in pruned:
        score = d.check(prune, pruning_set)
        if score >= max_score:
            max_score = score
            max_pruned = prune
    return max_pruned, max_score

def getClassError(fraction, data, runs=500):
    train, val = partition(data, fraction)
    t = d.buildTree(train, m.attributes)
    max_score = d.check(t, val)
    scores = []
    for _ in range(runs):
        train, val = partition(data, fraction)
        score = d.check(t, val)
        p, s = pruneTree(t, val)
        scores.append(s)
    avg, spread = np.mean(scores), np.std(scores)
    return avg, spread

# getting data
fractions = [f/10.0 for f in range(3, 9)]
means1, spreads1 = zip(*[getClassError(f, m.monk1) for f in fractions])
means3, spreads3 = zip(*[getClassError(f, m.monk3) for f in fractions])
# settings linespaces
x = np.array(fractions)
y1 = 1 - np.array(means1)
e1 = np.array(spreads1)
y3 = 1 - np.array(means3)
e3 = np.array(spreads3)
# plotting
plt.errorbar(x, y1, e1, linestyle='None', marker='x', label='monk1')
plt.errorbar(x, y3, e3, linestyle='None', marker='x', label='monk3')
plt.title('Reduced error pruning in Decision Trees')
plt.legend(loc='upper left')
plt.xlabel('Partition fraction')
plt.ylabel('Classification Error')
plt.show()
