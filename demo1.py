__author__ = 'chao-shu'
import numpy as np
from collections import defaultdict

dataset_filename = "affinity_dataset.txt"
X = np.loadtxt(dataset_filename)
# print X
num_apple_purchase = 0
num_balanace_purchase = 0
for sample in X:
    if sample[3] == 1:
        num_apple_purchase += 1
    if sample[4] == 1:
        num_balanace_purchase += 1
print(num_apple_purchase)
print(num_balanace_purchase)

valid_rules = defaultdict(int)
invalid_rules = defaultdict(int)
num_occurances = defaultdict(int)

for sample in X:
    for premise in range(4):
        if sample[premise] == 0:
            continue
        else:
            num_occurances[premise] += 1
            for conclusion in range(4):
                if premise == conclusion: continue
                if sample[conclusion] == 1:
                    valid_rules[(premise, conclusion)] += 1
                else:
                    invalid_rules[(premise, conclusion)] += 1

support = valid_rules
confidence = defaultdict(float)
for permise, conclusion in valid_rules.keys():
    rules = (premise, conclusion)
    confidence[rules] = valid_rules[rules]/float(num_occurances[premise])
    print(rules)
    print(confidence[rules])

print(support)
print(confidence)
