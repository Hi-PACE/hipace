#! /usr/bin/env python3

import numpy as np

with open("negative_gradient.txt") as file_in:
    negative_gradient_lines = []
    for line in file_in:
        words = line.split()
        if len(words)>0:
            if words[0] == "Rank":
                negative_gradient_lines.append(float(words[-1]))
negative_gradient_lines = np.array(negative_gradient_lines)

with open("positive_gradient.txt") as file_in:
    positive_gradient_lines = []
    for line in file_in:
        words = line.split()
        if len(words)>0:
            if words[0] == "Rank":
                positive_gradient_lines.append(float(words[-1]))
positive_gradient_lines = np.array(positive_gradient_lines)

error = np.sum(positive_gradient_lines - negative_gradient_lines)/np.sum(negative_gradient_lines)
print("Error: ", error)
# Assert sub-permille error
assert(error < 1e-6)
for i in range(len(negative_gradient_lines)-1):
    assert(negative_gradient_lines[i] > negative_gradient_lines[i+1])
