#! /usr/bin/env python3

import numpy as np
import re

def get_dt(filename):
    with open(filename) as f:
        text = f.read()
    L = re.findall('dt = (.*)', text)
    return np.array([ float(x.split(' ', 1)[0]) for x in L]) # This line will have to be changed

dt1 = get_dt("positive_gradient.txt");
dt2 = get_dt("negative_gradient.txt");

print(dt1)
print(dt2)

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

uz = 1000
nt_per_omega_betatron = 0.07
dt_analytic = np.sqrt(2*uz)*nt_per_omega_betatron
error_analytic = (dt1[1]-dt_analytic)/dt_analytic
assert(error_analytic < 1e-5)
print("Error on the first time step ", error_analytic)
#assert()
error = np.sum(dt1 - dt2)/np.sum(dt2)
print("Error: ", error)
# Assert sub-permille error
assert(error < 1e-6)
for i in range(len(dt2)-2):
    assert(dt2[i+1] > dt2[i+2])
