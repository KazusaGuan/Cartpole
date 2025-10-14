import numpy as np
from collections import defaultdict

def f():
    return np.zeros(2)
d = defaultdict(f)

print(d[0].shape)