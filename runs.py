import numpy as np
import os

alphas =np.linspace(0.1,1,10)
l1_ratios =np.linspace(0.1,1,10)

for alpha in alphas:
    for l1 in l1_ratios:
        os.system(f"python simple_ML_model.py -a {alpha} -l1 {l1}")