import numpy as np
import os
gpuIdxStr = '0'

random_seed = 2023

prefix = f'python new_exp.py --gpu {gpuIdxStr} --random_seed {random_seed} --wandb 1 --mode binary'

D = {
    3: [2**3],
    6: [2**3, 2**6],
    9: [2**3, 2**6, 2**9],
    12: [2**3, 2**6, 2**9, 2**12],
}

# Ensure the folder exists
if not os.path.exists('txtfiles'):
    os.makedirs('txtfiles')

for n in D.keys():
    for expo in D[n]:
        hyper_list = [
            f'--n {n} --m {expo} --train train1 --k 129 --llm_max_length 512 --epoch 8',
            f'--n {n} --m {expo} --train train1 --k {n+1} --llm_max_length 512 --epoch 8',
            f'--n {n} --m {expo} --train train2 --k 129 --llm_max_length 512 --epoch 8',
        ]
    
        for hyper in hyper_list:
            cl = f'{prefix} {hyper}'
            txt_name = cl.replace(' ', '-').replace('--', '')
            together = f'{cl} 2>&1 | tee txtfiles/{txt_name}.txt'
            #print(cl)
            print(together)
            os.system(together)