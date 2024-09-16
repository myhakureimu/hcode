import numpy as np
import os
gpuIdxStr = '0'

random_seed = 2023

prefix = f'python exp.py --gpu {gpuIdxStr} --random_seed {random_seed} --wandb 1'

D = {
    4: [2,4],
    8: [4,8],
    12: [4,8,12],
    16: [4,8,12,16]
}

# Ensure the folder exists
if not os.path.exists('txtfiles'):
    os.makedirs('txtfiles')

for n in D.keys():
    for expo in D[n]:
        hyper_list = [
            f'--n {n} --m {2**expo} --train train1 --k 129 --epoch 4',
            f'--n {n} --m {2**expo} --train train1 --k {n+1} --epoch 4',
            f'--n {n} --m {2**expo} --train train2 --k 129 --epoch 4',
        ]
    
        for hyper in hyper_list:
            cl = f'{prefix} {hyper}'
            txt_name = cl.replace(' ', '-').replace('--', '')
            together = f'{cl} 2>&1 | tee txtfiles/{txt_name}.txt'
            #print(cl)
            print(together)
            os.system(together)



