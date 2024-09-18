import numpy as np
import os
gpuIdxStr = '0'

random_seed = 2024

prefix = f'python exp.py --gpu {gpuIdxStr} --random_seed {random_seed} --wandb 1'

D = {
    4: [2],
    8: [2,6],
    12: [2,6,10],
}

# Ensure the folder exists
if not os.path.exists('txtfiles'):
    os.makedirs('txtfiles')

for n in D.keys():
    for expo in D[n]:
        hyper_list = [
            f'--n {n} --m {2**expo} --train train1 --k 129 --epoch 8',
            f'--n {n} --m {2**expo} --train train1 --k {n+1} --epoch 8',
            f'--n {n} --m {2**expo} --train train2 --k 129 --epoch 8',
        ]
    
        for hyper in hyper_list:
            cl = f'{prefix} {hyper}'
            txt_name = cl.replace(' ', '-').replace('--', '')
            together = f'{cl} 2>&1 | tee txtfiles/{txt_name}.txt'
            #print(cl)
            print(together)
            os.system(together)



