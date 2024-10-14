import numpy as np
import os
gpuIdxStr = '0'

random_seed = 2024

prefix = f'python new_exp.py --gpu {gpuIdxStr} --random_seed {random_seed} --wandb 1 --mode binary --expName H_ModelSize'

D = {
    2: [2**2],
    4: [2**4],
    8: [2**8],
}

# Ensure the folder exists
if not os.path.exists('txtfiles'):
    os.makedirs('txtfiles')

for n in D.keys():
    for expo in D[n]:
        hyper_list = [
            f'--n {n} --m {expo} --modelName nano --num_heads 2 --depth 8 --embed_dim 128 --train train1 --k 129 --llm_max_length 512 --epoch 8',
            f'--n {n} --m {expo} --modelName dual --num_heads 2 --depth 8 --embed_dim 128 --train train1 --k 129 --llm_max_length 512 --epoch 8',
            f'--n {n} --m {expo} --modelName dual --num_heads 8 --depth 8 --embed_dim 1024 --train train1 --k 129 --llm_max_length 512 --epoch 8',
            f'--n {n} --m {expo} --modelName nano --num_heads 8 --depth 8 --embed_dim 1024 --train train1 --k 129 --llm_max_length 512 --epoch 8',
        ]
    
        for hyper in hyper_list:
            cl = f'{prefix} {hyper}'
            txt_name = cl.replace(' ', '-').replace('--', '')
            together = f'{cl} 2>&1 | tee txtfiles/{txt_name}.txt'
            #print(cl)
            print(together)
            os.system(together)