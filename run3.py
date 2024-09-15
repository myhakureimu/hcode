import numpy as np
import os
gpuIdxStr = ' -gpuIdx 0'
cl_list = [
    'python exp.py --wandb 1 --expName OOD_paper --L 4 --TV 1 --save save --n_tokens 8 --dfas_index 3 --epoch 40',
    'python exp.py --wandb 1 --expName OOD_paper --L 4 --TV 0 --save load --n_tokens 8 --dfas_index 3 --epoch 40',
    'python exp.py --wandb 1 --expName OOD_paper --L 4 --TV 1 --save save --n_tokens 3 --dfas_index 3 --epoch 40',
    'python exp.py --wandb 1 --expName OOD_paper --L 4 --TV 0 --save load --n_tokens 3 --dfas_index 3 --epoch 40',
    'python exp.py --wandb 1 --expName OOD_paper --L 4 --TV 1 --save save --n_tokens 5 --dfas_index 3 --epoch 40',
    'python exp.py --wandb 1 --expName OOD_paper --L 4 --TV 0 --save load --n_tokens 5 --dfas_index 3 --epoch 40',
    ]
    
for cl in cl_list:
    os.system(cl)
#os.system('sudo shutdown')



