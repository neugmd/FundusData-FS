""" Generate commands for pre-train phase. """
import os

def run_exp(lr=0.1, gamma=0.2, step_size=30):
    max_epoch = 50
    shot = 1
    query = 15
    way = 6
    gpu = 6
    base_lr = 0.01
    
    the_command = 'CUDA_VISIBLE_DEVICES=' + str(gpu) \
        + ' python3 main.py' \
        + ' --pre_max_epoch=' + str(max_epoch) \
        + ' --shot=' + str(shot) \
        + ' --train_query=' + str(query) \
        + ' --way=' + str(way) \
        + ' --pre_step_size=' + str(step_size) \
        + ' --pre_gamma=' + str(gamma) \
        + ' --gpu=' + str(gpu) \
        + ' --base_lr=' + str(base_lr) \
        + ' --pre_lr=' + str(lr) \
        + ' --phase=pre_train' 

    os.system(the_command)

run_exp(lr=0.1, gamma=0.2, step_size=15)
