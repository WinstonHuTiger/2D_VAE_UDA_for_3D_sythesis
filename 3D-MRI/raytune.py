from argparse import Namespace
from models import create_model
from models.base_model import BaseModel
import os
import torch
from options.train_options import TrainOptions
from data import create_dataset
from ray import tune
from ray.tune.schedulers.pb2 import PB2
from ray.tune.trial import ExportFormat
import numpy as np
from util.plot_pbt import plotPBT
import random
SEED = random.randint(0,1e6)

validation_loss_fun = torch.nn.L1Loss()

def update_options(opt: Namespace, update_opts: dict):
    opt = vars(opt)
    opt.update(update_opts)
    opt = Namespace(**opt)
    return opt

# tensorboard --logdir ray_results/

def get_score(dataset, opt, model: BaseModel):
    validation_loss_array = []
    opt.phase='test'
    tmp = opt.serial_batches, opt.paired
    opt.serial_batches=True
    opt.paired = True
    for test_data in dataset:
        model.set_input(test_data)
        model.test()
        validation_loss_array.append(validation_loss_fun(model.fake_B, model.real_B).item())
    opt.phase='train'
    opt.serial_batches, opt.paired = tmp
    score = np.mean(validation_loss_array)
    return score

def training_function(config, checkpoint_dir=None):
    torch.manual_seed(config['seed'])
    opt = update_options(init_opt, config)
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    opt.phase='test'
    test_dataset = create_dataset(opt)
    opt.phase='train'

    model: BaseModel = create_model(opt)
    initialize=False
    if checkpoint_dir is not None:
        path = os.path.join(checkpoint_dir, "checkpoint")
        checkpoint = model.load_checkpoint(path)
        initialize=True
        step = checkpoint['step']
        scores = checkpoint['scores']
    else:
        scores=[]
        step=0

    iter_to_next_display = opt.display_freq
    while True:
        # Loads batch_size samples from the dataset
        for i, data in enumerate(dataset):
            if initialize:
                model.data_dependent_initialize(data)
                model.setSchedulers(opt)               # regular setup: load and print networks; create schedulers
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            dataset.dataset.updateDataAugmentation()
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            iter_to_next_display-= opt.batch_size

            if iter_to_next_display<=0:
                scores.append(get_score(test_dataset, opt, model))
                step+=1
                iter_to_next_display += opt.display_freq

                if step%STEPS_TO_NEXT_CHECKPOINT == 0 or scores[-1] <= min(scores):
                    with tune.checkpoint_dir(step=step) as checkpoint_dir:
                        path = os.path.join(checkpoint_dir, "checkpoint")
                        d = {
                            'step': step,
                            'score': scores[-1],
                            'scores': scores
                        }
                        model.create_checkpoint(path, d)

                tune.report(score=scores[-1])
            
            

class CustomStopper(tune.Stopper):
        '''
        Implements a custom Pleatau Stopper that stops all trails once there was no improvement on the score by more than
        self.tolerance for more than self.patience steps.
        '''
        def __init__(self):
            self.should_stop = False
            # self.max_iter = 350
            self.patience = 70
            self.tolerance = 0.001
            self.scores = []

        def __call__(self, trial_id, result):
            step = result["training_iteration"]-1
            if  len(self.scores)<=step:
                self.scores.append(result["score"])
            else:
                self.scores[step] = min(self.scores[step], result["score"])
            return self.should_stop or (len(self.scores)>self.patience and min(self.scores[-self.patience:]) > min(self.scores[:-self.patience])-self.tolerance)

        def stop_all(self):
            return self.should_stop

def compute_gpu_load(num_trails):
    return {"gpu": int(100.0/num_trails)/100.0 }

search_space = {
            "lr":  [0.0001, 0.0003]
        }

PBB = PB2(
    time_attr="training_iteration",
    perturbation_interval=5,
    hyperparam_bounds=search_space
)

init_opt = TrainOptions().parse()
os.environ["RAY_MEMORY_MONITOR_ERROR_THRESHOLD"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(list(map(str, init_opt.gpu_ids)))
init_opt.gpu_ids = [0]
stopper = CustomStopper()
BEST_CHECKPOINT_PATH = os.path.join('ray_results/', init_opt.name, 'best')
STEPS_TO_NEXT_CHECKPOINT = 5

start_config = {key: tune.uniform(*val) for key, val in search_space.items()}
start_config.update({'seed': SEED})

analysis = tune.run(
    training_function,
    local_dir='ray_results/',
    name=init_opt.name,
    scheduler=PBB,
    metric="score",
    checkpoint_score_attr="min-score",
    mode="max",
    stop=stopper,
    export_formats=[ExportFormat.MODEL],
    resources_per_trial=compute_gpu_load(3),
    keep_checkpoints_num=1,
    num_samples=8,
    config=start_config,
    raise_on_failed_trial=False
)

plotPBT(os.path.join('ray_results/', init_opt.name))