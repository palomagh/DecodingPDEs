import math, copy
import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F
from functools import partial

# import data loaders, task-specific, losses and metrics
from data_loaders import load_text, load_pde, load_pythia_14m, load_pythia_70m, load_pythia_160m, load_pythia_410m, load_pythia_1b, load_pythia_14b, load_gpt, load_gptm, load_gptl, load_gptxl
from utils import get_params_to_update, set_param_grad, set_grad_state
from utils import accuracy, inverse_score, nmse, rmse_loss, nrmse_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data(root, dataset, batch_size, valid_split, maxsize=None, get_shape=False, flip=False, double=False):
    data_kwargs = None

    if dataset == "your_new_task": # modify this to experiment with a new task
        train_loader, val_loader, test_loader = None, None, None
    elif dataset == "text": # the dataset created for roberta
        train_loader, val_loader, test_loader = load_text(root, batch_size, maxsize=maxsize)
    elif dataset[:3] == 'PDE':
        train_loader, val_loader, test_loader = load_pde(root, batch_size, flip, double, dataset=dataset[4:], valid_split=valid_split)
    elif dataset == "pythia-14m":
        train_loader, val_loader, test_loader = load_pythia_14m(root, batch_size, maxsize=maxsize)
    elif dataset == "pythia-70m":
        train_loader, val_loader, test_loader = load_pythia_70m(root, batch_size, maxsize=maxsize)
    elif dataset == "pythia-160m":
        train_loader, val_loader, test_loader = load_pythia_160m(root, batch_size, maxsize=maxsize)
    elif dataset == "pythia-410m":
        train_loader, val_loader, test_loader = load_pythia_410m(root, batch_size, maxsize=maxsize)
    elif dataset == "pythia-1b":
        train_loader, val_loader, test_loader = load_pythia_1b(root, batch_size, maxsize=maxsize)
    elif dataset == "pythia-1,4b":
        train_loader, val_loader, test_loader = load_pythia_14b(root, batch_size, maxsize=maxsize)
    elif dataset == "gpt2":
        train_loader, val_loader, test_loader = load_gpt(root, batch_size, maxsize=maxsize)
    elif dataset == 'gpt2-medium':
        train_loader, val_loader, test_loader = load_gptm(root, batch_size, maxsize=maxsize)
    elif dataset == 'gpt2-large':
        train_loader, val_loader, test_loader = load_gptl(root, batch_size, maxsize=maxsize)
    elif dataset == 'gpt2-xl':
        train_loader, val_loader, test_loader = load_gptxl(root, batch_size, maxsize=maxsize)
    
    n_train, n_val, n_test = len(train_loader), len(val_loader) if val_loader is not None else 0, len(test_loader)

    if not valid_split:
        val_loader = test_loader
        n_val = n_test

    return train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs


def get_config(root, args):
    dataset = args.dataset
    args.infer_label = False
    args.activation = None
    args.target_seq_len = 512 if not hasattr(args, 'target_seq_len') else args.target_seq_len
    print("target_seq_len", args.target_seq_len)
    
    if dataset == "your_new_task": # modify this to experiment with a new task
        dims, num_classes = None, None
        loss = None

    elif dataset == 'PDE-Burgers':
        dims, sample_shape, num_classes = 1, (1, 1, 256), (1, 1024)
        loss = rmse_loss 
        args.infer_label = True

    elif dataset == 'PDE-1DCFD':
        dims, sample_shape, num_classes = 1, (1, 1, 3072), (1, 3072) 
        loss = rmse_loss 
        args.infer_label = True

    elif dataset == 'PDE-ADV':
        dims, sample_shape, num_classes = 1, (1, 1, 256), (1, 256)
        loss = nrmse_loss 
        args.infer_label = True

    elif dataset == 'PDE-RD':
        dims, sample_shape, num_classes = 1, (1, 1, 1024), (1, 1024) 
        loss = nrmse_loss 
        args.infer_label = True

    elif dataset == 'PDE-DS':
        dims, sample_shape, num_classes = 1, (1, 1, 1024), (1, 1024)
        loss = nrmse_loss 
        args.infer_label = True
    
    return dims, sample_shape, num_classes, loss, args


def get_metric(root, dataset):
    if dataset == "your_new_task": # modify this to experiment with a new task
        return inverse_score(accuracy), np.min
    
    if dataset[:3] == 'PDE':
        return nmse, np.min


def get_optimizer(name, params):
    if name == 'SGD':
        return partial(torch.optim.SGD, lr=params['lr'], momentum=params['momentum'], weight_decay=params['weight_decay'])
    elif name == 'Adam':
        return partial(torch.optim.Adam, lr=params['lr'], betas=tuple(params['betas']), weight_decay=params['weight_decay'])
    elif name == 'AdamW':
        return partial(torch.optim.AdamW, lr=params['lr'], betas=tuple(params['betas']), weight_decay=params['weight_decay'])


def get_scheduler(name, params, epochs=200, n_train=None):
    if name == 'StepLR':
        sched = params['sched']

        def scheduler(epoch):    
            optim_factor = 0
            for i in range(len(sched)):
                if epoch > sched[len(sched) - 1 - i]:
                    optim_factor = len(sched) - i
                    break
                    
            return math.pow(params['base'], optim_factor)  

        lr_sched_iter = False

    elif name == 'WarmupLR':
        warmup_steps = int(params['warmup_epochs'] * n_train)
        total_steps = int(params['decay_epochs'] * n_train)
        lr_sched_iter = True

        def scheduler(step):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))

            current_decay_steps = total_steps - step
            total_decay_steps = total_steps - warmup_steps
            f = (current_decay_steps / total_decay_steps)

            return f  

    elif name == 'ExpLR':
        warmup_steps = int(params['warmup_epochs'] * n_train)
        total_steps = int(params['decay_epochs'] * n_train)
        lr_sched_iter = True

        def scheduler(step):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))

            current_decay_steps = total_steps - step
            total_decay_steps = total_steps - warmup_steps
            f = (current_decay_steps / total_decay_steps)

            return params['base'] * f  

    elif name == 'SinLR':

        cycles = 0.5
        warmup_steps = int(params['warmup_epochs'] * n_train)
        total_steps = int(params['decay_epochs'] * n_train)
        lr_sched_iter = True

        def scheduler(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            # progress after warmup
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1. + math.cos(math.pi * float(cycles) * 2.0 * progress)))

    return scheduler, lr_sched_iter


def get_optimizer_scheduler(args, model, module=None, n_train=1):
    if module is None:
        set_grad_state(model, True)
        set_param_grad(model, args.finetune_method)
        optimizer = get_optimizer(args.optimizer.name, args.optimizer.params)(get_params_to_update(model, ""))
        lr_lambda, args.lr_sched_iter = get_scheduler(args.scheduler.name, args.scheduler.params, args.epochs, n_train)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return args, model, optimizer, scheduler

    elif module == 'embedder':
        embedder_optimizer_params = copy.deepcopy(args.optimizer.params)
        if embedder_optimizer_params['lr'] <= 0.001:
            embedder_optimizer_params['lr'] *= 10
        embedder_optimizer = get_optimizer(args.optimizer.name, embedder_optimizer_params)(get_params_to_update(model, ""))
        lr_lambda, _ = get_scheduler(args.no_warmup_scheduler.name, args.no_warmup_scheduler.params, args.embedder_epochs, 1)
        embedder_scheduler = torch.optim.lr_scheduler.LambdaLR(embedder_optimizer, lr_lambda=lr_lambda)

        return args, model, embedder_optimizer, embedder_scheduler

    elif module == 'predictor':

        try:
            predictor = model.predictor
            set_grad_state(model, False)
            for n, m in model.embedder.named_parameters():
                m.requires_grad = True
            for n, m in model.predictor.named_parameters():
                m.requires_grad = True

            predictor_optimizer_params = copy.deepcopy(args.optimizer.params)
            if predictor_optimizer_params['lr'] <= 0.001:
                predictor_optimizer_params['lr'] *= 10
            predictor_optimizer = get_optimizer(args.optimizer.name, predictor_optimizer_params)(get_params_to_update(model, ""))
            lr_lambda, args.lr_sched_iter = get_scheduler(args.no_warmup_scheduler.name, args.no_warmup_scheduler.params, args.predictor_epochs, 1)
            predictor_scheduler = torch.optim.lr_scheduler.LambdaLR(predictor_optimizer, lr_lambda=lr_lambda)

            return args, model, predictor_optimizer, predictor_scheduler
        except:
            print("No predictor module.")

