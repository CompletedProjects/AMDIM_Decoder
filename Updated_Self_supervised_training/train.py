import os
#import argparse

import torch

import mixed_precision
from stats import StatTracker
from datasets import Dataset, build_dataset, get_dataset, get_encoder_size
from model import Model
from checkpoint import Checkpointer
from task_self_supervised import train_self_supervised
from task_classifiers import train_classifiers

args = {'dataset': 'C10',
'batch_size': 200,
'learning_rate': 0.0002,
'seed': 1,
'amp': False,
'classifiers': False,
'decoder': False, # Nawid - Used to train the decoder
'ndf': 128,
'n_rkhs': 1024,
'tclip': 20.0,
'n_depth': 3,
'use_bn':0,
'output_dir' : '/runs',
'input_dir': '/mnt/imagenet',
'cpt_load_path': None,
'cpt_name' : 'amdim_cpt.pth',
'run_name' : 'default_run'
}
def main():
    # create target output dir if it doesn't exist yet
    if not os.path.isdir(args['output_dir']):
        os.mkdir(args['output_dir'])

    # enable mixed-precision computation if desired
    if args['amp']:
        mixed_precision.enable_mixed_precision()

    # set the RNG seeds (probably more hidden elsewhere...)
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])

    # get the dataset
    dataset = get_dataset(args['dataset'])
    encoder_size = get_encoder_size(dataset)

    # get a helper object for tensorboard logging
    log_dir = os.path.join(args['output_dir'], args['run_name'])
    stat_tracker = StatTracker(log_dir=log_dir)

    # get dataloaders for training and testing
    train_loader, test_loader, num_classes = \
        build_dataset(dataset=dataset,
                      batch_size=args['batch_size'],
                      input_dir=args['input_dir'],
                      labeled_only=args['classifiers'])

    torch_device = torch.device('cuda')
    checkpointer = Checkpointer(args['output_dir'])
    if args['cpt_load_path']:
        model = checkpointer.restore_model_from_checkpoint(
                    args['cpt_load_path'],
                    training_classifier=args['classifiers'])
    else:
        # create new model with random parameters
        model = Model(ndf=args['ndf'], n_classes=num_classes, n_rkhs=args['n_rkhs'],
                    tclip=args['tclip'], n_depth=args['n_depth'], encoder_size=encoder_size,
                    use_bn=(args['use_bn'] == 1))
        model.init_weights(init_scale=1.0)
        checkpointer.track_new_model(model)


    model = model.to(torch_device)

    # select which type of training to do
    task = train_classifiers if args['classifiers'] else train_self_supervised
    if args['classifiers']:
        task = train_classifiers
    elif args['decoder']:
        task = train_decoder
    else:
        task = train_self_supervised

    task(model, args['learning_rate'], dataset, train_loader,
         test_loader, stat_tracker, checkpointer, args['output_dir'], torch_device)

'''
if __name__ == "__main__":
    print(args)
    main()
'''
