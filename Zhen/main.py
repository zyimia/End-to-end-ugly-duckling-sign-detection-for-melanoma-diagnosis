import os
import time
import json
import yaml
import torch
import random
import numpy as np
import pickle as pkl
import torch.optim as optim
from sklearn import metrics
from data_proc.augment import Augmentations
from data_proc.csv_ext_dataset import CSV_Ext_Dataset
from torch.utils.data import DataLoader
from misc.training_manager import training_manager
from collections import OrderedDict
from misc.utils import get_pred_class_index
from misc.loss_function import get_loss_fun

threshold = 0.5
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def get_model_output(model, data):
    inputs = data['image']
    outputs = model(inputs.to(device))  # label: N * H * W
    return outputs


def set_seed(seed=2019):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def train(model, train_dataloader, optimizer, loss_fn):

    model.train()
    train_epoch_results = OrderedDict({'train_loss': [],
                                       'target': [],
                                       'pred_score': [],
                                       'pred_label': []})

    torch.autograd.set_detect_anomaly(True)

    for data in train_dataloader:
        optimizer.zero_grad()
        outputs = get_model_output(model, data, )  # N*C
        outputs = torch.sigmoid(outputs)

        loss = 0
        if isinstance(outputs, list) or isinstance(outputs, tuple):
            if len(outputs) > 1:
                for i in range(len(outputs)):
                    loss += loss_fn(outputs[i], data['target'].to(device))
            else:
                loss = loss_fn(outputs[0], data['target'].to(device))

            pred_score = outputs[0].detach().cpu().numpy()
        else:
            loss = loss_fn(outputs, data['target'].to(device))
            pred_score = outputs.detach().cpu().numpy()

        loss.backward()
        optimizer.step()

        # TODO: checking the pred_score and pred_label format
        train_epoch_results['train_loss'].append(loss.item())
        train_epoch_results['target'] += list(data['target'].detach().cpu().numpy())
        train_epoch_results['pred_score'] += list(pred_score.squeeze())
        train_epoch_results['pred_label'] += list(get_pred_class_index(pred_score, threshold))

    return train_epoch_results


def val(model, val_dataloader, loss_fn):
    val_eval_results = OrderedDict({'val_loss': [], 'target': [], 'pred_score': [], 'pred_label': []})

    model.eval()
    with torch.no_grad():
        for data in val_dataloader:
            outputs = get_model_output(model, data)
            outputs = torch.sigmoid(outputs)
            if isinstance(outputs, list) or isinstance(outputs, tuple):
                if len(outputs) > 1:
                    val_loss = 0
                    for i in range(len(outputs)):
                        val_loss += loss_fn(outputs[i], data['target'].to(device))
                else:
                    val_loss = loss_fn(outputs[0], data['target'].to(device))

                pred_score = outputs[0].detach().cpu().numpy()

            else:
                val_loss = loss_fn(outputs, data['target'].to(device))
                pred_score = outputs.detach().cpu().numpy()

            val_eval_results['val_loss'].append(val_loss.item())
            val_eval_results['target'] += list(data['target'].detach().cpu().numpy())
            val_eval_results['pred_score'] += list(pred_score.squeeze())
            val_eval_results['pred_label'] += list(get_pred_class_index(pred_score, threshold))

    return val_eval_results


def main(cfg):

    """step 1: setup data"""
    if not os.path.exists(cfg['training']['result_dir']):
        os.makedirs(cfg['training']['result_dir'])

    train_aug_parameters = OrderedDict({'affine': {'rotation': 90,
                                                   'shear': 10,
                                                   'scale': [0.8, 1.2]},
                                        'hflip': True,
                                        'vflip': True,
                                        'hair_mask': '/home/zyi/MedicalAI/ISIC_2020/hairs2/mask_array.npy',
                                        'color_trans': {'brightness': (0.75, 1.25),
                                                        'contrast': (0.75, 1.25),
                                                        'saturation': (0.75, 1.25),
                                                        'hue': (-0.05, 0.05)},
                                        'normalization': {'mean': (0.485, 0.456, 0.406),
                                                          'std': (0.229, 0.224, 0.225)},
                                        'size': 320,
                                        'scale': (0.7, 1.3),
                                        'ratio': (0.7, 1.3)
                                        }
                                       )

    val_aug_parameters = OrderedDict({'affine': None,
                                      'hflip': None,
                                      'vflip': None,
                                      'color_trans': None,
                                      'hair_mask': None,
                                      'normalization': {'mean': (0.485, 0.456, 0.406),
                                                        'std': (0.229, 0.224, 0.225)},
                                      'size': 256,
                                      'scale': (1.0, 1.0),
                                      'ratio': (1.0, 1.0)
                                      }
                                     )

    train_augmentor = Augmentations(augs=train_aug_parameters)
    val_augmentor = Augmentations(augs=val_aug_parameters)

    train_dataset = CSV_Ext_Dataset(data_csv=cfg['data']['csv_file'], data_root=cfg['data']['root'],
                                    ext_csv=cfg['data']['ext_csv'], is_train=True, fold=cfg['fold'],
                                    transform=train_augmentor.transform)

    val_dataset = CSV_Ext_Dataset(data_csv=cfg['data']['csv_file'], data_root=cfg['data']['root'],
                                  is_train=False, fold=cfg['fold'], transform=val_augmentor.transform)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True,
                                  num_workers=cfg['training']['num_workers'], drop_last=True)

    val_dataloader = DataLoader(val_dataset, batch_size=cfg['training']['batch_size'], shuffle=False,
                                num_workers=cfg['training']['num_workers'])

    """step 2: setup model & loss_fn & optimizer"""
    assert cfg['training']['optimizer']['name'] == 'Adam'
    assert cfg['training']['lr_scheduler']['name'] == 'ReduceLROnPlateau'

    train_manager = training_manager(cfg['training']['result_dir'], cfg['model'], optim.Adam, optim.lr_scheduler.ReduceLROnPlateau, device,
                                     cfg['training']['max_epoch'])

    train_manager.initialize_model(cfg['data']['channel'], train_dataset.n_class, cfg['training']['learning_rate'],
                                   cfg['training']['optimizer'], cfg['training']['lr_scheduler'],
                                   device, cfg['training']['fine_tune'], model=cfg['model'])

    loss_fn = get_loss_fun(cfg['training']['loss'])
    train_manager.print_to_log_file('we are using ' + cfg['training']['loss']['name'])

    if os.path.exists(cfg['training']['resume_path']):
        train_manager.load_checkpoint(best_model=False)

    with open(os.path.join(train_manager.output_dir, 'configs.yml'), 'w') as config_file:
        yaml.dump(cfg, config_file, default_flow_style=False)

    while train_manager.epoch < train_manager.max_epoch:
        train_manager.print_to_log_file("\nepoch: ", train_manager.epoch)
        epoch_start_time = time.time()

        train_epoch_results = train(train_manager.model, train_dataloader, train_manager.optimizer, loss_fn)
        val_epoch_results = val(train_manager.model, val_dataloader, loss_fn)

        # evaluation, plot_training_curve, update_lr, save_checkpoint, update eval_criterion
        train_manager.train_losses.append(np.mean(train_epoch_results['train_loss']))
        train_manager.val_losses.append(np.mean(val_epoch_results['val_loss']))
        #
        print('train epoch loss {:.4f}'.format(train_manager.train_losses[-1]))
        print('val epoch loss {:.4f}'.format(train_manager.val_losses[-1]))

        train_manager.compute_metrics(train_epoch_results, train=True)
        train_manager.compute_metrics(val_epoch_results, train=False)
        print('train_auc {:.4f}'.format(train_manager.train_eval_metrics['auc'][-1]))
        print('val_auc {:.4f}'.format(train_manager.val_eval_metrics['auc'][-1]))

        train_manager.epoch += 1
        continue_training = train_manager.run_on_epoch_end()

        epoch_end_time = time.time()
        if not continue_training:
            break

        train_manager.print_to_log_file("This epoch took {:.2f} s\n".format(epoch_end_time - epoch_start_time))

    train_manager.save_checkpoint(os.path.join(train_manager.output_dir, train_manager.model_name + "_final.model"))
    # now we can delete latest as it will be identical with final
    if os.path.isfile(os.path.join(train_manager.output_dir, train_manager.model_name + "_scheduled.model")):
        os.remove(os.path.join(train_manager.output_dir, train_manager.model_name + "_scheduled.model"))


if __name__ == '__main__':
    seed = random.randint(1, 10000)
    yml_file = '/home/zyi/My_disk/ISIC_2020/configs/skin_config.yml'
    with open(yml_file) as f:
        config_file = yaml.load(f)

    config_file['training']['seed'] = seed

    if config_file['debug']:
        config_file['training']['max_epoch'] = 1

    config_file['model'] = 'efficient-siimd'
    config_file['run_exp'] = '/home/zyi/My_disk/ISIC_2020/run_exp'
    config_file['data']['csv_file'] = '/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/ISIC_2020/train.csv'
    config_file['data']['root'] = '/home/zyi/MedicalAI/ISIC_2020/data/train'
    config_file['data']['ext_csv'] = '/home/zyi/MedicalAI/ISIC_2020/ext_dataset.csv'
    print(config_file['training']['learning_rate'])

    if not os.path.exists(config_file['training']['resume_path']):
        run_dir = config_file['model']
        config_file['training']['result_dir'] = os.path.join(config_file['run_exp'], run_dir, 'fold_' + cfg['fold'])
    else:
        config_file['training']['result_dir'] = config_file['training']['resume_path']

    main(config_file)
