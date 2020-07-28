import os
import sys
import torch
import time
import matplotlib
from models import get_model
from datetime import datetime
from time import time, sleep
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch.optim as optim
import sklearn.metrics as metrics
from models import convert_state_dict
from misc.utils import calculate_metrics
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class training_manager():
    def __init__(self, output_dir, model_name, optimizer, lr_scheduler, device, max_epoch):
        self.model = None
        self.model_name = model_name
        self.optimizer = optimizer
        self.output_dir = output_dir
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.after_lr_scheduler = None
        self.log_file = None

        self.epoch = 0
        self.max_epoch = max_epoch
        self.patience = 5
        self.save_every = 5

        self.train_losses = []
        self.val_losses = []
        self.val_eval_metrics = OrderedDict({'accuracy': [], 'auc': [], 'f1-score': [], 'recall': [], 'precision': []})
        self.train_eval_metrics = OrderedDict({'accuracy': [], 'auc': [], 'f1-score': [], 'recall': [], 'precision': []})

        self.monitor_MA_eps = 5e-4
        self.monitor_MA = None
        self.val_eval_criterion_MA = None
        self.lr_threshold = 3e-6

        self.best_monitor_MA = None            # used for managing lr_update
        self.best_val_eval_criterion_MA = None    # used for saving best model
        self.best_epoch_based_on_monitor_MA = None  # used for checking if continue training or not
        self.val_eval_criterion_alpha = 0.8  # alpha * old + (1-alpha) * new
        # if this is too low then the moving average will be too noisy and the training may terminate early. If it is
        # too high the training will take forever
        self.monitor_MA_alpha = 0.8  # alpha * old + (1-alpha) * new

    def initialize_model(self, data_channel, n_classes, learning_rate,
                         optim_dict, lr_scheduler_dict, device, fine_tune=None, model=None):

        self.model = get_model(self.model_name, data_channel, n_classes).to(device)

        if device == torch.device('cuda') and torch.cuda.device_count() > 1:
            print('we are using multiple gpus!!!')
            self.model = torch.nn.DataParallel(self.model)

        if os.path.exists(fine_tune['pre_trained_model']):
            pass

        if model == 'resnet-siim':
            # freeze the feature encoding part and train the combiner & classifier from scratch
            print('we are training the last part of the model from scratch!!!')

            try:
                print('we are training all layers')
                self.optimizer = self.optimizer([{'params': self.model.parameters()}],
                                                learning_rate, weight_decay=optim_dict['weight_decay'])
            except AttributeError:
                print('we are only training classifier of cnn-diff')
                self.optimizer = self.optimizer([{'params': self.model.module.parameters()}
                                                 ],
                                                learning_rate, weight_decay=optim_dict['weight_decay'])

        elif model == 'efficient-siim':
            print('we are using efficient!!!')
            try:
                print('all CPU: for conv-diff we only train the classifier!!!')
                self.optimizer = self.optimizer([{'params': self.model.classifier.parameters()},
                                                 {'params': self.model.block0.parameters(), 'lr': 1e-5},
                                                 {'params': self.model.block1.parameters(), 'lr': 1e-5},
                                                 ],
                                                learning_rate, weight_decay=optim_dict['weight_decay'])
            except AttributeError:
                print('GPU: for conv-diff we only train the classifier!!!')
                self.optimizer = self.optimizer([{'params': self.model.module.classifier.parameters()},
                                                 # {'params': self.model.module.block0.parameters(), 'lr': 3e-5},
                                                 # {'params': self.model.module.block1.parameters(), 'lr': 3e-5},
                                                 ],
                                                learning_rate, weight_decay=optim_dict['weight_decay'])
        elif model == 'efficient-siimd':
            print('we are using efficient!!!')
            try:
                print('all CPU: for conv-diff we only train the classifier!!!')
                self.optimizer = self.optimizer([{'params': self.model.classifier.parameters()},
                                                 {'params': self.model.block0.parameters(), 'lr': 1e-5},
                                                 {'params': self.model.block1.parameters(), 'lr': 1e-5},
                                                 {'params': self.model.block2.parameters(), 'lr': 1e-5},
                                                 {'params': self.model.block3.parameters(), 'lr': 1e-5}
                                                 ],
                                                learning_rate, weight_decay=optim_dict['weight_decay'])
            except AttributeError:
                print('GPU: for conv-diff we only train the classifier!!!')
                self.optimizer = self.optimizer([{'params': self.model.module.classifier.parameters()},
                                                 {'params': self.model.module.block0.parameters(), 'lr': 1e-5},
                                                 {'params': self.model.module.block1.parameters(), 'lr': 1e-5},
                                                 {'params': self.model.module.block2.parameters(), 'lr': 1e-5},
                                                 {'params': self.model.module.block3.parameters(), 'lr': 1e-5}
                                                 ],
                                                    learning_rate, weight_decay=optim_dict['weight_decay'])
        else:
            raise ValueError

        self.patience = lr_scheduler_dict['patience']
        self.save_every = self.patience

        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=self.patience,
                                                    verbose=True, threshold=1.0e-3, threshold_mode='abs')

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        timestamp = time()
        dt_object = datetime.fromtimestamp(timestamp)

        if add_timestamp:
            args = ("%s:" % dt_object, *args)

        if self.log_file is None:
            timestamp = datetime.now()
            self.log_file = os.path.join(self.output_dir, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                         (timestamp.year, timestamp.month, timestamp.day, timestamp.hour,
                                          timestamp.minute,
                                          timestamp.second))

            with open(self.log_file, 'w') as f:
                f.write("Starting... \n")

        successful = False
        max_attempts = 5
        ctr = 0

        while not successful and ctr < max_attempts:
            try:
                with open(self.log_file, 'a+') as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
                sleep(0.5)
                ctr += 1

        if also_print_to_console:
            print(*args)

    def load_checkpoint(self, best_model=False):

        if os.path.isfile(os.path.join(self.output_dir, self.model_name + '_final.model')):
            model_file = os.path.join(self.output_dir, self.model_name + '_final.model')

        if os.path.isfile(os.path.join(self.output_dir, self.model_name + '_scheduled.model')):
            model_file = os.path.join(self.output_dir, self.model_name + '_scheduled.model')

        if best_model:
            if os.path.isfile(os.path.join(self.output_dir, self.model_name + '_best.model')):
                model_file = os.path.join(self.output_dir, self.model_name + '_best.model')

        try:
            return self.load_saved_model(model_file)
        except IOError:
            print('model file is not existing!!!')

    def load_saved_model(self, fname, train=True):

        self.print_to_log_file("loading checkpoint", fname, "train=", train)
        saved_model = torch.load(fname, map_location=self.device)

        # load model state dict
        try:
            self.model.load_state_dict(saved_model['model_state_dict'])  # maybe need to change
        except RuntimeError:
            model_state_dict = convert_state_dict(saved_model['model_state_dict'])
            self.model.load_state_dict(model_state_dict)

        # load optimizer and lr_scheduler
        if train:
            optimizer_state_dict = saved_model['optimizer_state_dict']
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)

            if self.lr_scheduler is not None and isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.load_state_dict(saved_model['lr_scheduler_state_dict'])

        # load other related training results
        self.epoch = saved_model['epoch']
        self.monitor_MA = saved_model['monitor_MA']
        self.val_eval_criterion_MA = saved_model['val_eval_criterion_MA']
        self.best_val_eval_criterion_MA = saved_model['best_val_eval_criterion_MA']
        self.best_epoch_based_on_monitor_MA = saved_model['best_epoch_based_on_monitor_MA']
        self.train_losses, self.train_eval_metrics, self.val_losses, self.val_eval_metrics = saved_model['plot_stuff']

    def plot_progress(self):
        try:
            font = {'weight': 'normal',
                    'size': 18}

            matplotlib.rc('font', **font)

            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            x_values = list(range(self.epoch))

            ax.plot(x_values, self.train_losses, color='b', ls='-', label="loss_train")
            ax.plot(x_values, self.val_losses, color='r', ls='-', label="loss_val, train=False")

            if len(self.val_eval_metrics['accuracy']) == len(self.val_losses):
                ax2.plot(x_values, self.val_eval_metrics['accuracy'], color='g', ls=':', label="val acc")
                ax2.plot(x_values, self.val_eval_metrics['auc'], color='k', ls=':', label="val auc")
                ax2.plot(x_values, self.train_eval_metrics['accuracy'], color='darkviolet', ls=':', label="train acc")
                ax2.plot(x_values, self.train_eval_metrics['auc'], color='coral', ls=':', label="train auc")
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax2.set_ylabel("val evaluation metric")
            ax.legend()
            ax2.legend(loc=9)

            fig.savefig(os.path.join(self.output_dir, "progress.png"))
            plt.close()

        except IOError:
            self.print_to_log_file("failed to plot: ", sys.exc_info())

    def save_checkpoint(self, file_name):
        start_time = time()

        # get model state dict
        # if self.device == torch.device('cuda'):
        #     model_state_dict = self.model.module.state_dict()
        # else:

        model_state_dict = self.model.state_dict()

        # get lr_scheduler dict
        if self.lr_scheduler is not None and isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            lr_sched_state_dict = self.lr_scheduler.state_dict()
        else:
            lr_sched_state_dict = None

        # get optimizer state dict
        optimizer_state_dict = self.optimizer.state_dict()

        self.print_to_log_file("saving checkpoint...")

        assert self.epoch == len(self.train_losses), "train epoch and losses has different length!!!"
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_sched_state_dict,
            'val_eval_criterion_MA': self.val_eval_criterion_MA,
            'best_val_eval_criterion_MA': self.best_val_eval_criterion_MA,
            'monitor_MA': self.monitor_MA,
            'best_epoch_based_on_monitor_MA': self.best_epoch_based_on_monitor_MA,
            'plot_stuff': (self.train_losses, self.train_eval_metrics, self.val_losses, self.val_eval_metrics)}, file_name)

        self.print_to_log_file("done, saving took %.2f seconds" % (time() - start_time))

        self.print_to_log_file("done")

    def manage_patience(self):
        # update patience
        continue_training = True
        if self.patience is not None:
            # if best_MA_tr_loss_for_patience and best_epoch_based_on_MA_tr_loss were not yet initialized,
            # initialize them
            if self.best_monitor_MA is None:
                self.best_monitor_MA = self.monitor_MA

            if self.best_epoch_based_on_monitor_MA is None:
                self.best_epoch_based_on_monitor_MA = self.epoch

            if self.best_val_eval_criterion_MA is None:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA

            # check if the current epoch is the best one according to moving average of validation criterion. If so
            # then save 'best' model
            # Do not use this for validation. This is intended for test set prediction only.
            self.print_to_log_file("current best_val_eval_criterion_MA is {:.3f}".format(self.best_val_eval_criterion_MA))
            self.print_to_log_file("current val_eval_criterion_MA is {:.3f}".format(self.val_eval_criterion_MA))

            if self.val_eval_criterion_MA >= self.best_val_eval_criterion_MA:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                self.print_to_log_file("saving best epoch checkpoint...")
                self.save_checkpoint(os.path.join(self.output_dir, self.model_name + "_best.model"))
        # Now see if the moving average of the train loss has improved. If yes then reset patience, else
            # increase patience
            if self.monitor_MA + self.monitor_MA_eps <= self.best_monitor_MA:
                self.best_monitor_MA = self.monitor_MA
                self.best_epoch_based_on_monitor_MA = self.epoch
                self.print_to_log_file("New best epoch (train auc MA): {:.3f}".format(self.best_monitor_MA))
            else:
                self.print_to_log_file("No improvement: current train_auc_MA {:.3f}, best: {:.3f}, eps is {:.3f}".format(
                                       self.monitor_MA, self.best_monitor_MA, self.monitor_MA_eps))

            # if patience has reached its maximum then finish training (provided lr is low enough)
            if self.epoch - self.best_epoch_based_on_monitor_MA > self.patience:
                if self.optimizer.param_groups[0]['lr'] > self.lr_threshold:
                    self.print_to_log_file("My patience ended, but I believe I need more time (lr > 1e-6)")
                    # self.best_epoch_based_on_monitor_MA = self.epoch - self.patience // 2
                else:
                    self.print_to_log_file("My patience ended")
                    continue_training = False
            else:
                self.print_to_log_file(
                    "Patience: %d/%d" % (self.epoch - self.best_epoch_based_on_monitor_MA, self.patience))

        return continue_training

    def update_val_criterion_MA(self):
        """
        If self.all_val_eval_metrics is unused (len=0) then we fall back to using -self.all_val_losses for the MA to determine early stopping
        (not a minimization, but a maximization of a metric and therefore the - in the latter case)
        :return:
        """
        if self.val_eval_criterion_MA is None:
            # if len(self.val_eval_metrics) == 0:
            # self.val_eval_criterion_MA = self.val_losses[-1]
            # else:
            self.val_eval_criterion_MA = self.val_eval_metrics['auc'][-1]
            # self.val_eval_criterion_MA = self.train_eval_metrics['auc'][-1]
        else:
            # if len(self.val_eval_metrics['auc']) == 0:
            # print('we are using val loss as eval criterion!!!')
            """
            We here use alpha * old - (1 - alpha) * new because new in this case is the vlaidation loss and lower 
            is better, so we need to negate it. 
            """
            # self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA + \
            #                              (1 - self.val_eval_criterion_alpha) * self.val_losses[-1]
            # else:
            print('we are using val metrics as eval criterion!!!')
            self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA + \
                                         (1 - self.val_eval_criterion_alpha) * self.val_eval_metrics['auc'][-1]
            # print('we are using train metrics as eval criterion!!!')
            # self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA + \
            #                              (1 - self.val_eval_criterion_alpha) * self.train_eval_metrics['auc'][-1]

    def update_monitor_MA(self):
        if self.monitor_MA is None:
            # self.monitor_MA = self.train_losses[-1]
            self.monitor_MA = self.val_losses[-1]
            # self.monitor_MA = self.val_eval_metrics['auc'][-1]
        else:
            self.monitor_MA = self.monitor_MA_alpha * self.monitor_MA + \
                              (1 - self.monitor_MA_alpha) * self.val_losses[-1]
            # self.monitor_MA = self.monitor_MA_alpha * self.monitor_MA + \
            #                   (1 - self.monitor_MA_alpha) * self.train_losses[-1]

    def maybe_update_lr(self):
        # maybe update learning rate
        if self.lr_scheduler is not None:
            assert isinstance(self.lr_scheduler, (optim.lr_scheduler.ReduceLROnPlateau, optim.lr_scheduler._LRScheduler))

            if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                print('we are using monitor as lr_scheduler input!!!')
                # lr scheduler is updated with moving average val loss. should be more robust
                self.lr_scheduler.step(self.monitor_MA)
            else:
                self.lr_scheduler.step(self.epoch)

        self.print_to_log_file("lr is now (scheduler) %s" % str(self.optimizer.param_groups[0]['lr']))

    def compute_metrics(self, prediction_results, train=True, multi_class=False):
        assert len(prediction_results['target']) == len(prediction_results['pred_label'])
        acc, auc, f1_score, precision, recall = calculate_metrics(prediction_results['pred_score'],
                                                                  prediction_results['pred_label'],
                                                                  prediction_results['target'], multi_class)
        # TODO: add more other metrics
        if train:
            self.train_eval_metrics['accuracy'].append(acc)
            self.train_eval_metrics['auc'].append(auc)
            self.train_eval_metrics['recall'].append(recall)
            self.train_eval_metrics['precision'].append(precision)
            self.train_eval_metrics['f1-score'].append(f1_score)
        else:
            self.val_eval_metrics['accuracy'].append(acc)
            self.val_eval_metrics['auc'].append(auc)
            self.val_eval_metrics['recall'].append(recall)
            self.val_eval_metrics['precision'].append(precision)
            self.val_eval_metrics['f1-score'].append(f1_score)

    def run_on_epoch_end(self):
        self.plot_progress()

        if self.epoch % self.save_every == 0:
            self.print_to_log_file("saving scheduled checkpoint file...")
            file_name = os.path.join(self.output_dir, self.model_name + "_scheduled.model")
            self.save_checkpoint(file_name)

        self.update_monitor_MA()
        self.update_val_criterion_MA()

        self.maybe_update_lr()
        continue_trining = self.manage_patience()

        return continue_trining
