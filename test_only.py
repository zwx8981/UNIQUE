import os
import time
import scipy.stats
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

from ImageDataset3 import ImageDataset
from ListLoss import ListLoss_CE

from BaseCNN import BaseCNN
from BaseCNN2 import BaseCNN2
from DBCNN import DBCNN
from tqdm import tqdm
from MNL_Loss import Fidelity_Loss, Std_Loss

from scipy.optimize import curve_fit

# from E2euiqa import E2EUIQA
# from MNL_Loss import L2_Loss, Binary_Loss
# from Gdn import Gdn2d, Gdn1d

from Transformers import AdaptiveResize


class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)

        self.config = config

        self.train_transform = transforms.Compose([
            # transforms.RandomRotation(3),
            AdaptiveResize(512),
            transforms.RandomCrop(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        self.test_transform = transforms.Compose([
            AdaptiveResize(768),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        self.train_batch_size = config.batch_size
        self.test_batch_size = 1

        self.ranking = config.ranking

        self.train_data = ImageDataset(
            csv_file=os.path.join(config.trainset, 'splits2', str(config.split), config.train_txt),
            img_dir=config.trainset,
            transform=self.train_transform,
            test=(not config.ranking))
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=self.train_batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=12)

        # testing set configuration
        self.live_data = ImageDataset(
            csv_file=os.path.join(config.live_set, 'splits2', str(config.split), 'live_test.txt'),
            img_dir=config.live_set,
            transform=self.test_transform,
            test=True)

        self.live_loader = DataLoader(self.live_data,
                                      batch_size=self.test_batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=1)

        self.csiq_data = ImageDataset(
            csv_file=os.path.join(config.csiq_set, 'splits2', str(config.split), 'csiq_test.txt'),
            img_dir=config.csiq_set,
            transform=self.test_transform,
            test=True)

        self.csiq_loader = DataLoader(self.csiq_data,
                                      batch_size=self.test_batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=1)

        self.tid2013_data = ImageDataset(
            csv_file=os.path.join(config.tid2013_set, 'splits2', str(config.split), 'tid_test.txt'),
            img_dir=config.tid2013_set,
            transform=self.test_transform,
            test=True)

        self.tid2013_loader = DataLoader(self.tid2013_data,
                                         batch_size=self.test_batch_size,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=1)

        self.kadid10k_data = ImageDataset(
            csv_file=os.path.join(config.kadid10k_set, 'splits2', str(config.split), 'kadid10k_test.txt'),
            img_dir=config.kadid10k_set,
            transform=self.test_transform,
            test=True)

        self.kadid10k_loader = DataLoader(self.kadid10k_data,
                                          batch_size=self.test_batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=1)

        self.bid_data = ImageDataset(
            csv_file=os.path.join(config.bid_set, 'splits2', str(config.split), 'bid_test.txt'),
            img_dir=config.bid_set,
            transform=self.test_transform,
            test=True)

        self.bid_loader = DataLoader(self.bid_data,
                                     batch_size=self.test_batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=1)

        # self.cid_data = ImageDataset(csv_file=os.path.join(config.cid_set, 'cid_test.txt'),
        #                             img_dir=config.cid_set,
        #                             transform=self.test_transform,
        #                             test=True)

        # self.cid_loader = DataLoader(self.cid_data,
        #                             batch_size=self.test_batch_size,
        #                             shuffle=False,
        #                             pin_memory=True,
        #                             num_workers=1)

        self.clive_data = ImageDataset(
            csv_file=os.path.join(config.clive_set, 'splits2', str(config.split), 'clive_test.txt'),
            img_dir=config.clive_set,
            transform=self.test_transform,
            test=True)

        self.clive_loader = DataLoader(self.clive_data,
                                       batch_size=self.test_batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       num_workers=1)

        self.koniq10k_data = ImageDataset(
            csv_file=os.path.join(config.koniq10k_set, 'splits2', str(config.split), 'koniq10k_test.txt'),
            img_dir=config.koniq10k_set,
            transform=self.test_transform,
            test=True)

        self.koniq10k_loader = DataLoader(self.koniq10k_data,
                                          batch_size=self.test_batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=1)

        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")

        # initialize the model
        # self.model = E2EUIQA()
        if config.network == 'basecnn':
            if config.std_modeling:
                self.model = BaseCNN2(config)
                self.model = nn.DataParallel(self.model, device_ids=[0])
            else:
                # self.model = BaseCNN(config)
                self.model = BaseCNN(config)
                self.model = nn.DataParallel(self.model).cuda()
        elif config.network == 'dbcnn':
            self.model = DBCNN(config)
            self.model = nn.DataParallel(self.model).cuda()
        else:
            raise NotImplementedError("Not supported network!")
        self.model.to(self.device)
        self.model_name = type(self.model).__name__
        print(self.model)

        # oracle's log variance
        # self.oracle_num = config.oracle_num
        # self.sensitivity = Variable(torch.rand(1, self.oracle_num, device=self.device), requires_grad=True)
        # self.specificity = Variable(torch.rand(1, self.oracle_num, device=self.device), requires_grad=True)

        # loss function
        # self.loss_fn = ListLoss_CE(k=2)
        if config.ranking:
            if config.fidelity:
                self.loss_fn = Fidelity_Loss()
            else:
                self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.MSELoss()
        self.loss_fn.to(self.device)

        if self.config.std_modeling:
            # self.std_loss_fn = Std_Loss(input_var=True)
            self.std_loss_fn = nn.BCEWithLogitsLoss()
            self.std_loss_fn.to(self.device)

        self.initial_lr = config.lr
        if self.initial_lr is None:
            lr = 0.0005
        else:
            lr = self.initial_lr

        # self.optimizer = optim.Adam([{'params': self.model.parameters(), 'lr': lr},
        #                             {'params': self.sensitivity, 'lr': 1e-3},
        #                             {'params': self.specificity, 'lr': 1e-3}]
        #                            )
        # self.optimizer = optim.Adam([{'params': self.model.backbone.parameters(), 'lr': lr},
        #                             {'params': self.model.fc.parameters(), 'lr': 10 * lr}],
        #                            lr=lr, weight_decay=5e-4
        #                            )
        # self.optimizer = optim.SGD([{'params': self.model.backbone.parameters(), 'lr': lr},
        #                             {'params': self.model.fc.parameters(), 'lr': lr}],
        #                            lr=lr, weight_decay=5e-4, momentum=0.9
        #                            )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr,
            weight_decay=5e-4)

        # some states
        self.start_epoch = 0
        self.start_step = 0
        self.train_loss = []
        self.test_results_srcc = {'live': [], 'csiq': [], 'tid2013': [], 'kadid10k': [], 'bid': [], 'clive': [],
                                  'koniq10k': []}
        self.test_results_plcc = {'live': [], 'csiq': [], 'tid2013': [], 'kadid10k': [], 'bid': [], 'clive': [],
                                  'koniq10k': []}
        self.ckpt_path = config.ckpt_path
        self.max_epochs = config.max_epochs
        self.epochs_per_eval = config.epochs_per_eval
        self.epochs_per_save = config.epochs_per_save

        # try load the model
        if config.resume or not config.train:
            if config.ckpt:
                ckpt = os.path.join(config.ckpt_path, config.ckpt)
            else:
                ckpt = self._get_latest_checkpoint(path=config.ckpt_path)
            self._load_checkpoint(ckpt=ckpt)

        self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                             last_epoch=self.start_epoch - 1,
                                             step_size=config.decay_interval,
                                             gamma=config.decay_ratio)

    def fit(self):
        if self.ranking:
            for epoch in range(self.start_epoch, self.max_epochs):
                _ = self._train_single_epoch(epoch)
                # self.scheduler.step()
        else:
            for epoch in range(self.start_epoch, self.max_epochs):
                _ = self._train_single_epoch_regression(epoch)
                # self.scheduler.step()

    def _train_single_epoch(self, epoch):
        # initialize logging system
        num_steps_per_epoch = len(self.train_loader)
        local_counter = epoch * num_steps_per_epoch + 1
        start_time = time.time()
        beta = 0.9
        running_loss = 0 if epoch == 0 else self.train_loss[-1]
        loss_corrected = 0.0
        running_duration = 0.0

        # start training
        print('Adam learning rate: {:.8f}'.format(self.optimizer.param_groups[0]['lr']))
        self.model.train()
        self.scheduler.step()
        for step, sample_batched in enumerate(self.train_loader, 0):

            if step < self.start_step:
                continue

            x1, x2, g, gstd1, gstd2 = sample_batched['I1'], sample_batched['I2'], sample_batched['y'], sample_batched[
                'std1'], sample_batched['std2']
            x1 = Variable(x1)
            x2 = Variable(x2)
            g = Variable(g).view(-1, 1)
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            g = g.to(self.device)

            gstd1 = gstd1.to(self.device)
            gstd2 = gstd2.to(self.device)
            # g = torch.cat((g[:, :6], g[:, 7:9], g[:, 10].unsqueeze(-1)), dim=-1)

            self.optimizer.zero_grad()
            if self.config.std_modeling:
                y1, y1_var = self.model(x1)
                y2, y2_var = self.model(x2)
                y_diff = y1 - y2
                y_var = y1_var + y2_var + 1e-8
                p = 0.5 * (1 + torch.erf(y_diff / torch.sqrt(2 * y_var)))
                # self.std_loss = self.std_loss_fn(y1_var, y2_var, gstd1.detach(), gstd2.detach())
                std_label = 0.5 * (torch.sign((gstd1 - gstd2)) + 1)
                self.std_loss = self.std_loss_fn((torch.sqrt(y1_var) - torch.sqrt(y2_var)), std_label.detach())
            else:
                y1 = self.model(x1)
                y2 = self.model(x2)
                y_diff = y1 - y2
                p = y_diff

            # p = F.sigmoid(y_diff)
            # p = torch.exp(y_diff) / (1 + torch.exp(y_diff))
            # p = (y_diff > 0)

            self.loss = self.loss_fn(p, g.detach())
            if self.config.std_loss:
                self.loss += 0.01 * self.std_loss
            self.loss.backward()
            self.optimizer.step()
            # self._gdn_param_proc()

            # statistics
            running_loss = beta * running_loss + (1 - beta) * self.loss.data.item()
            loss_corrected = running_loss / (1 - beta ** local_counter)

            current_time = time.time()
            duration = current_time - start_time
            running_duration = beta * running_duration + (1 - beta) * duration
            duration_corrected = running_duration / (1 - beta ** local_counter)
            examples_per_sec = self.train_batch_size / duration_corrected
            format_str = ('(E:%d, S:%d / %d) [Loss = %.4f] (%.1f samples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (epoch, step, num_steps_per_epoch, loss_corrected,
                                examples_per_sec, duration_corrected))

            local_counter += 1
            self.start_step = 0
            start_time = time.time()

        self.train_loss.append(loss_corrected)

        if (epoch + 1) % self.epochs_per_eval == 0:
            # evaluate after every other epoch
            test_results_srcc, test_results_plcc = self.eval()
            self.test_results_srcc['live'].append(test_results_srcc['live'])
            self.test_results_srcc['csiq'].append(test_results_srcc['csiq'])
            self.test_results_srcc['tid2013'].append(test_results_srcc['tid2013'])
            self.test_results_srcc['kadid10k'].append(test_results_srcc['kadid10k'])
            self.test_results_srcc['bid'].append(test_results_srcc['bid'])
            # self.test_results['cid'].append(test_results['cid'])
            self.test_results_srcc['clive'].append(test_results_srcc['clive'])
            self.test_results_srcc['koniq10k'].append(test_results_srcc['koniq10k'])

            self.test_results_plcc['live'].append(test_results_plcc['live'])
            self.test_results_plcc['csiq'].append(test_results_plcc['csiq'])
            self.test_results_plcc['tid2013'].append(test_results_plcc['tid2013'])
            self.test_results_plcc['kadid10k'].append(test_results_plcc['kadid10k'])
            self.test_results_plcc['bid'].append(test_results_plcc['bid'])
            # self.test_results_plcc['cid'].append(test_results_plcc['cid'])
            self.test_results_plcc['clive'].append(test_results_plcc['clive'])
            self.test_results_plcc['koniq10k'].append(test_results_plcc['koniq10k'])

            # out_str = 'Epoch {} Testing: LIVE SRCC: {:.4f}  CSIQ SRCC: {:.4f} TID2013 SRCC:' \
            #          ' {:.4f} BID SRCC: {:.4f} CID SRCC: {:.4f} CLIVE SRCC: {:.4f}'.format(epoch, test_results['live'],
            #                                                                                test_results['csiq'],
            #                                                                                test_results['tid2013'],
            #                                                                                test_results['bid'],
            #                                                                                #test_results['cid'],
            #                                                                                test_results['clive'],
            #                                                                                test_results['koniq10k'])
            out_str = 'Testing: LIVE SRCC: {:.4f}  CSIQ SRCC: {:.4f} TID2013 SRCC: {:.4f} KADID10K SRCC: {:.4f} ' \
                      'BID SRCC: {:.4f} CLIVE SRCC: {:.4f}  KONIQ10K SRCC: {:.4f}'.format(
                test_results_srcc['live'],
                test_results_srcc['csiq'],
                test_results_srcc['tid2013'],
                test_results_srcc['kadid10k'],
                test_results_srcc['bid'],
                # test_results_srcc['cid'],
                test_results_srcc['clive'],
                test_results_srcc['koniq10k'])
            out_str2 = 'Testing: LIVE PLCC: {:.4f}  CSIQ PLCC: {:.4f} TID2013 PLCC: {:.4f} KADID10K PLCC: {:.4f} ' \
                       'BID PLCC: {:.4f} CLIVE PLCC: {:.4f}  KONIQ10K PLCC: {:.4f}'.format(
                test_results_plcc['live'],
                test_results_plcc['csiq'],
                test_results_plcc['tid2013'],
                test_results_plcc['kadid10k'],
                test_results_plcc['bid'],
                # test_results_plcc['cid'],
                test_results_plcc['clive'],
                test_results_plcc['koniq10k'])
            print(out_str)
            print(out_str2)

        if (epoch + 1) % self.epochs_per_save == 0:
            model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch)
            model_name = os.path.join(self.ckpt_path, model_name)
            self._save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'train_loss': self.train_loss,
                'test_results_srcc': self.test_results_srcc,
                'test_results_plcc': self.test_results_plcc,
            }, model_name)

        return self.loss.data.item()

    def _train_single_epoch_regression(self, epoch):
        # initialize logging system
        num_steps_per_epoch = len(self.train_loader)
        local_counter = epoch * num_steps_per_epoch + 1
        start_time = time.time()
        beta = 0.9
        running_loss = 0 if epoch == 0 else self.train_loss[-1]
        loss_corrected = 0.0
        running_duration = 0.0

        # start training
        self.scheduler.step()
        print('Adam learning rate: {:.8f}'.format(self.optimizer.param_groups[0]['lr']))
        self.model.train()
        for step, sample_batched in enumerate(self.train_loader, 0):

            if step < self.start_step:
                continue

            x, g = sample_batched['I'], sample_batched['mos']
            x = Variable(x)
            g = Variable(g).view(-1, 1)
            x = x.to(self.device)
            g = g.to(self.device)

            self.optimizer.zero_grad()
            y = self.model(x)

            self.loss = self.loss_fn(y, g.float().detach())
            self.loss.backward()
            self.optimizer.step()

            # statistics
            running_loss = beta * running_loss + (1 - beta) * self.loss.data.item()
            loss_corrected = running_loss / (1 - beta ** local_counter)

            current_time = time.time()
            duration = current_time - start_time
            running_duration = beta * running_duration + (1 - beta) * duration
            duration_corrected = running_duration / (1 - beta ** local_counter)
            examples_per_sec = self.train_batch_size / duration_corrected
            format_str = ('(E:%d, S:%d / %d) [Loss = %.4f] (%.1f samples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (epoch, step, num_steps_per_epoch, loss_corrected,
                                examples_per_sec, duration_corrected))

            local_counter += 1
            self.start_step = 0
            start_time = time.time()

        self.train_loss.append(loss_corrected)

        if (epoch + 1) % self.epochs_per_eval == 0:
            # evaluate after every other epoch
            test_results_srcc, test_results_plcc = self.eval()
            self.test_results_srcc['live'].append(test_results_srcc['live'])
            self.test_results_srcc['csiq'].append(test_results_srcc['csiq'])
            self.test_results_srcc['tid2013'].append(test_results_srcc['tid2013'])
            self.test_results_srcc['tid2013'].append(test_results_srcc['kadid10k'])
            self.test_results_srcc['bid'].append(test_results_srcc['bid'])
            # self.test_results['cid'].append(test_results['cid'])
            self.test_results_srcc['clive'].append(test_results_srcc['clive'])
            self.test_results_srcc['koniq10k'].append(test_results_srcc['koniq10k'])

            self.test_results_plcc['live'].append(test_results_plcc['live'])
            self.test_results_plcc['csiq'].append(test_results_plcc['csiq'])
            self.test_results_plcc['tid2013'].append(test_results_plcc['tid2013'])
            self.test_results_plcc['kadid10k'].append(test_results_plcc['kadid10k'])
            self.test_results_plcc['bid'].append(test_results_plcc['bid'])
            # self.test_results_plcc['cid'].append(test_results_plcc['cid'])
            self.test_results_plcc['clive'].append(test_results_plcc['clive'])
            self.test_results_plcc['koniq10k'].append(test_results_plcc['koniq10k'])

            # out_str = 'Epoch {} Testing: LIVE SRCC: {:.4f}  CSIQ SRCC: {:.4f} TID2013 SRCC:' \
            #          ' {:.4f} BID SRCC: {:.4f} CID SRCC: {:.4f} CLIVE SRCC: {:.4f}'.format(epoch, test_results['live'],
            #                                                                                test_results['csiq'],
            #                                                                                test_results['tid2013'],
            #                                                                                test_results['bid'],
            #                                                                                #test_results['cid'],
            #                                                                                test_results['clive'],
            #                                                                                test_results['koniq10k'])
            out_str = 'Testing: LIVE SRCC: {:.4f}  CSIQ SRCC: {:.4f} TID2013 SRCC: {:.4f} KADID10K SRCC: {:.4f} ' \
                      'BID SRCC: {:.4f} CLIVE SRCC: {:.4f}  KONIQ10K SRCC: {:.4f}'.format(
                test_results_srcc['live'],
                test_results_srcc['csiq'],
                test_results_srcc['tid2013'],
                test_results_srcc['kadid10k'],
                test_results_srcc['bid'],
                # test_results_srcc['cid'],
                test_results_srcc['clive'],
                test_results_srcc['koniq10k'])
            out_str2 = 'Testing: LIVE PLCC: {:.4f}  CSIQ PLCC: {:.4f} TID2013 PLCC: {:.4f} KADID10K PLCC: {:.4f} ' \
                       'BID PLCC: {:.4f} CLIVE PLCC: {:.4f}  KONIQ10K PLCC: {:.4f}'.format(
                test_results_plcc['live'],
                test_results_plcc['csiq'],
                test_results_plcc['tid2013'],
                test_results_plcc['kadid10k'],
                test_results_plcc['bid'],
                # test_results_plcc['cid'],
                test_results_plcc['clive'],
                test_results_plcc['koniq10k'])
            print(out_str)
            print(out_str2)

        if (epoch + 1) % self.epochs_per_save == 0:
            model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch)
            model_name = os.path.join(self.ckpt_path, model_name)
            self._save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'train_loss': self.train_loss,
                'test_results_srcc': self.test_results_srcc,
                'test_results_plcc': self.test_results_plcc,
            }, model_name)

        return self.loss.data.item()

    def logistic_fit(self, X):
        bayta1 = 10.0
        bayta2 = 0
        bayta3 = np.mean(X)
        bayta4 = 0.1
        bayta5 = 0.1

        logisticPart = 0.5 - 1 / (1 + np.exp(bayta2 * (X - bayta3)))

        yhat = bayta1 * logisticPart + bayta4 * X + bayta5

        return yhat

    def eval(self):
        q_mos = []
        q_hat = []
        srcc = {}
        plcc = {}
        self.model.eval()
        for step, sample_batched in enumerate(self.live_loader, 0):
            x, y = sample_batched['I'], sample_batched['mos']
            x = Variable(x)
            x = x.to(self.device)

            if self.config.std_modeling:
                y_bar, _ = self.model(x)
            else:
                y_bar = self.model(x)
            y_bar.cpu()
            q_mos.append(y.data.numpy())
            q_hat.append(y_bar.cpu().data.numpy())

        # q_hat = np.squeeze(np.array(q_hat),2)
        # q_mos = np.array(q_mos)
        # print(np.shape(q_hat))
        # print(np.shape(q_mos))
        # popt, pcov = curve_fit(self.logistic_fit, q_hat, q_mos)
        # q_hat = self.logistic_fit(np.array(q_hat), *popt)

        srcc['live'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
        plcc['live'] = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat)[0]

        q_mos = []
        q_hat = []
        for step, sample_batched in enumerate(self.csiq_loader, 0):
            x, y = sample_batched['I'], sample_batched['mos']
            x = Variable(x)
            x = x.to(self.device)

            if self.config.std_modeling:
                y_bar, _ = self.model(x)
            else:
                y_bar = self.model(x)
            y_bar.cpu()
            q_mos.append(y.data.numpy())
            q_hat.append(y_bar.cpu().data.numpy())

        srcc['csiq'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
        plcc['csiq'] = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat)[0]

        q_mos = []
        q_hat = []
        for step, sample_batched in enumerate(self.tid2013_loader, 0):
            x, y = sample_batched['I'], sample_batched['mos']
            x = Variable(x)
            x = x.to(self.device)

            if self.config.std_modeling:
                y_bar, _ = self.model(x)
            else:
                y_bar = self.model(x)
            y_bar.cpu()
            q_mos.append(y.data.numpy())
            q_hat.append(y_bar.cpu().data.numpy())

        srcc['tid2013'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
        plcc['tid2013'] = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat)[0]

        q_mos = []
        q_hat = []
        for step, sample_batched in enumerate(self.kadid10k_loader, 0):
            x, y = sample_batched['I'], sample_batched['mos']
            x = Variable(x)
            x = x.to(self.device)

            if self.config.std_modeling:
                y_bar, _ = self.model(x)
            else:
                y_bar = self.model(x)
            y_bar.cpu()
            q_mos.append(y.data.numpy())
            q_hat.append(y_bar.cpu().data.numpy())

        srcc['kadid10k'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
        plcc['kadid10k'] = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat)[0]

        q_mos = []
        q_hat = []
        for step, sample_batched in enumerate(self.bid_loader, 0):
            x, y = sample_batched['I'], sample_batched['mos']
            x = Variable(x)
            x = x.to(self.device)

            if self.config.std_modeling:
                y_bar, _ = self.model(x)
            else:
                y_bar = self.model(x)
            y_bar.cpu()
            q_mos.append(y.data.numpy())
            q_hat.append(y_bar.cpu().data.numpy())

        srcc['bid'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
        plcc['bid'] = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat)[0]

        # q_mos = []
        # q_hat = []
        # for step, sample_batched in enumerate(self.cid_loader, 0):
        #    x, y = sample_batched['I'], sample_batched['mos']
        #    x = Variable(x)
        #    x = x.to(self.device)

        #   y_bar = self.model(x)
        #    y_bar.cpu()
        #    q_mos.append(y.data.numpy())
        #    q_hat.append(y_bar.cpu().data.numpy())

        # srcc['cid'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]

        q_mos = []
        q_hat = []
        for step, sample_batched in enumerate(self.clive_loader, 0):
            x, y = sample_batched['I'], sample_batched['mos']
            x = Variable(x)
            x = x.to(self.device)

            if self.config.std_modeling:
                y_bar, _ = self.model(x)
            else:
                y_bar = self.model(x)
            y_bar.cpu()
            q_mos.append(y.data.numpy())
            q_hat.append(y_bar.cpu().data.numpy())

        srcc['clive'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
        plcc['clive'] = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat)[0]

        q_mos = []
        q_hat = []
        for step, sample_batched in enumerate(self.knoiq10k_loader, 0):
            x, y = sample_batched['I'], sample_batched['mos']
            x = Variable(x)
            x = x.to(self.device)

            if self.config.std_modeling:
                y_bar, _ = self.model(x)
            else:
                y_bar = self.model(x)
            y_bar.cpu()
            q_mos.append(y.data.numpy())
            q_hat.append(y_bar.cpu().data.numpy())

        srcc['koniq10k'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
        plcc['koniq10k'] = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat)[0]

        return srcc, plcc

    def get_scores(self):
        q_mos = []
        q_hat = []
        q_std = []
        q_pstd = []
        all_mos = {}
        all_hat = {}
        all_std = {}
        all_pstd = {}
        self.model.eval()

        if self.config.eval_live:
            for step, sample_batched in enumerate(self.live_loader, 0):
                x, y, std = sample_batched['I'], sample_batched['mos'], sample_batched['std']
                x = Variable(x)
                x = x.to(self.device)

                if self.config.std_modeling:
                    y_bar, var = self.model(x)
                    q_std.append(std.data.numpy())
                    q_pstd.append(torch.sqrt(var).cpu().data.numpy())
                else:
                    y_bar = self.model(x)
                y_bar.cpu()
                q_mos.append(y.data.numpy())
                q_hat.append(y_bar.cpu().data.numpy())

            all_mos['live'] = q_mos
            all_hat['live'] = q_hat
            all_std['live'] = q_std
            all_pstd['live'] = q_pstd
        else:
            all_mos['live'] = 0
            all_hat['live'] = 0
            all_std['live'] = 0
            all_pstd['live'] = 0



        q_mos = []
        q_hat = []
        q_std = []
        q_pstd = []
        if self.config.eval_csiq:
            for step, sample_batched in enumerate(self.csiq_loader, 0):
                x, y, std = sample_batched['I'], sample_batched['mos'], sample_batched['std']
                x = Variable(x)
                x = x.to(self.device)

                if self.config.std_modeling:
                    y_bar, var = self.model(x)
                    q_std.append(std.data.numpy())
                    q_pstd.append(torch.sqrt(var).cpu().data.numpy())
                else:
                    y_bar = self.model(x)
                y_bar.cpu()
                q_mos.append(y.data.numpy())
                q_hat.append(y_bar.cpu().data.numpy())

            all_mos['csiq'] = q_mos
            all_hat['csiq'] = q_hat
            all_std['csiq'] = q_std
            all_pstd['csiq'] = q_pstd
        else:
            all_mos['csiq'] = 0
            all_hat['csiq'] = 0
            all_std['csiq'] = 0
            all_pstd['csiq'] = 0




        q_mos = []
        q_hat = []
        q_std = []
        q_pstd = []
        if self.config.eval_tid2013:
            for step, sample_batched in enumerate(self.tid2013_loader, 0):
                x, y, std = sample_batched['I'], sample_batched['mos'], sample_batched['std']
                x = Variable(x)
                x = x.to(self.device)

                if self.config.std_modeling:
                    y_bar, var = self.model(x)
                    q_std.append(std.data.numpy())
                    q_pstd.append(torch.sqrt(var).cpu().data.numpy())
                else:
                    y_bar = self.model(x)
                y_bar.cpu()
                q_mos.append(y.data.numpy())
                q_hat.append(y_bar.cpu().data.numpy())

            all_mos['tid2013'] = q_mos
            all_hat['tid2013'] = q_hat
            all_std['tid2013'] = q_std
            all_pstd['tid2013'] = q_pstd
        else:
            all_mos['tid2013'] = 0
            all_hat['tid2013'] = 0
            all_std['tid2013'] = 0
            all_pstd['tid2013'] = 0


        q_mos = []
        q_hat = []
        q_std = []
        q_pstd = []

        if self.config.eval_kadid10k:
            for step, sample_batched in enumerate(self.kadid10k_loader, 0):
                x, y, std = sample_batched['I'], sample_batched['mos'], sample_batched['std']
                x = Variable(x)
                x = x.to(self.device)

                if self.config.std_modeling:
                    y_bar, var = self.model(x)
                    q_std.append(std.data.numpy())
                    q_pstd.append(torch.sqrt(var).cpu().data.numpy())
                else:
                    y_bar = self.model(x)
                y_bar.cpu()
                q_mos.append(y.data.numpy())
                q_hat.append(y_bar.cpu().data.numpy())

            all_mos['kadid10k'] = q_mos
            all_hat['kadid10k'] = q_hat
            all_std['kadid10k'] = q_std
            all_pstd['kadid10k'] = q_pstd
        else:
            all_mos['kadid10k'] = 0
            all_hat['kadid10k'] = 0
            all_std['kadid10k'] = 0
            all_pstd['kadid10k'] = 0

        q_mos = []
        q_hat = []
        q_std = []
        q_pstd = []

        if self.config.eval_bid:
            for step, sample_batched in enumerate(self.bid_loader, 0):
                x, y, std = sample_batched['I'], sample_batched['mos'], sample_batched['std']
                x = Variable(x)
                x = x.to(self.device)

                if self.config.std_modeling:
                    y_bar, var = self.model(x)
                    q_std.append(std.data.numpy())
                    q_pstd.append(torch.sqrt(var).cpu().data.numpy())
                else:
                    y_bar = self.model(x)
                y_bar.cpu()
                q_mos.append(y.data.numpy())
                q_hat.append(y_bar.cpu().data.numpy())

            all_mos['bid'] = q_mos
            all_hat['bid'] = q_hat
            all_std['bid'] = q_std
            all_pstd['bid'] = q_pstd
        else:
            all_mos['bid'] = 0
            all_hat['bid'] = 0
            all_std['bid'] = 0
            all_pstd['bid'] = 0



        q_mos = []
        q_hat = []
        q_std = []
        q_pstd = []
        if self.config.eval_clive:
            for step, sample_batched in enumerate(self.clive_loader, 0):
                x, y, std = sample_batched['I'], sample_batched['mos'], sample_batched['std']
                x = Variable(x)
                x = x.to(self.device)

                if self.config.std_modeling:
                    y_bar, var = self.model(x)
                    q_std.append(std.data.numpy())
                    #q_pstd.append(torch.sqrt(var).cpu().data.numpy())
                    q_pstd.append(var.cpu().data.numpy())
                else:
                    y_bar = self.model(x)
                y_bar.cpu()
                q_mos.append(y.data.numpy())
                q_hat.append(y_bar.cpu().data.numpy())

            all_mos['clive'] = q_mos
            all_hat['clive'] = q_hat
            all_std['clive'] = q_std
            all_pstd['clive'] = q_pstd
        else:
            all_mos['clive'] = 0
            all_hat['clive'] = 0
            all_std['clive'] = 0
            all_pstd['clive'] = 0

        q_mos = []
        q_hat = []
        q_std = []
        q_pstd = []
        if self.config.eval_koniq10k:
            for step, sample_batched in enumerate(self.koniq10k_loader, 0):
                x, y, std = sample_batched['I'], sample_batched['mos'], sample_batched['std']
                x = Variable(x)
                x = x.to(self.device)

                if self.config.std_modeling:
                    y_bar, var = self.model(x)
                    q_std.append(std.data.numpy())
                    q_pstd.append(torch.sqrt(var).cpu().data.numpy())
                else:
                    y_bar = self.model(x)
                y_bar.cpu()
                q_mos.append(y.data.numpy())
                q_hat.append(y_bar.cpu().data.numpy())

            all_mos['koniq10k'] = q_mos
            all_hat['koniq10k'] = q_hat
            all_std['koniq10k'] = q_std
            all_pstd['koniq10k'] = q_pstd
        else:
            all_mos['koniq10k'] = 0
            all_hat['koniq10k'] = 0
            all_std['koniq10k'] = 0
            all_pstd['koniq10k'] = 0


        return all_mos, all_hat, all_std, all_pstd

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            # self.sensitivity = checkpoint['sensitivity']
            # self.specificity = checkpoint['specificity']
            self.start_epoch = checkpoint['epoch'] + 1
            self.train_loss = checkpoint['train_loss']
            self.test_results_srcc = checkpoint['test_results_srcc']
            self.test_results_plcc = checkpoint['test_results_plcc']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.initial_lr is not None:
                for param_group in self.optimizer.param_groups:
                    param_group['initial_lr'] = self.initial_lr
            print("[*] loaded checkpoint '{}' (epoch {})"
                  .format(ckpt, checkpoint['epoch']))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt))

    @staticmethod
    def _get_latest_checkpoint(path):
        ckpts = os.listdir(path)
        ckpts = [ckpt for ckpt in ckpts if not os.path.isdir(os.path.join(path, ckpt))]
        all_times = sorted(ckpts, reverse=True)
        return os.path.join(path, all_times[0])

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename='checkpoint.pth.tar'):
        # if os.path.exists(filename):
        #     shutil.rmtree(filename)
        torch.save(state, filename)

