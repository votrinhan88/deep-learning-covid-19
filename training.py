import torch
import torch.optim
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import time, datetime
import numpy as np

import logging
import modules.logconf

fix = 220

class TrainingClass:
    def __init__(self, project_path, num_workers, criterion, num_epochs, threshold, stop_metric, strip_len, cooldown, patience, latency, GL_stop, NGL_stop, SI_stop):
        # Init logger
        self.project_path = project_path
        self.logger = logging.getLogger(__name__)
        modules.logconf.initLogger(self.logger, project_path = self.project_path)
        
        # Handle parameters
        # todo: different learning_rate for OneCycleLR
        self.num_workers = num_workers
        self.criterion   = criterion
        self.num_epochs  = num_epochs
        self.threshold   = threshold
        
        # Stopping conditions
        self.stop_metric = stop_metric
        self.strip_len   = strip_len
        self.cooldown    = cooldown
        self.patience    = patience
        self.latency     = latency
        self.GL_stop     = GL_stop
        self.NGL_stop    = NGL_stop
        self.SI_stop     = SI_stop
        
        self.logger.info(f'Non-tunable hyperparameters: num_workers = {self.num_workers}, criterion = {self.criterion}, num_epochs = {self.num_epochs}, threshold = {self.threshold}')
        self.logger.info(f'Stopping conditions: stop_metric = {self.stop_metric}, strip_len = {self.strip_len}, cooldown = {self.cooldown}, patience = {self.patience}, latency = {self.latency}, GL_stop = {self.GL_stop}, NGL_stop = {self.NGL_stop}, SI_stop = {self.SI_stop}')
        
        self.generator = torch.Generator()
        
        # Use CUDA
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.logger.info(f"Using {self.device} device.")
        
        self.subsets = ['train', 'val']
        self.metric_vars = ['confusion_matrix', 'loss', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auroc', 'roc', 'auprc', 'prc']
        self.stopping_vars = ['D_train', 'D_val', 'D_opt', 'TF', 'GL', 'epoch_GL', 'NGL', 'epoch_NGL', 'SI', 'epoch_SI']
        self.stopping_params = ['stop_metric', 'strip_len', 'cooldown', 'patience', 'latency', 'GL_stop', 'NGL_stop', 'SI_stop']
        self.tunable_hparam_names = ['batch_size', 'max_lr', 'ratio_lr', 'max_momentum', 'base_momentum', 'weight_decay']
        
        self.logger.debug(f'[CMPL] Initialized TrainingClass - fix {fix}')
    
    def loadTunedHyperparameters(self, batch_size, max_lr, ratio_lr, max_momentum, base_momentum, weight_decay):
        self.batch_size    = batch_size
        self.max_lr        = max_lr
        self.ratio_lr      = ratio_lr
        self.base_lr       = self.max_lr * self.ratio_lr
        self.max_momentum  = max_momentum
        self.base_momentum = base_momentum
        self.weight_decay  = weight_decay
        
        self.logger.info(f'Tuned hyperparameters: batch_size = {self.batch_size}, max_lr = {self.max_lr}, base_lr = {self.base_lr}, max_momentum = {self.max_momentum}, base_momentum = {self.base_momentum}, weight_decay = {self.weight_decay}')
    
    def worker_init_fn(self, worker_id):
        process_seed = torch.initial_seed()
        # Back out the base_seed so we can use all the bits.
        base_seed = process_seed - worker_id
        ss = np.random.SeedSequence([worker_id, base_seed])
        # More than 128 bits (4 32-bit words) would be overkill.
        np.random.seed(ss.generate_state(4))
    
    def initOptimizer(self, model):
        optimizer = torch.optim.SGD(params       = model.parameters(),
                                    lr           = self.max_lr,
                                    momentum     = self.max_momentum,
                                    weight_decay = self.weight_decay)
        # Or
        #optimizer = torch.optim.Adam(params = model.parameters())
        #optimizer = torch.optim.AdamW(params = model.parameters())
        self.logger.debug("[CMPL] Initialized SGD optimizer")
        return optimizer
    
    def training_loop(self, modelClass, model_name, train_set, val_set):
        self.loadTimer(model_name = model_name)
        model = modelClass.initModel(model_name = model_name)
        optimizer = self.initOptimizer(model = model)
        
        train_loader = torch.utils.data.DataLoader(dataset        = train_set,
                                                   batch_size     = self.batch_size,
                                                   shuffle        = True,
                                                   num_workers    = self.num_workers,
                                                   worker_init_fn = self.worker_init_fn,
                                                   generator      = self.generator)
        val_loader = torch.utils.data.DataLoader(dataset        = val_set,
                                                 batch_size     = self.batch_size,
                                                 shuffle        = True,
                                                 num_workers    = self.num_workers,
                                                 worker_init_fn = self.worker_init_fn,
                                                 generator      = self.generator)
        
        model, optimizer, cp_epoch = self.loadCheckpoint(model = model,
                                                         optimizer = optimizer)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer      = optimizer,
                                                      max_lr         = self.max_lr,
                                                      base_lr        = self.base_lr,
                                                      max_momentum   = self.max_momentum,
                                                      base_momentum  = self.base_momentum,
                                                      step_size_up   = len(train_loader) // 2,
                                                      step_size_down = len(train_loader) - len(train_loader) // 2)
        metrics, cp_epoch_2 = self.loadMetrics(model_name = model.name)
        stoppings, cp_epoch_3 = self.loadStoppings(model_name = model.name)

        # If not matched flags, reload metrics and stoppings to corresponding cp_epoch
        matched_flag, cp_epoch = self.checkMatchedCheckpoint(cp_epoch   = cp_epoch,
                                                             cp_epoch_2 = cp_epoch_2,
                                                             cp_epoch_3 = cp_epoch_3)
        if not matched_flag:
            self.logger.debug(f'[STTS] Reload metrics and stoppings at cp_epoch {cp_epoch}')
            metrics, cp_epoch_2 = self.loadMetrics(model_name = model.name,
                                                   cp_epoch   = cp_epoch)
            stoppings, cp_epoch_3 = self.loadStoppings(model_name = model.name,
                                                       cp_epoch   = cp_epoch)
            
        self.logger.info(f'[STTS] Start training with model {model.name} for {self.num_epochs} epochs from cp_epoch {cp_epoch}')
        
        epoch_range = torch.arange(self.num_epochs) + cp_epoch + 1
        num_epochs_tqdm = tqdm(epoch_range, unit = 'epoch', leave = False)
        num_epochs_tqdm.set_description_str('training_loop')
        for epoch in num_epochs_tqdm:
            periodic_save_flag = False
            # Early stopping
            stop_flag = self.checkStop(stoppings = stoppings)
            if stop_flag:
                self.updateTimer(model_name = model_name)
                self.logger.info(f"[STTS] Training is early stopped at epoch {epoch} with GL = {stoppings['GL'][-1]:.4f}, NGL = {stoppings['NGL'][-1]:.4f}, SI = {stoppings['SI'][-1]}")
                break
                
            # Training, Validation
            train_predictions, train_labels, train_loss = self.doTraining(model        = model,
                                                                          train_loader = train_loader,
                                                                          class_dist   = train_set.getClassDistribution(),
                                                                          optimizer    = optimizer,
                                                                          scheduler    = scheduler)
            val_predictions, val_labels, val_loss = self.doValidation(model      = model,
                                                                      val_loader = val_loader,
                                                                      class_dist = val_set.getClassDistribution())
            # Metrics
            model.eval()
            epoch_metrics = {}
            epoch_metrics['train'] = self.calculateMetrics(predictions = train_predictions,
                                                           labels = train_labels)
            epoch_metrics['val']   = self.calculateMetrics(predictions = val_predictions,
                                                           labels = val_labels)
            epoch_metrics['train']['loss'] = train_loss.to('cpu')
            epoch_metrics['val']['loss']   = val_loss.to('cpu')            
            
            # Stopping conditions
            epoch_stoppings = self.calculateStoppings(stoppings = stoppings,
                                                      epoch_metrics = epoch_metrics,
                                                      epoch = epoch)
            
            # Save progress
            metrics = self.updateMetrics(model_name    = model.name,
                                         metrics       = metrics,
                                         epoch_metrics = epoch_metrics,
                                         cp_epoch      = epoch)
            stoppings = self.updateStoppings(model_name      = model.name,
                                             stoppings       = stoppings,
                                             epoch_stoppings = epoch_stoppings,
                                             cp_epoch        = epoch)
                                             
            # Save model every 5 epochs:
            if (epoch + 1) % 5 == 0:
                self.updateCheckpoint(model     = model,
                                      optimizer = optimizer,
                                      cp_epoch  = epoch)
                periodic_save_flag = True
                self.updateTimer(model_name = model_name)
            # Save best model
            if stoppings['D_val'][-1] == stoppings['D_opt'][-1]:
                self.updateBestCheckpoint(model     = model,
                                          optimizer = optimizer,
                                          cp_epoch  = epoch)
        # Save model at end of training_loop
        if not periodic_save_flag:
            if not stop_flag:                
                self.updateCheckpoint(model     = model,
                                      optimizer = optimizer,
                                      cp_epoch  = epoch)
            elif not (epoch - 1 == cp_epoch):
                self.updateCheckpoint(model     = model,
                                      optimizer = optimizer,
                                      cp_epoch  = epoch - 1)
        self.updateTimer(model_name = model_name)
        self.logger.info(f'[CMPL] Trained for {epoch - cp_epoch} epochs in {self.timer}')

    def doTraining(self, model, train_loader, class_dist, optimizer, scheduler):
        model.train()
        predictions = torch.zeros(sum(class_dist), 2)
        labels = torch.zeros(sum(class_dist)).type(torch.LongTensor)
        epoch_loss = 0
        
        trainloader_tqdm = tqdm(train_loader, unit = 'batch', leave = False)
        trainloader_tqdm.set_description_str('doTraining')
        for batch_ind, (inputs, batch_labels) in enumerate(trainloader_tqdm):
            # Load data and move to cuda
            inputs_size = inputs.size(0)
            inputs = inputs.to(self.device)
            batch_labels = batch_labels.to(self.device)
            # forward
            outputs = model(inputs)
            batch_loss = self.criterion(outputs, batch_labels)
            # backward
            optimizer.zero_grad()
            batch_loss.backward()
            # gradient descent
            optimizer.step()
            scheduler.step()
            
            epoch_loss += batch_loss.detach() * inputs.size(0)
            
            # stack batch outputs to one tensor
            data_id = (torch.arange(inputs_size) + self.batch_size * batch_ind)
            predictions[data_id] = outputs.to('cpu')
            labels[data_id] = batch_labels.to('cpu')
            
        avg_loss = epoch_loss/sum(class_dist)
        
        return predictions, labels, avg_loss
        
    def doValidation(self, model, val_loader, class_dist):
        with torch.no_grad():
            model.eval()
            predictions = torch.zeros(sum(class_dist), 2)
            labels = torch.zeros(sum(class_dist)).type(torch.LongTensor)
            epoch_loss = 0
            
            valloader_tqdm = tqdm(val_loader, unit = 'batch', leave = False)
            valloader_tqdm.set_description_str('doValidation')
            for batch_ind, (inputs, batch_labels) in enumerate(valloader_tqdm):
                # Load data and move to cuda
                inputs_size = inputs.size(0)
                inputs = inputs.to(self.device)
                batch_labels = batch_labels.to(self.device)
                # forward
                outputs = model(inputs)
                batch_loss = self.criterion(outputs, batch_labels)
                
                epoch_loss += batch_loss.detach() * inputs.size(0)
                
                # stack batch outputs to one tensor
                data_id = (torch.arange(inputs_size) + self.batch_size * batch_ind)
                predictions[data_id] = outputs.to('cpu')
                labels[data_id] = batch_labels.to('cpu')
                
            avg_loss = epoch_loss/sum(class_dist)
            
            return predictions, labels, avg_loss
    
    def checkMatchedCheckpoint(self, cp_epoch, cp_epoch_2, cp_epoch_3):
        matched_flag = (cp_epoch == cp_epoch_2) & (cp_epoch == cp_epoch_3)
        if matched_flag:
            self.logger.debug(f'[PASS] Triple-check cp_epoch. Matched: {cp_epoch}')
        else:
            self.logger.warning(f'[FAIL] Triple-check cp_epoch. Mismatched: {cp_epoch}, {cp_epoch_2}, {cp_epoch_3}')
            cp_epoch = min(cp_epoch, cp_epoch_2, cp_epoch_3)
        return matched_flag, cp_epoch
    
    def checkStop(self, stoppings):
        stop_flag = ((stoppings['GL'][-1]  >= stoppings['GL_stop']) &
                     (stoppings['NGL'][-1] >= stoppings['NGL_stop']) &
                     (stoppings['SI'][-1]  >= stoppings['SI_stop']))
        return stop_flag
    
    # Metrics
    def calculateMetrics(self, predictions, labels):
        softmax = torch.nn.Softmax(dim = 1)
        probability = softmax(predictions)
        
        # AU-ROC, ROC, AU_PRC, PRC
        steps           = 101
        threshold_arr   = torch.linspace(start = -1e-7, end = 1+1e-7, steps = steps)
        recall_arr      = torch.zeros(steps)
        precision_arr   = torch.zeros(steps)
        specificity_arr = torch.zeros(steps)
        fall_out_arr    = torch.zeros(steps)
        
        for threshold_ind, threshold in enumerate(threshold_arr):
            # neg = 0, pos = 1
            pos_pred_mask   = probability[:, 1] >= threshold
            neg_pred_mask   = ~pos_pred_mask
            neg_label_mask  = (labels == 0)
            pos_label_mask  = ~neg_label_mask
            
            true_pos = (pos_label_mask & pos_pred_mask).sum()
            true_neg = (neg_label_mask & neg_pred_mask).sum()
            
            # TPR/sensi/recall
            recall_arr[threshold_ind]       = true_pos/pos_label_mask.sum()
            # PPV/precision
            precision_arr[threshold_ind]    = true_pos/pos_pred_mask.sum()
            # FPR
            specificity_arr[threshold_ind]  = true_neg/neg_label_mask.sum()
            fall_out_arr[threshold_ind]     = 1 - specificity_arr[threshold_ind]
        
        roc             = torch.cat((fall_out_arr.unsqueeze(dim = 0), recall_arr.unsqueeze(dim = 0)),
                                     dim = 0)
        fall_out_diff   = fall_out_arr[:-1] - fall_out_arr[1:]
        recall_avg      = (recall_arr[:-1] + recall_arr[1:])/2
        auroc           = (fall_out_diff * recall_avg).sum()
        
        # Fix NaN in precision_arr
        # Check for nan
        sum_nan = precision_arr.isnan().sum()
        if sum_nan > 0:
            nan_index = precision_arr.isnan().nonzero(as_tuple = True)[0]
            # Check if the nan strip is continuous
            if nan_index.max() - nan_index.min() + 1 == sum_nan:
                # Assign previously adjacent value
                pre_adj_value = precision_arr[nan_index.min() - 1].clone()
                precision_arr[nan_index] = pre_adj_value
        
        prc             = torch.cat((recall_arr.unsqueeze(dim = 0), precision_arr.unsqueeze(dim = 0)),
                                     dim = 0)
        recall_diff     = recall_arr[:-1] - recall_arr[1:]
        precision_avg   = (precision_arr[:-1] + precision_arr[1:])/2
        auprc           = (recall_diff * precision_avg).sum()
        
        # G-mean
        gmean = torch.sqrt(recall_arr * specificity_arr)
        max_gmean, max_gmean_ind = gmean.detach().unsqueeze(dim = 1).max(dim = 0)
        max_gmean_threshold = threshold_arr[max_gmean_ind]
        # Youden’s J
        youdensJ = (recall_arr - fall_out_arr)
        max_youdensJ, max_youdensJ_ind = youdensJ.detach().unsqueeze(dim = 1).max(dim = 0)
        max_youdensJ_threshold = threshold_arr[max_gmean_ind]
        # todo: save threshold
        
        # neg = 0, pos = 1
        pos_pred_mask   = probability[:, 1] >= self.threshold
        neg_pred_mask   = ~pos_pred_mask
        neg_label_mask  = (labels == 0)
        pos_label_mask  = ~neg_label_mask
        
        true_pos    = (pos_label_mask & pos_pred_mask).sum()
        false_neg   = (pos_label_mask & neg_pred_mask).sum()
        false_pos   = (neg_label_mask & pos_pred_mask).sum()
        true_neg    = (neg_label_mask & neg_pred_mask).sum()

        # Metrics
        confusion_matrix    = torch.tensor([[true_pos, false_neg], [false_pos, true_neg]])
        accuracy            = (true_pos + true_neg)/(pos_label_mask.sum() + neg_label_mask.sum())
        precision           = true_pos/pos_pred_mask.sum()
        recall              = true_pos/pos_label_mask.sum()
        specificity         = true_neg/neg_label_mask.sum()
        f1_score            = 2 * (precision * recall) / (precision + recall) 
        
        epoch_metrics = {}
        for metric in self.metric_vars:
            if metric != 'loss':
                epoch_metrics[metric] = vars()[metric]
        return epoch_metrics
        
    def preallocateMetrics(self):
        metrics = {}
        for subset in self.subsets:
            metrics[subset] = {}
            for metric in self.metric_vars:
                if metric == 'confusion_matrix':
                    metrics[subset][metric] = torch.zeros(2, 2).type(torch.LongTensor).unsqueeze(dim = 0)
                elif metric in ['loss', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auroc', 'auprc']:
                    metrics[subset][metric] = torch.zeros(1)
                # todo: fit all size of roc & prc
                elif metric in ['roc']:
                    metrics[subset][metric] = torch.zeros(2, 51).unsqueeze(dim = 0)
                elif metric in ['prc']:
                    metrics[subset][metric] = torch.cat((torch.zeros(1, 51), torch.ones(1, 51)), dim = 0).unsqueeze(dim = 0)
        cp_epoch = -1
        self.logger.debug(f'[CMPL] Pre-allocate metrics at cp_epoch {cp_epoch}')
        return metrics, cp_epoch
        
    def loadMetrics(self, model_name, cp_epoch = None):
        metrics_path = f'{self.project_path}logs/{model_name}/training/metrics.pt'
        self.logger.debug(f'[STTS] Loading metrics of model {model_name} at {metrics_path}...')
        try:
            # Load metrics
            load_dict = torch.load(metrics_path)
            metrics = load_dict['metrics']
            if cp_epoch is None:
                cp_epoch    = load_dict['cp_epoch']
            elif cp_epoch == -1:
                metrics, _ = self.preallocateMetrics()
            else:
                for subset in self.subsets:
                    for metric in self.metric_vars:
                        metrics[subset][metric] = metrics[subset][metric][0:cp_epoch + 1]
            
            self.logger.debug(f'[CMPL] Load metrics at cp_epoch {cp_epoch}')
        except FileNotFoundError:
            self.logger.debug(f'[FAIL] No savefile found')
            metrics, cp_epoch = self.preallocateMetrics()
        
        return metrics, cp_epoch
        
    def updateMetrics(self, model_name, metrics, epoch_metrics, cp_epoch):
        metrics_path = f'{self.project_path}logs/{model_name}/training/metrics.pt'
        # Update metrics
        
        if cp_epoch == 0:
            # overwrite
            for subset in self.subsets:
                for metric in self.metric_vars:
                    metrics[subset][metric] = epoch_metrics[subset][metric].unsqueeze(dim = 0)
        else:
            # extend
            for subset in self.subsets:
                for metric in self.metric_vars:
                    metrics[subset][metric] = torch.cat(
                    (   
                        metrics[subset][metric],
                        epoch_metrics[subset][metric].unsqueeze(dim = 0)
                    ),
                    dim = 0)
        
        # Save metrics
        torch.save({
            'metrics': metrics,
            'cp_epoch': cp_epoch
            }, metrics_path)
        self.logger.debug(f'[CMPL] Save metrics at cp_epoch {cp_epoch}')
        return metrics
    
    # Checkpoint
    def loadCheckpoint(self, model, optimizer):
        checkpoint_path = f'{self.project_path}logs/{model.name}/training/checkpoint.pt'
        self.logger.debug(f'[STTS] Loading checkpoint of model {model.name} at {checkpoint_path}...')
        try:
            load_dict = torch.load(checkpoint_path)
            model.load_state_dict(load_dict['model_state_dict'])
            optimizer.load_state_dict(load_dict['optimizer_state_dict'])
            cp_epoch = load_dict['cp_epoch']
            self.logger.debug(f'[CMPL] Load checkpoint at cp_epoch {cp_epoch}')
        except FileNotFoundError:
            cp_epoch = -1
            self.logger.debug(f'[FAIL] No savefile found. Presume training from scratch at cp_epoch {cp_epoch}.')
            
        return model, optimizer, cp_epoch
    
    def updateCheckpoint(self, model, optimizer, cp_epoch):
        checkpoint_path = f'{self.project_path}logs/{model.name}/training/checkpoint.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'cp_epoch': cp_epoch
            }, checkpoint_path)
        self.logger.debug(f'[CMPL] Save model {model.name} at cp_epoch {cp_epoch}')
    
    def updateBestCheckpoint(self, model, optimizer, cp_epoch):
        checkpoint_path = f'{self.project_path}logs/{model.name}/training/bestcheckpoint.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'cp_epoch': cp_epoch
            }, checkpoint_path)
        self.logger.debug(f'[CMPL] Save best model {model.name} at cp_epoch {cp_epoch}')

    # Stoppings
    def calculateStoppings(self, stoppings, epoch_metrics, epoch):
        stop_metric = stoppings['stop_metric']
        strip_len   = stoppings['strip_len']
        cooldown    = stoppings['cooldown']
        patience    = stoppings['patience']
        latency     = stoppings['latency']
        
        is_incr         = torch.zeros(patience).type(torch.bool)
        incr_from_start = torch.zeros(patience).type(torch.bool)
        
        D_train = 1 - epoch_metrics['train'][stop_metric]
        D_val   = 1 - epoch_metrics['val'][stop_metric]
        D_opt   = torch.min(stoppings['D_opt'][-1], D_val)
        
        # GL
        epoch_GL = D_val/D_opt - 1
        GL       = torch.maximum(stoppings['GL'][-1]*cooldown, epoch_GL)
        
        # TF
        strip_len_var = strip_len
        if epoch == 0:
            strip_len_var = 1
            strip = D_train
        else:
            if epoch < strip_len:
                strip_len_var = epoch + 1
            strip = torch.cat((stoppings['D_train'][-strip_len_var + 1:],
                               D_train.unsqueeze(dim = 0)),
                              dim = 0)
        
        # TF
        if strip.sum() == 0:
            # If the whole strip are 0s
            TF = torch.tensor(0)
        else: 
            TF = strip.sum()/(strip_len_var * strip.min()) - 1
            
        # NGL
        if (epoch_GL == 0):
            epoch_NGL = torch.tensor(0)
        else:
            epoch_NGL = epoch_GL/TF
        NGL = torch.maximum(stoppings['NGL'][-1]*cooldown, epoch_NGL)
        
        # epoch_SI
        long_D_val = torch.cat((stoppings['D_val'],
                                D_val.unsqueeze(dim = 0)),
                               dim = 0)
        strip_L_ind = torch.arange(epoch + 1 - patience - latency, epoch + 1 - latency)
        strip_R_ind = torch.arange(epoch + 1 - patience, epoch + 1)
        strip_L_ind[strip_L_ind < 0] = 0
        strip_R_ind[strip_R_ind < 0] = 0
        strip_L = long_D_val[strip_L_ind]      # Causing the tcmalloc crashes on Colab
        strip_R = long_D_val[strip_R_ind]
        
        for ind in torch.arange(patience):
            if strip_L_ind[ind] >= 0:
                is_incr[ind] = strip_L[ind] < strip_R[ind]
            else:
                is_incr[ind] = False
            incr_from_start[ind] = is_incr[0: ind + 1].prod().type(torch.LongTensor)
        
        epoch_SI = incr_from_start.sum()
        SI = torch.maximum(stoppings['SI'][-1], epoch_SI)
        
        epoch_stoppings = {}
        for stopping in self.stopping_vars:
            epoch_stoppings[stopping] = vars()[stopping]
        return epoch_stoppings
    
    def preallocateStoppings(self):
        stoppings = {}
        for stopping in self.stopping_params:
            stoppings[stopping] = vars(self)[stopping]
        for stopping in self.stopping_vars:
            if stopping in ['D_train', 'D_val', 'D_opt']:
                stoppings[stopping] = torch.ones(1)
            elif stopping == 'SI':
                stoppings[stopping] = torch.zeros(1).type(torch.LongTensor)
            else:
                stoppings[stopping] = torch.zeros(1)
        
        cp_epoch = -1
        self.logger.debug(f'[CMPL] Pre-allocate stoppings at cp_epoch {cp_epoch}')
        return stoppings, cp_epoch
    
    def loadStoppings(self, model_name, cp_epoch = None):
        stoppings_path = f'{self.project_path}logs/{model_name}/training/stoppings.pt'
        self.logger.debug(f'[STTS] Loading stoppings of model {model_name} at {stoppings_path}...')
        try:
            # Load stopping
            load_dict = torch.load(stoppings_path)
            stoppings           = load_dict['stoppings']
            if cp_epoch is None:
                cp_epoch    = load_dict['cp_epoch']
            elif cp_epoch == -1:
                stoppings, _ = self.preallocateStoppings()
            else:
                for stopping in self.stopping_vars:
                    stoppings[stopping] = stoppings[stopping][0:cp_epoch + 1]
            self.logger.debug(f'[CMPL] Load stoppings at cp_epoch {cp_epoch}')
        except FileNotFoundError:
            self.logger.debug(f'[FAIL] No savefile found. Pre-allocating stoppings.')
            stoppings, cp_epoch = self.preallocateStoppings()
        
        return stoppings, cp_epoch
        
    def updateStoppings(self, model_name, stoppings, epoch_stoppings, cp_epoch):
        stoppings_path = f'{self.project_path}logs/{model_name}/training/stoppings.pt'
        
        # Update stoppings
        # todo: for stopping in []
        if cp_epoch == 0:
            # overwrite
            for stopping in self.stopping_vars:
                stoppings[stopping] = epoch_stoppings[stopping].unsqueeze(dim = 0)
        else:
            # extend
            for stopping in self.stopping_vars:
                stoppings[stopping] = torch.cat(
                (   
                    stoppings[stopping],
                    epoch_stoppings[stopping].unsqueeze(dim = 0)
                ),
                dim = 0)
        
        # Save stoppings
        torch.save({
            'stoppings': stoppings,
            'cp_epoch': cp_epoch
            }, stoppings_path)
        self.logger.debug(f'[CMPL] Save stoppings at cp_epoch {cp_epoch}')
        return stoppings
    
    # Timer
    def loadTimer(self, model_name):
        timer_path = f'{self.project_path}logs/{model_name}/training/timer.pt'
        try:
            load_dict = torch.load(timer_path)
            self.timer = load_dict['timer']
        except FileNotFoundError:
            self.timer = datetime.timedelta(0)
            self.logger.debug(f'[FAIL] No savefile found. Preallocate timer')
        
        self.since = datetime.datetime.now()
        timer_to_log = self.timer - datetime.timedelta(microseconds = self.timer.microseconds)
        self.logger.debug(f'[CMPL] Load timer = {timer_to_log}')
        
    def updateTimer(self, model_name):
        timer_path = f'{self.project_path}logs/{model_name}/training/timer.pt'
        now = datetime.datetime.now()
        self.timer += now - self.since
        self.since = now
        torch.save({
            'timer': self.timer
            }, timer_path)
            
        timer_to_log = self.timer - datetime.timedelta(microseconds = self.timer.microseconds)
        self.logger.debug(f'[CMPL] Save timer = {timer_to_log}')

##################################################
#                                                #
#                  Retired codes                 #
#                                                #                    
##################################################

#Tensorboard Writer

    # __init__(...):
        # self.writer = None
    
    # def initTensorboardWriter(self):
        # if self.writer is None:
            # current_time = (datetime.datetime.now() + datetime.timedelta(hours = 7)).strftime("%Y-%m-%d %H:%M:%S")
            # tensorboard_path = self.project_path + '/logs/tensorboard/' + current_time
            # self.writer = SummaryWriter(log_dir = tensorboard_path)
            # self.logger.debug("[CMPL] Initialized Tensorboard Writer")

    # training_loop(...):
        # self.initTensorboardWriter()
        # for epoch in ...:
            # for metric in ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auroc', 'loss']:
              # self.writer.add_scalars(metric,
                                       # {'train': epoch_metrics['train'][metric], 'val': epoch_metrics['val'][metric]},
                                       # epoch)

##################################################