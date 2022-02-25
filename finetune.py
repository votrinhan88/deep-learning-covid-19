import torch
from tqdm.auto import tqdm
import numpy as np
import time, datetime
import bayes_opt

import logging
import modules.logconf

fix = 165

class FinetuneClass:
    def __init__(self, project_path, num_workers, criterion, num_trials, num_folds, num_epochs):
        # Init logger
        self.project_path = project_path
        self.logger       = logging.getLogger(__name__)
        modules.logconf.initLogger(self.logger, project_path = self.project_path)
        
        # Non-tunable hparams
        self.num_workers  = num_workers
        self.criterion    = criterion
        self.num_trials   = num_trials
        self.num_folds    = num_folds
        self.num_epochs   = num_epochs
        
        self.metric_names = ['auprc', 'prc', 'loss']
        self.subset_names = ['cv_train', 'cv_val']
        self.tunable_hparam_names = ['batch_size', 'max_lr', 'ratio_lr', 'max_momentum', 'base_momentum', 'weight_decay']
        
        self.logger.info(f'Non-tunable hyperparameters: num_workers = {self.num_workers}, criterion = {self.criterion}, num_trials = {self.num_trials}, num_folds = {self.num_folds}, num_epochs = {self.num_epochs}')
        self.logger.info(f'Tunable hyperparameters: {self.tunable_hparam_names}')
        
        self.generator = torch.Generator()
        
        # Use CUDA
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.logger.info(f"Using {self.device} device.")
        
        self.logger.debug(f'[CMPL] Initialized FinetuneClass - fix {fix}')
    
    def worker_init_fn(self, worker_id):
        process_seed = torch.initial_seed()
        # Back out the base_seed so we can use all the bits.
        base_seed = process_seed - worker_id
        ss = np.random.SeedSequence([worker_id, base_seed])
        # More than 128 bits (4 32-bit words) would be overkill.
        np.random.seed(ss.generate_state(4))
    
    # Crossval
    def crossval(self, modelClass, model_name, train_set, cp_trial, cp_fold, cp_epoch, reset_cv, batch_size, max_lr, ratio_lr, max_momentum, base_momentum, weight_decay):
        batch_size = int(batch_size)
        base_lr = max_lr*ratio_lr
        self.logger.info(f'[STTS] Start {self.num_folds}-fold crossval with model {model_name} for {self.num_epochs} epochs')
        
        # Init model & optimizer
        model = modelClass.initModel(model_name = model_name)
        optimizer = torch.optim.SGD(params       = model.parameters(),
                                    lr           = max_lr,
                                    momentum     = max_momentum,
                                    weight_decay = weight_decay)

        # Reload metrics
        metrics, _, _, _ = self.loadMetrics(model_name = model.name,
                                            cp_trial   = cp_trial,
                                            cp_fold    = cp_fold,
                                            cp_epoch   = cp_epoch)
        
        # New crossval run OR finetune from scratch
        if reset_cv == True:
            class_dist = torch.tensor(train_set.getClassDistribution())
            train_ind, val_ind = self.stratifiedSplit(class_dist)
        # Continue from last crossval run
        elif reset_cv == False:
            # Reload model
            model_state_dict, optimizer_state_dict, train_ind, val_ind, _, _, _ = self.loadCheckpoint(model_name = model_name)
            model.load_state_dict(model_state_dict)
            optimizer.load_state_dict(optimizer_state_dict)
            
        # Increase 1 epoch before continue:
        cp_epoch += 1
        if cp_epoch == self.num_epochs:
            self.logger.debug(f'[STTS] Already completed fold {cp_fold}')
            # Prepare for next fold: Reset model, optimizer
            cp_fold += 1
            if cp_fold == self.num_folds:
                return metrics
            cp_epoch = 0
            model = modelClass.initModel(model_name = model_name)
            optimizer = torch.optim.SGD(params       = model.parameters(),
                                        lr           = max_lr,
                                        momentum     = max_momentum,
                                        weight_decay = weight_decay)
        
        self.logger.info(f'[STTS] Continue from T, F, E = {cp_trial}, {cp_fold}, {cp_epoch}')
        fold_range = range(cp_fold, self.num_folds)
        crossval_tqdm = tqdm(fold_range, unit = 'fold', leave = False)
        crossval_tqdm.set_description_str('crossval')
        for fold in crossval_tqdm:
            # Init at begin of every fold: subsets, loaders, and scheduler
            # Must behave independently each fold, reset each epoch if needed to
            train_subset = torch.utils.data.dataset.Subset(dataset = train_set,
                                                           indices = train_ind[cp_fold])
            val_subset = torch.utils.data.dataset.Subset(dataset = train_set,
                                                         indices = val_ind[cp_fold])
            train_loader = torch.utils.data.DataLoader(dataset        = train_subset,
                                                       batch_size     = batch_size,
                                                       shuffle        = True,
                                                       num_workers    = self.num_workers,
                                                       worker_init_fn = self.worker_init_fn,
                                                       generator      = self.generator)
            val_loader = torch.utils.data.DataLoader(dataset        = val_subset,
                                                     batch_size     = batch_size,
                                                     shuffle        = True,
                                                     num_workers    = self.num_workers,
                                                     worker_init_fn = self.worker_init_fn,
                                                     generator      = self.generator)
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer      = optimizer,
                                                          max_lr         = max_lr,
                                                          base_lr        = base_lr,
                                                          max_momentum   = max_momentum,
                                                          base_momentum  = base_momentum,
                                                          step_size_up   = len(train_loader) // 2,
                                                          step_size_down = len(train_loader) - len(train_loader) // 2)

            epoch_range = range(cp_epoch, self.num_epochs)
            fold_tqdm = tqdm(epoch_range, unit = 'epoch', leave = False)
            fold_tqdm.set_description_str(f'Fold {fold}')
            for epoch in fold_tqdm:
                train_preds, train_labels, train_loss = self.doTraining(model            = model,
                                                                        optimizer        = optimizer,
                                                                        scheduler        = scheduler,
                                                                        train_loader     = train_loader,
                                                                        train_subset_len = train_subset.__len__(),
                                                                        batch_size       = batch_size)
                val_preds, val_labels, val_loss = self.doValidation(model          = model,
                                                                    val_loader     = val_loader,
                                                                    val_subset_len = val_subset.__len__(),
                                                                    batch_size     = batch_size)
                # Metrics
                model.eval()
                epoch_metrics = {}
                epoch_metrics['cv_train'] = self.calculateMetrics(predictions = train_preds,
                                                                  labels      = train_labels)
                epoch_metrics['cv_val']   = self.calculateMetrics(predictions = val_preds,
                                                                  labels      = val_labels)
                epoch_metrics['cv_train']['loss'] = train_loss.to('cpu')
                epoch_metrics['cv_val']['loss']   = val_loss.to('cpu')
                
                # Save metrics every epochs
                metrics = self.updateMetrics(model_name    = model.name,
                                             metrics       = metrics,
                                             epoch_metrics = epoch_metrics,
                                             cp_trial      = cp_trial,
                                             cp_fold       = fold,
                                             cp_epoch      = epoch)
                # Save model every 5 epochs:
                if (epoch % 5 == 0) | (epoch == self.num_epochs - 1):
                    self.updateCheckpoint(model     = model,
                                          optimizer = optimizer,
                                          train_ind = train_ind,
                                          val_ind   = val_ind,
                                          cp_trial  = cp_trial,
                                          cp_fold   = fold,
                                          cp_epoch  = epoch)
                    self.updateTimer(model_name = model_name)

            # Prepare for next fold: Reset model, optimizer
            if fold < self.num_folds - 1:
                cp_epoch = 0
            model = modelClass.initModel(model_name = model_name)
            optimizer = torch.optim.SGD(params       = model.parameters(),
                                        lr           = max_lr,
                                        momentum     = max_momentum,
                                        weight_decay = weight_decay)

        
        return metrics
    
    def stratifiedSplit(self, class_dist):
        num_labels = class_dist.size()[0]
        num_val = (1/self.num_folds * class_dist)
        
        # Calculate start and end points
        label_start = class_dist.cumsum(dim = 0) - class_dist
        label_end   = class_dist.cumsum(dim = 0)
        shuffled_ind = torch.cat([torch.randperm(class_dist[label]) + label_start[label] for label in range(num_labels)], dim = 0)
        
        folds       = torch.arange(self.num_folds).unsqueeze(dim = 0).transpose(0, 1)
        val_start   = (folds * num_val + label_start).round().type(torch.LongTensor)
        val_end     = ((folds + 1)*num_val + label_start).round().type(torch.LongTensor)

        train_ind = {fold: {label:{} for label in range(num_labels)} for fold in range(self.num_folds)}
        val_ind = {fold: {label:{} for label in range(num_labels)} for fold in range(self.num_folds)}
        
        split_check = torch.zeros(self.num_folds, num_labels, 4).type(torch.bool)
        
        for fold in range(self.num_folds):
            for label in range(num_labels):
                val_ind[fold][label] = shuffled_ind[val_start[fold][label]: val_end[fold][label]]
                train_ind[fold][label] = shuffled_ind[
                    torch.cat(
                        (
                            torch.arange(label_start[label], val_start[fold][label]),
                            torch.arange(val_end[fold][label], label_end[label])
                        ),
                        dim = 0)
                ]
                # Double-check with equivalent range of class distribution
                split_check[fold, label, :] = torch.tensor([
                    val_ind[fold][label].min() >= label_start[label],
                    val_ind[fold][label].max() <= label_end[label],
                    train_ind[fold][label].min() >= label_start[label],
                    train_ind[fold][label].max() <= label_end[label]
                ])
            val_ind[fold] = torch.cat([val_ind[fold][label] for label in range(num_labels)], dim = 0)
            train_ind[fold] = torch.cat([train_ind[fold][label] for label in range(num_labels)], dim = 0)
            
        # Splitting table
        if False:
            print('\ttrain_subset\t\t\tval_subset')
            for fold in range(self.num_folds):
                for label in range(num_labels):
                    print(f'F{fold}L{label}:\t[{label_start[label]:4d}, {val_start[fold][label]:4d}) U\t[{val_end[fold][label]:4d}, {label_end[label]:4d})\t[{val_start[fold][label]:4d}, {val_end[fold][label]:4d})')
        
        self.logger.debug(f'Double-check indices in range: {split_check.prod().bool()}')
        
        # Double-check if indices are different
        train_first = torch.zeros(self.num_folds)
        val_first = torch.zeros(self.num_folds)
        for fold in range(self.num_folds):
            val_first[fold] = val_ind[fold][0]
            train_first[fold] = train_ind[fold][0]
        different_flag = ((train_first.std() != 0) & (val_first.std() != 0)).item()
        self.logger.debug(f'Double-check indices difference: {different_flag}. val_first.std() = {val_first.std():.2f}, train_first.std() = {train_first.std():.2f}')
       
        return train_ind, val_ind
 
    def doTraining(self, model, optimizer, scheduler, train_loader, train_subset_len, batch_size):
        model.train()
        predictions = torch.zeros(train_subset_len, 2)
        labels = torch.zeros(train_subset_len).type(torch.LongTensor)
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
            data_id = (torch.arange(inputs_size) + batch_size * batch_ind)
            predictions[data_id] = outputs.to('cpu')
            labels[data_id] = batch_labels.to('cpu')
            
        avg_loss = epoch_loss/train_subset_len
        
        return predictions, labels, avg_loss

    def doValidation(self, model, val_loader, val_subset_len, batch_size):
        with torch.no_grad():
            model.eval()
            predictions = torch.zeros(val_subset_len, 2)
            labels = torch.zeros(val_subset_len).type(torch.LongTensor)
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
                data_id = (torch.arange(inputs_size) + batch_size * batch_ind)
                predictions[data_id] = outputs.to('cpu')
                labels[data_id] = batch_labels.to('cpu')
                
            avg_loss = epoch_loss/val_subset_len
            
            return predictions, labels, avg_loss
    
    # Checkpoint
    def loadCheckpoint(self, model_name):
        checkpoint_path = f'{self.project_path}logs/{model_name}/finetune/checkpoint.pt'
        self.logger.debug(f'[STTS] Loading checkpoint of model {model_name} at {checkpoint_path}...')
        try:
            #################################################################
            # ROLL BACK FIX 160
            checkpoint = torch.load(checkpoint_path, map_location = self.device)
            model_state_dict     = checkpoint['model_state_dict']
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            #################################################################
            train_ind            = checkpoint['train_ind']
            val_ind              = checkpoint['val_ind']
            cp_trial             = checkpoint['cp_trial']
            cp_fold              = checkpoint['cp_fold']
            cp_epoch             = checkpoint['cp_epoch']
            self.logger.debug(f'[CMPL] Load checkpoint at T, F, E = {cp_trial}, {cp_fold}, {cp_epoch}')
        except FileNotFoundError:
            model_state_dict     = None
            optimizer_state_dict = None
            train_ind            = None
            val_ind              = None
            cp_trial = -1
            cp_fold  = self.num_folds - 1
            cp_epoch = self.num_epochs - 1
            self.logger.debug(f'[FAIL] No savefile found. Start new crossval at T, F, E = {cp_trial}, {cp_fold}, {cp_epoch}.')
            
        return model_state_dict, optimizer_state_dict, train_ind, val_ind, cp_trial, cp_fold, cp_epoch
    
    def updateCheckpoint(self, model, optimizer, train_ind, val_ind, cp_trial, cp_fold, cp_epoch):
        checkpoint_path = f'{self.project_path}logs/{model.name}/finetune/checkpoint.pt'
        torch.save({
                    #########################################################
                    # ROLL BACK FIX 160
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    #################################################################
                    'train_ind': train_ind,
                    'val_ind': val_ind,
                    'cp_trial': cp_trial,
                    'cp_fold': cp_fold,
                    'cp_epoch': cp_epoch},
                   checkpoint_path)
        self.logger.debug(f'[CMPL] Save model {model.name} at T, F, E = {cp_trial}, {cp_fold}, {cp_epoch}')
    
    # Metrics
    def calculateMetrics(self, predictions, labels):
        softmax = torch.nn.Softmax(dim = 1)
        probability = softmax(predictions)
        
        # AU-ROC, ROC, AU_PRC, PRC
        steps           = 51
        threshold_arr   = torch.linspace(start = -1e-7, end = 1+1e-7, steps = 51)
        recall_arr      = torch.zeros(steps)
        precision_arr   = torch.zeros(steps)
        
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
        
        epoch_metrics = {}
        for metric in self.metric_names:
            if metric != 'loss':
                epoch_metrics[metric] = vars()[metric]
        return epoch_metrics
        
    def preallocateMetrics(self):
        metrics = {}
        for subset in self.subset_names:
            metrics[subset] = {}
            for metric in self.metric_names:
                if metric == 'confusion_matrix':
                    metrics[subset][metric] = torch.zeros(self.num_trials, self.num_folds, self.num_epochs, 2, 2).type(torch.LongTensor)
                elif metric in ['loss', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auroc', 'auprc']:
                    metrics[subset][metric] = torch.zeros(self.num_trials, self.num_folds, self.num_epochs)
                elif metric in ['roc']:
                    metrics[subset][metric] = torch.zeros(self.num_trials, self.num_folds, self.num_epochs, 2, 51)
                elif metric in ['prc']:
                    metrics[subset][metric] = torch.zeros(self.num_trials, self.num_folds, self.num_epochs, 2, 51)
                    metrics[subset][metric][:, :, :, 1, :] = 1
        cp_trial = -1
        cp_fold  = self.num_folds - 1
        cp_epoch = self.num_epochs - 1
        self.logger.debug(f'[CMPL] Pre-allocate metrics at T, F, E = {cp_trial}, {cp_fold}, {cp_epoch}')
        return metrics, cp_trial, cp_fold, cp_epoch
        
    def loadMetrics(self, model_name, cp_trial = None, cp_fold = None, cp_epoch = None):
        metrics_path = f'{self.project_path}logs/{model_name}/finetune/metrics.pt'
        self.logger.debug(f'[STTS] Loading metrics of model {model_name} at {metrics_path}...')
        try:
            # Load metrics
            load_dict = torch.load(metrics_path)
            metrics = load_dict['metrics']
            
            if (cp_trial is None) & (cp_fold is None) & (cp_epoch is None):
                cp_trial = load_dict['cp_trial']
                cp_fold  = load_dict['cp_fold']
                cp_epoch = load_dict['cp_epoch']
            else:       
                raw_metrics, _, _, _ = self.preallocateMetrics()
                larger_trials = torch.arange(cp_trial + 1, self.num_trials)
                larger_folds  = torch.arange(cp_fold + 1, self.num_folds)
                larger_epochs = torch.arange(cp_epoch + 1, self.num_epochs)
                # Reset values exceeding checkpoint
                for subset in self.subset_names:
                    for metric in self.metric_names:
                        metrics[subset][metric][larger_trials] = raw_metrics[subset][metric][larger_trials]        
                        metrics[subset][metric][cp_trial, larger_folds] = raw_metrics[subset][metric][cp_trial, larger_folds]
                        metrics[subset][metric][cp_trial, cp_fold, larger_epochs] = raw_metrics[subset][metric][cp_trial, cp_fold, larger_epochs]
            self.logger.debug(f'[CMPL] Load metrics at T, F, E = {cp_trial}, {cp_fold}, {cp_epoch}')
        except FileNotFoundError:
            self.logger.debug(f'[FAIL] No savefile found')
            metrics, cp_trial, cp_fold, cp_epoch = self.preallocateMetrics()
        
        return metrics, cp_trial, cp_fold, cp_epoch
        
    def updateMetrics(self, model_name, metrics, epoch_metrics, cp_trial, cp_fold, cp_epoch):
        metrics_path = f'{self.project_path}logs/{model_name}/finetune/metrics.pt'
        
        # Update metrics
        for subset in self.subset_names:
            for metric in self.metric_names:
                metrics[subset][metric][cp_trial, cp_fold, cp_epoch] = epoch_metrics[subset][metric]
        
        # Save metrics
        torch.save({
            'metrics' : metrics,
            'cp_trial': cp_trial,
            'cp_fold' : cp_fold,
            'cp_epoch': cp_epoch
            }, metrics_path)
        self.logger.debug(f'[CMPL] Save metrics at T, F, E = {cp_trial}, {cp_fold}, {cp_epoch}')
        return metrics
    
    # Bayesian
    def preallocateBayesian(self):
        cp_trial = 0
        new_cv_flag = True
        
        hparams = {}
        for hparam_name in self.tunable_hparam_names:
            if hparam_name == 'batch_size':
                hparams[hparam_name] = torch.zeros(self.num_trials).type(torch.IntTensor)
            else:
                hparams[hparam_name] = torch.zeros(self.num_trials)
        
        target = torch.zeros(self.num_trials)
        
        self.logger.debug(f'[CMPL] Pre-allocate Bayesian at T = {cp_trial}, new_cv_flag {new_cv_flag}')
        return hparams, target, cp_trial, new_cv_flag

    def loadBayesian(self, model_name, cp_trial = None):
        bayesian_path = f'{self.project_path}logs/{model_name}/finetune/bayesian.pt'
        self.logger.debug(f'[STTS] Loading bayesian of model {model_name} at {bayesian_path}...')
        try:
            # Load bayesian
            load_dict = torch.load(bayesian_path)
            hparams = load_dict['hparams']
            target  = load_dict['target']
            if (cp_trial is None):
                cp_trial    = load_dict['cp_trial']
                new_cv_flag = load_dict['new_cv_flag']
            else:
                raw_hparams, raw_target, _, _ = self.preallocateBayesian()
                larger_trials = torch.arange(cp_trial + 1, self.num_trials)
                
                # todo ?
                new_cv_flag = False
                
                # Reset values exceeding checkpoint
                target[larger_trials] = raw_target[larger_trials]
                for hparam_name in self.tunable_hparam_names:
                    hparams[hparam_name][larger_trials] = raw_hparams[hparam_name][larger_trials]
            self.logger.debug(f'[CMPL] Load bayesian at T = {cp_trial}, new_cv_flag {new_cv_flag}')
        except FileNotFoundError:
            self.logger.debug(f'[FAIL] No savefile found')
            hparams, target, cp_trial, new_cv_flag = self.preallocateBayesian()
        
        return hparams, target, cp_trial, new_cv_flag
    
    def loadBayesianOptimizer(self, B_optimizer, norm_hparams, target, cp_trial):
        # Load previous data
        if cp_trial == -1:
            self.logger.debug(f'[STTS] Start finetune from scratch')
            return B_optimizer

        for trial in range(cp_trial):
            # Do not Suggest or Evaluate. Only Register: tell the optimizer what target value was observed
            next_point = {}
            for hparam in norm_hparams.keys():
                next_point[hparam] = norm_hparams[hparam][trial]
            next_target = target[trial]
            B_optimizer.register(params=next_point, target=next_target)        
        self.logger.debug(f'[CMPL] Loaded B_optimizer with {len(B_optimizer.res)} trials')
        return B_optimizer
    
    def updateBayesian(self, model_name, hparams, target, trial_hparams, trial_target, cp_trial, new_cv_flag):
        bayesian_path = f'{self.project_path}logs/{model_name}/finetune/bayesian.pt'
        # Update bayesian
        for hparam_name in self.tunable_hparam_names:
            hparams[hparam_name][cp_trial] = trial_hparams[hparam_name]

        # When save hparams only without target
        if (trial_target != None):
            target[cp_trial] = trial_target
        
        # Save bayesian
        torch.save({
            'hparams'    : hparams,
            'target'     : target,
            'cp_trial'   : cp_trial,
            'new_cv_flag': new_cv_flag
            }, bayesian_path)
        self.logger.debug(f'[CMPL] Save bayesian at T = {cp_trial}')
        return hparams, target
    
    # Finetune
    def normalizeHparams(self, hparams):
        norm_hparams = {}
        norm_hparams['batch_size']    = hparams['batch_size'].log2()
        norm_hparams['max_lr']        = hparams['max_lr'].log10()
        norm_hparams['ratio_lr']      = hparams['ratio_lr'].log10()
        norm_hparams['max_momentum']  = hparams['max_momentum']
        norm_hparams['base_momentum'] = hparams['base_momentum']
        norm_hparams['weight_decay']  = hparams['weight_decay'].log10()
        return norm_hparams

    def inverseNormalizeHparams(self, norm_trial_hparams):
        trial_hparams = {}
        trial_hparams['batch_size']    = 2**(norm_trial_hparams['batch_size'].round().type(torch.LongTensor))
        trial_hparams['max_lr']        = 10**norm_trial_hparams['max_lr']
        trial_hparams['ratio_lr']      = 10**norm_trial_hparams['ratio_lr']
        trial_hparams['max_momentum']  = norm_trial_hparams['max_momentum']
        trial_hparams['base_momentum'] = norm_trial_hparams['base_momentum']
        trial_hparams['weight_decay']  = 10**norm_trial_hparams['weight_decay']
        return trial_hparams
    
    def checkMatchedCheckpoint(self, cp_trial_1, cp_trial_2, cp_trial_3, new_cv_flag, cp_fold_1, cp_fold_2, cp_epoch_1, cp_epoch_2):
        matched_flag = True
        # cp_fold_2, cp_epoch_2 will always exceed cp_fold_1, cp_epoch_1
        cp_trial, cp_fold, cp_epoch = cp_trial_1, cp_fold_1, cp_epoch_1
        
        matched_flag = matched_flag & (cp_trial_1 == cp_trial) & (cp_trial_2 == cp_trial) & (cp_trial_3 == cp_trial)
        self.logger.debug(f'cp_trial = {cp_trial}, matched_flag = {matched_flag}. Compare: {cp_trial_1}, {cp_trial_2}, {cp_trial_3}')
        if matched_flag == True:
            matched_flag = matched_flag & (cp_fold_1 == cp_fold) & (cp_fold_2 == cp_fold)
            self.logger.debug(f'cp_fold = {cp_fold}, matched_flag = {matched_flag}. Compare: {cp_fold_1}, {cp_fold_2}')
            if matched_flag == True:
                matched_flag = matched_flag & (cp_epoch_1 == cp_epoch) & (cp_epoch_2 == cp_epoch)
                self.logger.debug(f'cp_epoch = {cp_epoch}, matched_flag = {matched_flag}. Compare: {cp_epoch_1}, {cp_epoch_2}')
        
        # bayesian already init for next trial, but metrics & model haven't saved
        if (cp_trial_3 == cp_trial + 1) & (new_cv_flag == False):
            # Will discard the init, perform again
            new_cv_flag = True
        
        self.logger.debug(f'Choose checkpoint at T, F, E = {cp_trial}, {cp_fold}, {cp_epoch}, new_cv_flag = {new_cv_flag}')
        return matched_flag, cp_trial, cp_fold, cp_epoch, new_cv_flag
    
    def finetune(self, modelClass, model_name, train_set):
        self.loadTimer(model_name = model_name)
        # Init B_optimizer & utility
        B_optimizer = bayes_opt.BayesianOptimization(f       = None,
                                                     pbounds = {
                                                        'batch_size'   : (3, 6),
                                                        'max_lr'       : (-4, -2),
                                                        'ratio_lr'     : (-2, 0),
                                                        'max_momentum' : (0.85, 0.95),
                                                        'base_momentum': (0.75, 0.85),
                                                        'weight_decay' : (-3, -1)})
        utility = bayes_opt.UtilityFunction(kind = "ucb", kappa = 10, xi = 0.1)
        
        _, _, _, _, cp_trial_1, cp_fold_1, cp_epoch_1 = self.loadCheckpoint(model_name = model_name)
        _, cp_trial_2, cp_fold_2, cp_epoch_2 = self.loadMetrics(model_name = model_name)
        _, _, cp_trial_3, new_cv_flag = self.loadBayesian(model_name = model_name)
        matched_flag, cp_trial, cp_fold, cp_epoch, new_cv_flag = self.checkMatchedCheckpoint(cp_trial_1,
                                                                                             cp_trial_2,
                                                                                             cp_trial_3,
                                                                                             new_cv_flag,
                                                                                             cp_fold_1,
                                                                                             cp_fold_2,
                                                                                             cp_epoch_1,
                                                                                             cp_epoch_2)
        reset_cv = new_cv_flag
        
        hparams, target, _, _ = self.loadBayesian(model_name = model_name,
                                                  cp_trial   = cp_trial)
        # Normalize before loading B_optimizer
        norm_hparams = self.normalizeHparams(hparams = hparams)
        
        if new_cv_flag == True:
            # Move to new trial
            cp_trial += 1
            cp_fold  = 0
            cp_epoch = -1
            if cp_trial >= self.num_trials:
                self.logger.info('[STTS] Finetune already completed')
                return hparams, target, cp_trial, new_cv_flag
            # Continue with last B_optimizer
            B_optimizer = self.loadBayesianOptimizer(B_optimizer  = B_optimizer,
                                                     norm_hparams = norm_hparams,
                                                     target       = target,
                                                     cp_trial     = cp_trial)
            # Suggest: Next point to probe
            norm_trial_hparams = B_optimizer.suggest(utility)
            for key in norm_trial_hparams.keys():
                norm_trial_hparams[key] = torch.tensor(norm_trial_hparams[key])
            trial_hparams = self.inverseNormalizeHparams(norm_trial_hparams)
            # Save new hparams for later crossval runs
            new_cv_flag = False
            hparams, target = self.updateBayesian(model_name    = model_name,
                                                  hparams       = hparams,
                                                  target        = target,
                                                  trial_hparams = trial_hparams,
                                                  trial_target  = None,
                                                  cp_trial      = cp_trial,
                                                  new_cv_flag   = new_cv_flag)
            
        elif new_cv_flag == False:
            # Continue with last B_optimizer
            B_optimizer = self.loadBayesianOptimizer(B_optimizer  = B_optimizer,
                                                     norm_hparams = norm_hparams,
                                                     target       = target,
                                                     cp_trial     = cp_trial)
            # Continue with last trial's hparams
            trial_hparams = {}
            for hparam_name in self.tunable_hparam_names:
                trial_hparams[hparam_name] = hparams[hparam_name][cp_trial]

        self.logger.info(f'[STTS] Continue from T = {cp_trial}')
        
        # Evaluate: Target value
        self.logger.info(f"Tunable hyperparameters: batch_size = {trial_hparams['batch_size']}, max_lr = {trial_hparams['max_lr']:.3e}, ratio_lr = {trial_hparams['ratio_lr']:.3e}, max_momentum = {trial_hparams['max_momentum']:.3f}, base_momentum = {trial_hparams['base_momentum']:.3f}, weight_decay = {trial_hparams['weight_decay']:.3e}")
        metrics = self.crossval(modelClass  = modelClass,
                                model_name  = model_name,
                                train_set   = train_set,
                                cp_trial    = cp_trial,
                                cp_fold     = cp_fold,
                                cp_epoch    = cp_epoch,
                                reset_cv    = reset_cv,
                                **trial_hparams)
        new_cv_flag = True
        max_auprc_by_fold, _ = metrics['cv_val']['auprc'][cp_trial].max(dim = 1)
        trial_target = max_auprc_by_fold.mean()

        hparams, target = self.updateBayesian(model_name    = model_name,
                                              hparams       = hparams,
                                              target        = target,
                                              trial_hparams = trial_hparams,
                                              trial_target  = trial_target,
                                              cp_trial      = cp_trial,
                                              new_cv_flag   = new_cv_flag)
        self.updateTimer(model_name = model_name)
        self.logger.info(f'[CMPL] Finetune completed in {self.timer}')
        return hparams, target, cp_trial, new_cv_flag

    def loadTimer(self, model_name):
        timer_path = f'{self.project_path}logs/{model_name}/finetune/timer.pt'
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
        timer_path = f'{self.project_path}logs/{model_name}/finetune/timer.pt'
        now = datetime.datetime.now()
        self.timer += now - self.since
        self.since = now
        torch.save({
            'timer': self.timer
            }, timer_path)
            
        timer_to_log = self.timer - datetime.timedelta(microseconds = self.timer.microseconds)
        self.logger.debug(f'[CMPL] Save timer = {timer_to_log}')
#
#
#