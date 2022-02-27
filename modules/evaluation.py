import torch
import torch.optim
from tqdm.auto import tqdm
import time, datetime
import numpy as np

import logging
import modules.logconf
    
fix = 32

class EvaluationClass:
    def __init__(self, project_path):
        # Init logger
        self.project_path = project_path
        self.logger = logging.getLogger(__name__)
        modules.logconf.initLogger(self.logger, project_path = self.project_path)
        
        self.batch_size = 16        # Hot fix
        self.num_workers = 0        # Hot fix
        self.num_candidates = 5
        self.metric_vars = ['confusion_matrix', 'loss', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auroc', 'roc', 'auprc', 'prc']
        self.candidate_metric_names = ['confusion_matrix', 'accuracy', 'precision', 'recall', 'specificity', 'max_f1', 'auroc', 'roc', 'auprc', 'prc', 'threshold_max_f1', 'cp_epoch_total', 'cp_epoch_best', 'timer']
        
        self.generator = torch.Generator()
        
        
        # Use CUDA
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.logger.info(f"Using {self.device} device.")
        
        self.logger.debug(f'[CMPL] Initialized EvaluationClass - fix {fix}')
    
    def worker_init_fn(self, worker_id):
        process_seed = torch.initial_seed()
        # Back out the base_seed so we can use all the bits.
        base_seed = process_seed - worker_id
        ss = np.random.SeedSequence([worker_id, base_seed])
        # More than 128 bits (4 32-bit words) would be overkill.
        np.random.seed(ss.generate_state(4))
    
    # Validate
    def validate(self, modelClass, model_name, val_set):
        # Model inference
        model = modelClass.initModel(model_name = model_name)
        model_state_dict, cp_epoch_best = self.loadBestCheckpoint(model_name)
        model.load_state_dict(model_state_dict)
        
        val_loader = torch.utils.data.DataLoader(dataset        = val_set,
                                                 batch_size     = self.batch_size,
                                                 shuffle        = True,
                                                 num_workers    = self.num_workers,
                                                 worker_init_fn = self.worker_init_fn,
                                                 generator      = self.generator)
                                                 
        val_predictions, val_labels = self.doValidation(model      = model,
                                                        loader = val_loader,
                                                        class_dist = val_set.getClassDistribution())
        
        # Re-find best threshold (again) with higher sampling
        candidate_metrics = {}
        candidate_metrics = self.calculateROCPRC(predictions   = val_predictions,
                                                 labels        = val_labels,
                                                 candidate_metrics = candidate_metrics)
        
        
        prc = candidate_metrics['prc']
        f1s = 2 * (prc[0] * prc[1]) / (prc[0] + prc[1])

        max_f1, threshold_ind_max_f1 = torch.max(f1s, dim = 0)
        threshold_max_f1 = torch.linspace(start = -1e-7, end = 1+1e-7, steps = 501)[threshold_ind_max_f1]
        
        candidate_metrics = self.calculateOtherMetrics(predictions   = val_predictions,
                                                   labels        = val_labels,
                                                   threshold     = threshold_max_f1,
                                                   candidate_metrics = candidate_metrics)
        
        _, cp_epoch_total = self.loadMetrics(model_name = model_name)
        timer = self.loadTimer(model_name = model_name)
        
        candidate_metrics['max_f1']           = max_f1
        candidate_metrics['threshold_max_f1'] = threshold_max_f1
        candidate_metrics['cp_epoch_total']   = cp_epoch_total
        candidate_metrics['cp_epoch_best']    = cp_epoch_best
        candidate_metrics['timer']            = timer.total_seconds()
        
        return model, candidate_metrics
        
    def doValidation(self, model, loader, class_dist):
        with torch.no_grad():
            model.eval()
            predictions = torch.zeros(sum(class_dist), 2)
            labels = torch.zeros(sum(class_dist)).type(torch.LongTensor)
            epoch_loss = 0
            
            loader_tqdm = tqdm(loader, unit = 'batch', leave = False)
            loader_tqdm.set_description_str('doValidation')
            for batch_ind, (inputs, batch_labels) in enumerate(loader_tqdm):
                # Load data and move to cuda
                inputs_size = inputs.size(0)
                inputs = inputs.to(self.device)
                batch_labels = batch_labels.to(self.device)
                # forward
                outputs = model(inputs)                
                # stack batch outputs to one tensor
                data_id = (torch.arange(inputs_size) + self.batch_size * batch_ind)
                predictions[data_id] = outputs.to('cpu')
                labels[data_id] = batch_labels.to('cpu')
            
            return predictions, labels
    
    # Test
    def test(self, modelClass, model_name, candidate_ind, test_set):
        # Model inference
        model = modelClass.initModel(model_name = model_name)
        model_state_dict = self.loadCandidate(model_name, candidate_ind)
        model.load_state_dict(model_state_dict)
        
        test_loader = torch.utils.data.DataLoader(dataset        = test_set,
                                                 batch_size     = self.batch_size,
                                                 shuffle        = True,
                                                 num_workers    = self.num_workers,
                                                 worker_init_fn = self.worker_init_fn,
                                                 generator      = self.generator)
                                                 
        test_predictions, test_labels = self.doValidation(model      = model,
                                                          loader     = test_loader,
                                                          class_dist = test_set.getClassDistribution())
        
        # Re-find best threshold (again) with higher sampling
        candidate_metrics = {}
        candidate_metrics = self.calculateROCPRC(predictions       = test_predictions,
                                                 labels            = test_labels,
                                                 candidate_metrics = candidate_metrics)

        threshold_max_f1 = self.loadValidationMetrics(model_name = model_name)['threshold_max_f1'][candidate_ind]
        
        candidate_metrics = self.calculateOtherMetrics(predictions       = test_predictions,
                                                       labels            = test_labels,
                                                       threshold         = threshold_max_f1,
                                                       candidate_metrics = candidate_metrics)
        
        candidate_metrics['threshold_max_f1'] = threshold_max_f1
        
        return model, candidate_metrics
       
    # Calculate metrics
    def calculateROCPRC(self, predictions, labels, candidate_metrics):
        softmax = torch.nn.Softmax(dim = 1)
        probability = softmax(predictions)
        
        steps           = 501
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
        
        for metric in ['roc', 'auroc', 'prc', 'auprc']:
            candidate_metrics[metric] = vars()[metric]
        return candidate_metrics
    
    def calculateOtherMetrics(self, predictions, labels, threshold, candidate_metrics):
        softmax = torch.nn.Softmax(dim = 1)
        probability = softmax(predictions)
        
        # neg = 0, pos = 1
        pos_pred_mask   = probability[:, 1] >= threshold
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
        
        for metric in ['confusion_matrix', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score']:
            candidate_metrics[metric] = vars()[metric]
        return candidate_metrics
    
    # Load from training.py
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
    
    def loadTimer(self, model_name):
        timer_path = f'{self.project_path}logs/{model_name}/training/timer.pt'
        
        load_dict = torch.load(timer_path)
        timer = load_dict['timer']
        
        timer_to_log = timer - datetime.timedelta(microseconds = timer.microseconds)
        self.logger.debug(f'[CMPL] Load timer = {timer_to_log}')
        return timer
    
    def loadBestCheckpoint(self, model_name):
        checkpoint_path = f'{self.project_path}logs/{model_name}/training/bestcheckpoint.pt'
        self.logger.debug(f'[STTS] Loading best checkpoint of model {model_name} at {checkpoint_path}...')
        try:
            checkpoint = torch.load(checkpoint_path, map_location = self.device)
            model_state_dict     = checkpoint['model_state_dict']
            cp_epoch             = checkpoint['cp_epoch']
            self.logger.debug(f'[CMPL] Load best checkpoint at cp_epoch {cp_epoch}')
        except FileNotFoundError:
            cp_epoch = -1
            self.logger.debug(f'[FAIL] No savefile found. Presume training from scratch at cp_epoch {cp_epoch}.')
            
        return model_state_dict, cp_epoch
    
    # Validation metrics
    def loadValidationMetrics(self, model_name):
        validation_metrics_path = f'{self.project_path}logs/{model_name}/evaluation/validation_metrics.pt'
        self.logger.debug(f'[STTS] Loading validation metrics of model {model_name} at {validation_metrics_path}...')
        try:
            # Load
            load_dict = torch.load(validation_metrics_path)
            validation_metrics = load_dict['validation_metrics']
        except FileNotFoundError:
            self.logger.debug(f'[FAIL] No savefile found')
            # Pre-allocate
            validation_metrics = {}
            for metric in self.candidate_metric_names:
                if metric in ['accuracy', 'precision', 'recall', 'specificity', 'max_f1', 'auroc', 'auprc', 'threshold_max_f1', 'cp_epoch_total', 'cp_epoch_best', 'timer']:
                    validation_metrics[metric] = torch.zeros(self.num_candidates)
                elif metric in ['confusion_matrix']:
                    validation_metrics[metric] = torch.zeros(self.num_candidates, 2, 2)
                elif metric in ['roc', 'prc']:
                    validation_metrics[metric] = torch.zeros(self.num_candidates, 2, 501)
        return validation_metrics
        
    def updateValidationMetrics(self, model_name, candidate_metrics, candidate_ind):
        validation_metrics_path = f'{self.project_path}logs/{model_name}/evaluation/validation_metrics.pt'
        # Load
        validation_metrics = self.loadValidationMetrics(model_name)
        # Update
        for metric in self.candidate_metric_names:
            validation_metrics[metric][candidate_ind] = candidate_metrics[metric]
        # Save
        torch.save({
            'validation_metrics': validation_metrics,
            }, validation_metrics_path)
        self.logger.debug(f'[CMPL] Save validation metrics for candidate {candidate_ind}')
        return validation_metrics

    def updateCandidate(self, model, candidate_ind):
        candidate_path = f'{self.project_path}logs/{model.name}/evaluation/candidate_{candidate_ind}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            }, candidate_path)
        self.logger.debug(f'[CMPL] Save candidate {candidate_ind} of model {model.name}')

    def loadCandidate(self, model_name, candidate_ind):
        candidate_path = f'{self.project_path}logs/{model_name}/evaluation/candidate_{candidate_ind}.pt'
        load_dict = torch.load(candidate_path, map_location = self.device)
        model_state_dict = load_dict['model_state_dict']
        self.logger.debug(f'[CMPL] Load candidate {candidate_ind} of model {model_name}')
        return model_state_dict

    
#       
#
#
#