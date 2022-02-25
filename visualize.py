import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
import matplotlib.colors as colors
import glob
import PIL

import logging
import modules.logconf

fix = 73

class VisualizeClass:
    def __init__(self, project_path):
        # Init logger
        self.project_path = project_path
        self.logger = logging.getLogger(__name__)
        modules.logconf.initLogger(self.logger, project_path = self.project_path)
        
        self.subsets = ['train', 'val']
        
        self.logger.debug(f'[CMPL] Initialized VisualizeClass - fix {fix}')
        
    def showPerformance(self, model_name):
        self.model_name = model_name
        # Initialization some dicts
        self.subset_2_name = {
            'train':    'Training set',
            'val':      'Validation set',
            'test':     'Test set'
        }
        self.metric_2_name = {
            'confusion_matrix': 'Confusion matrix',
            'loss':             'Loss',
            'accuracy':         'Accuracy',
            'precision':        'Precision',
            'recall':           'Recall',
            'specificity':      'Specificity',
            'f1_score':         'F1 score',
            'auroc':            'AU-ROC',
            'roc':              'ROC',
            'auprc':            'AU-PRC',
            'prc':              'PRC'
        }
        # Load metrics
        metrics, cp_epoch = self.loadMetrics()
        stoppings, _ = self.loadStoppings()
        if False:
            self.visConfusionMatrix(metrics = metrics,
                                    cp_epoch = cp_epoch)
            self.visGeneral(metrics = metrics,
                            cp_epoch = cp_epoch)
            self.visROC(metrics = metrics,
                        cp_epoch = cp_epoch)
            self.visPRC(metrics = metrics,
                        cp_epoch = cp_epoch)
        
        self.visStopping(stoppings = stoppings,
                         cp_epoch = cp_epoch)
        
    def loadMetrics(self):
        metrics_path = f'{self.project_path}logs/{self.model_name}/training/metrics.pt'
        self.logger.debug(f'[STTS] Loading metrics of model {self.model_name} at {metrics_path}...')
        try:
            # Load metrics
            load_dict = torch.load(metrics_path)
            metrics = load_dict['metrics']
            cp_epoch = load_dict['cp_epoch']
            self.logger.debug(f'[CMPL] Load metrics at checkpoint epoch {cp_epoch}')
        except FileNotFoundError:
            self.logger.debug(f'[FAIL] No savefile found.')
            metrics = None
            cp_epoch = -1
        
        return metrics, cp_epoch
        
    def loadStoppings(self):
        stoppings_path = f'{self.project_path}logs/{self.model_name}/training/stoppings.pt'
        self.logger.debug(f'[STTS] Loading stopping conditions of model {self.model_name} at {stoppings_path}...')
        try:
            # Load stopping
            load_dict = torch.load(stoppings_path)
            stoppings           = load_dict['stoppings']
            cp_epoch    = load_dict['cp_epoch']
            self.logger.debug(f'[CMPL] Load stopping conditions at checkpoint epoch {cp_epoch}')
        except FileNotFoundError:
            self.logger.debug(f'[FAIL] No savefile found.')
            stoppings = None
            cp_epoch = -1
        
        return stoppings, cp_epoch
    
    def interpolate_nan(self, array):
        # Check for nan
        if array.isnan().sum() > 0:
            nan_ind = array.isnan().nonzero(as_tuple = True)[0]
            last_ind = torch.tensor(array.size()[0]) - 1
            for ind in nan_ind:
                # Find nan "window"
                window_L = ind - 1
                while window_L in nan_ind:
                    window_L = window_L - 1    
                window_R = ind + 1
                while window_R in nan_ind:
                    window_R = window_R + 1
                
                if window_L < 0:
                    array[ind] = 0
                elif window_R > last_ind:
                    array[ind] = array[window_L]
                else:
                    # Interpolate (vector math)
                    array[ind] = ((window_R - ind)*array[window_L] + (ind - window_L)*array[window_R])/(window_R - window_L)
        else:
            nan_ind = []
        return array, nan_ind
    
    def visConfusionMatrix(self, metrics, cp_epoch):
        for epoch in range(cp_epoch + 1):
            fig1, ax1 = plt.subplots(1, 2, figsize = (10, 5), dpi = 80)
            suptitle1 = fig1.suptitle(self.metric_2_name['confusion_matrix'], x = 0.5, y = 1.06, size = 15, ha = 'center', va = 'center')
            for subset_ind, subset in enumerate(self.subsets):
                # Heatmap
                heatmap = ax1[subset_ind].pcolormesh(
                    metrics[subset]['confusion_matrix'][epoch],
                    cmap = 'Blues',
                    norm = colors.PowerNorm(gamma=0.2))
                heatmap.set_clim(0, metrics[subset]['confusion_matrix'][0].sum())
                # ax[subset_ind].invert_xaxis() Confusion matrix sucks. I was lucky.
                ax1[subset_ind].invert_yaxis()
                ax1[subset_ind].set_xticks(torch.arange(2) + 0.5)
                ax1[subset_ind].set_yticks(torch.arange(2) + 0.5)
                ax1[subset_ind].set_xticklabels(['P+', 'P-'], fontsize = 20)
                ax1[subset_ind].set_yticklabels(['L+', 'L-'], fontsize = 20)
                ax1[subset_ind].set_title(self.subset_2_name[subset])
                fig1.text(0.5, 1, f'Epoch {epoch:02d}', horizontalalignment='center', verticalalignment='center', size = 12)
                # Colorbar
                divider = make_axes_locatable(ax1[subset_ind])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(heatmap, cax = cax)
                # Text
                for i in range(2):
                    for j in range(2):
                        text = ax1[subset_ind].text(j + 0.5, i + 0.5, metrics[subset]['confusion_matrix'][epoch][i, j].item(), ha="center", va="center", size = 20)
            
            # Save all but show only last epoch
            fig1.tight_layout()
            fig1.savefig(f"{self.project_path}logs/{self.model_name}/confusion_matrix/{epoch:02d}.png", bbox_inches='tight', bbox_extra_artists=[suptitle1])
            if epoch == cp_epoch:
                plt.show(fig1)
            else:
                plt.close(fig1)
        
        # Make GIF
        fp_in = self.project_path + f'logs/{self.model_name}/confusion_matrix/*.png'
        fp_out = self.project_path + f'logs/{self.model_name}/graphs/confusion_matrix.gif'
        img, *imgs = [PIL.Image.open(f) for f in sorted(glob.glob(fp_in))[0:cp_epoch + 1]]
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                        save_all=True, duration=400, loop=0)
    
    def visGeneral(self, metrics, cp_epoch):
        # Init fig2
        fig2 = plt.figure(figsize = (10, 5), dpi = 80)
        suptitle2 = fig2.suptitle(f'Performance of {self.model_name}', x = 0.07, y = 1.07, size = 15, ha = 'left', va = 'center')
        legend_elements = [
            Line2D([0], [0], color='tab:blue', label='Training set'),
            Line2D([0], [0], color='tab:orange', label='Validation set'),
            Line2D([0], [0], color='white', marker='X', markerfacecolor='red', markersize = 10, label='NaN values replaced by interpolation')]
        legend2 = fig2.legend(handles = legend_elements, bbox_to_anchor=(0.98, 1.05, 0, 0), loc='center right')
        
        gs2 = fig2.add_gridspec(2, 4)
        metric_2_gs = {
            'loss':         gs2[0, 0],
            'accuracy':     gs2[0, 1],
            'specificity':  gs2[0, 2],
            'recall':       gs2[0, 3],
            'precision':    gs2[1, 2],
            'f1_score':     gs2[1, 3],
            'auroc':        gs2[1, 0],
            'auprc':        gs2[1, 1]
        }
        for metric in ['loss', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auroc', 'auprc']:
            ax2 = fig2.add_subplot(metric_2_gs[metric])
            ax2.set_title(self.metric_2_name[metric])
            ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
            for subset in self.subsets:
                if metric in ['precision', 'f1_score']:
                    metrics[subset][metric], nan_ind = self.interpolate_nan(metrics[subset][metric])
                    ax2.scatter(nan_ind, metrics[subset][metric][nan_ind], marker = 'x', color = 'red')
                ax2.plot(metrics[subset][metric])
            if metric not in ['loss', 'accuracy', 'specificity']:
                ax2.set_ylim([-0.05, 1.05])
        
        fig2.tight_layout()
        fig2.savefig(f'{self.project_path}logs/{self.model_name}/graphs/performance.png', bbox_inches='tight', bbox_extra_artists=[suptitle2, legend2])
        fig2.show()
        
    def visROC(self, metrics, cp_epoch):
        for epoch in range(cp_epoch + 1):
            # Init fig3
            fig3 = plt.figure(figsize = (6, 8), dpi = 80)
            suptitle3 = fig3.suptitle('Receiver Operating Characteristic (ROC) curve\nand Area Under the Curve (AU-ROC)', x = 0.5, y = 1.05, size = 15, ha = 'center', va = 'center')
            fig3.text(0.5, 1, f'Epoch {epoch:02d}', ha='center', va='center', size = 12)
            gs3 = fig3.add_gridspec(3, 1)
            metric_2_gs = {
                'roc': gs3[0:2],
                'auroc': gs3[2]
            }
            ax3_roc = fig3.add_subplot(metric_2_gs['roc'])
            ax3_auroc = fig3.add_subplot(metric_2_gs['auroc'])

            # ax3_roc: ROC, No skill
            for subset in self.subsets:
                ax3_roc.plot(metrics[subset]['roc'][epoch][0],
                             metrics[subset]['roc'][epoch][1])
                ax3_roc.scatter(metrics[subset]['roc'][epoch][0],
                                metrics[subset]['roc'][epoch][1])
            ax3_roc.plot([0, 1], [0, 1], '--')
            ax3_roc.set_xlabel('Fall-out (FPR)')
            ax3_roc.set_ylabel('Recall/Sensitivy (TPR)')
            ax3_roc.legend(
                labels = [
                    f"AU-ROC = {metrics['train']['auroc'][epoch]:.4f}: Training set",
                    f"AU-ROC = {metrics['val']['auroc'][epoch]:.4f}: Validation set",
                    'No skill'],
                loc = 'lower right')

            # ax3_auroc: AU-ROC, No skill
            for subset in ['train', 'val']: # self.subsets
                ax3_auroc.plot(metrics[subset]['auroc'][0:epoch + 1])
            ax3_auroc.plot([0, cp_epoch], [0.5, 0.5], '--')
            ax3_auroc.set_xlim([-0.05*cp_epoch, cp_epoch*1.05]) # Padding to the sides
            ax3_auroc.set_ylim([-0.05, 1.05])
            ax3_auroc.set_xlabel('Epoch')
            ax3_auroc.set_ylabel(self.metric_2_name['auroc'])
            ax3_auroc.xaxis.set_major_locator(MaxNLocator(integer=True))

            # Finish fig3
            fig3.tight_layout()
            fig3.savefig(f'{self.project_path}logs/{self.model_name}/roc/{epoch:02d}.png', bbox_inches='tight', bbox_extra_artists=[suptitle3])
            if epoch == cp_epoch:
                plt.show(fig3)
            else:
                plt.close(fig3)
        # Make GIF
        fp_in = self.project_path + f'logs/{self.model_name}/roc/*.png'
        fp_out = self.project_path + f'logs/{self.model_name}/graphs/roc.gif'
        img, *imgs = [PIL.Image.open(f) for f in sorted(glob.glob(fp_in))[0:cp_epoch + 1]]
        img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=400, loop=0)
        
    def visPRC(self, metrics, cp_epoch):
        for epoch in range(cp_epoch + 1):
            # Init fig4
            fig4 = plt.figure(figsize = (6, 8), dpi = 80)
            suptitle4 = fig4.suptitle('Precision-Recall Curve (PRC)\nand Area Under the Curve (AU-PRC)', x = 0.5, y = 1.05, size = 15, ha = 'center', va = 'center')
            fig4.text(0.5, 1, f'Epoch {epoch:02d}', ha='center', va='center', size = 12)
            gs4 = fig4.add_gridspec(3, 1)
            metric_2_gs = {
                'prc': gs4[0:2],
                'auprc': gs4[2]
            }
            ax4_prc = fig4.add_subplot(metric_2_gs['prc'])
            ax4_auprc = fig4.add_subplot(metric_2_gs['auprc'])
            
            # ax4_prc: PRC, No skill, Iso-F1 curves
            for subset in self.subsets:
                ax4_prc.plot(metrics[subset]['prc'][epoch][0],
                             metrics[subset]['prc'][epoch][1])
                ax4_prc.scatter(metrics[subset]['prc'][epoch][0],
                                metrics[subset]['prc'][epoch][1])
            no_skill_y = 184/5184
            ax4_prc.plot([-0.05, 1.05], [no_skill_y, no_skill_y], '--')
            iso_f1_x = torch.linspace(start = 0, end = 1.1, steps = 51)
            for f1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
                iso_f1_y = f1/iso_f1_x
                ax4_prc.plot(iso_f1_x, iso_f1_y, '--', color = (0.5, 0.5, 0.5))
                ax4_prc.text(x = 1, y = f1 + 0.02, s = f1, color = (0.5, 0.5, 0.5))
            
            ax4_prc.set_xlabel('Recall/Sensitivy (TPR)')
            ax4_prc.set_ylabel('Precision (PPV)')
            ax4_prc.set_xlim([-0.05, 1.05])
            ax4_prc.set_ylim([-0.05, 1.05])
            ax4_prc.legend(
                labels = [
                    f"AU-PRC = {metrics['train']['auprc'][epoch]:.4f}: Training set",
                    f"AU-PRC = {metrics['val']['auprc'][epoch]:.4f}: Validation set",
                    'No skill',
                    f'Iso-F1 curves'],
                loc = 'lower left')

            # ax4_auprc: AU-PRC, No skill
            for subset in ['train', 'val']: # self.subsets
                ax4_auprc.plot(metrics[subset]['auprc'][0:epoch + 1])
            ax4_auprc.plot([0, cp_epoch], [no_skill_y, no_skill_y], '--')
            
            ax4_auprc.set_xlim([-0.05*cp_epoch, cp_epoch*1.05]) # Padding to the sides
            ax4_auprc.set_ylim([-0.05, 1.05])
            ax4_auprc.set_xlabel('Epoch')
            ax4_auprc.set_ylabel(self.metric_2_name['auprc'])
            ax4_auprc.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            # Finish fig4
            fig4.tight_layout()
            fig4.savefig(f'{self.project_path}logs/{self.model_name}/prc/{epoch:02d}.png', bbox_inches='tight', bbox_extra_artists=[suptitle4])
            if epoch == cp_epoch:
                plt.show(fig4)
            else:
                plt.close(fig4)
        # Make GIF
        fp_in = self.project_path + f'logs/{self.model_name}/prc/*.png'
        fp_out = self.project_path + f'logs/{self.model_name}/graphs/prc.gif'
        img, *imgs = [PIL.Image.open(f) for f in sorted(glob.glob(fp_in))[0:cp_epoch + 1]]
        img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=400, loop=0)
        
    def visStopping(self, stoppings, cp_epoch):
        # Init fig5
        fig5, ax5 = plt.subplots(2, 1, figsize = (10, 6), sharex = 'col') # gridspec_kw={'height_ratios': [2, 1]}
        suptitle5 = fig5.suptitle(f'Stopping conditions of {self.model_name}', x = 0.5, y = 1.00, size = 20, ha = 'center', va = 'bottom')
        
        plt.rc('xtick', labelsize = 12)    # fontsize of the tick labels
        plt.rc('ytick', labelsize = 12)    # fontsize of the tick labels
        
        #ax5[0].plot(stoppings['TF'], color = 'black', linestyle='dashed', label = 'TF')
        ax5[0].plot(stoppings['D_train'], color = 'black', lw = 3, label = '$\mathregular{{D}_{tr}}$')
        ax5[0].plot(stoppings['D_opt'], color = 'tab:orange', lw = 2, linestyle='dashed', label = '$\mathregular{{D}_{opt}}$')
        ax5[0].plot(stoppings['D_val'], color = 'tab:orange', lw = 3, label = '$\mathregular{{D}_{val}}$')
        # for step: torch.arange(cp_epoch + 1), , where = 'mid'

        # ax5[1].plot(stoppings['epoch_GL'], color = 'red', label = 'epoch_GL', linestyle='dashed')
        # ax5[1].plot(stoppings['GL'], color = 'red', lw = 3, label = 'GL')
        ax5[1].plot(stoppings['epoch_NGL'], color = 'blue', label = '$\mathregular{{NGL}_{e}}$', linestyle='dashed')
        ax5[1].plot(stoppings['NGL'], color = 'blue', lw = 3, label = 'NGL')
        ax5[1].plot(stoppings['epoch_SI'], color = 'orange', label = '$\mathregular{{SI}_{e}}$', linestyle='dashed')
        ax5[1].plot(stoppings['SI'], color = 'orange', lw = 3, label = 'SI')
        ax5[1].set_xlabel('Epoch', fontsize = 15)
        
        # Plot trend lines of D_val
        if True:
            patience    = stoppings['patience']
            latency     = stoppings['latency']
            
            strip_L_ind = torch.arange(cp_epoch + 1 - patience - latency, cp_epoch + 1 - latency)
            strip_R_ind = torch.arange(cp_epoch + 1 - patience, cp_epoch + 1)
            strip_L_ind[strip_L_ind < 0] = 0
            strip_R_ind[strip_R_ind < 0] = 0
            strip_L = stoppings['D_val'][strip_L_ind]      # Causing the tcmalloc crashes on Colab
            strip_R = stoppings['D_val'][strip_R_ind]
            for i in range(patience):
                x = [strip_L_ind[i], strip_R_ind[i]]
                y = [strip_L[i], strip_R[i]]
                if strip_L[i] < strip_R[i]:
                    color = 'tab:blue'
                else:
                    color = 'tab:red'
                ax5[0].plot(x, y, color = color)

        ax5[0].legend(loc = 'upper right', fontsize = 15)
        ax5[1].legend(loc = 'upper left', fontsize = 15)
        
        fig5.tight_layout()
        fig5.savefig(f'{self.project_path}logs/{self.model_name}/graphs/stoppings.png', bbox_inches='tight', bbox_extra_artists=[suptitle5])
        plt.show(fig5)

    # barplot with cmap for finetune
    # x: 10 ranges of hparam
    # y: avg score (target) of each range
    # color: frequency of hparam in range
#
#
#
#
##################################################
# Test Bayesian Optimization
# from bayes_opt import BayesianOptimization
# from bayes_opt import UtilityFunction
# import torch

# def black_box_function(x, y, noise = 0):
    # return -x ** 2 - (y - 1) ** 2 + 1 - noise + 2*noise*torch.rand(1)

# num_trials = 100
# hparams = {'x': torch.zeros(num_trials),
           # 'y': torch.zeros(num_trials)}
# target = torch.zeros(num_trials)
# cp_trial = 0

# def finetune(bayesian, target, cp_trial, black_box_function):
    # # Init B_optimizer
    # optimizer = BayesianOptimization(
        # f=black_box_function,
        # pbounds={'x': (-5, 5), 'y': (-5, 5)},
        # verbose=2,
    # )
    # utility = UtilityFunction(kind="ucb", kappa=10, xi=0.1)
    
    # # Reload B_optimizer
    # for trial in range(cp_trial):
        # optimizer.register(
            # params = {
                # 'x': hparams['x'][trial],
                # 'y': hparams['y'][trial]
            # },
            # target = target[trial])

    # # Suggest
    # trial_hparams = optimizer.suggest(utility)

    # # Evaluate
    # trial_target = black_box_function(noise = 0.1, **trial_hparams)
    
    # hparams['x'][cp_trial] = trial_hparams['x']
    # hparams['y'][cp_trial] = trial_hparams['y']
    # target[cp_trial] = trial_target

    # cp_trial += 1
    # return bayesian, target, cp_trial

# for _ in range(num_trials):
    # hparams, target, cp_trial = finetune(hparams, target, cp_trial, black_box_function)
# print(hparams['x'])
# print(hparams['y'])
# print(target)
# print(cp_trial)
# print(target.max())

# from matplotlib.pyplot import colorbar, pcolor, show

# fig, ax = plt.subplots(1, 4, figsize = (20, 5))

# def smoothen(var, momentum):
    # smooth_var = torch.zeros(size = var.size())
    # for i in range(num_trials):
        # if i == 0:
            # smooth_var[i] =  var[i]   
        # elif i >= 0:
            # smooth_var[i] = smooth_var[i-1]*momentum + (1 - momentum)*var[i]
    # return smooth_var

# ax[0].plot(hparams['x'])
# ax[0].plot(smoothen(var = hparams['x'], momentum = 0.8))

# ax[1].plot(hparams['y'])
# ax[1].plot(smoothen(var = hparams['y'], momentum = 0.8))

# ax[2].plot(target)
# ax[2].plot(smoothen(var = target, momentum = 0.8))

# alpha = torch.arange(start = 0, end = 1, step = 1/100)

# x_raw = torch.linspace(start = -5, end = 5, steps = 101)
# x = x_raw.unsqueeze(dim = 0).transpose(0, 1)
# y = torch.linspace(start = -5, end = 5, steps = 101)
# z = -x ** 2 - (y - 1) ** 2 + 1

# ax[3].pcolormesh(x_raw, y, z) # cmap = 'Blues'
# ax[3].scatter(hparams['x'], hparams['y'], color = 'black', s = 5)

##################################################
#@title plot scheduler
# import torch
# import matplotlib.pyplot as plt

# model = torch.nn.Linear(2, 1)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=5,mode="triangular")
# lrs = []


# for epoch in range(10):
    # for batch in range(10):
        # optimizer.step()
        # lrs.append(optimizer.param_groups[0]["lr"])
        # #print("Learning Rate = ",optimizer.param_groups[0]["lr"])
        # scheduler.step()

# plt.plot(lrs)