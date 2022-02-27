import torch
import torchvision

import logging
import modules.logconf
import pretrainedmodels

class ModelClass:
    def __init__(self, project_path):
        # Init logger
        self.project_path = project_path
        self.logger = logging.getLogger(__name__)
        modules.logconf.initLogger(self.logger, project_path = self.project_path)
        # Use CUDA
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.logger.info(f"Using {self.device} device.")
        self.logger.debug('[CMPL] Initialized ModelClass - fix 14')
        self.model_names = [
            'ResNet18',
            'ResNet50',
            'Xception',
            'DenseNet121',
            'Inception_v4',
            'DualPathNet68',
            # For Mobile
            'NASNet_A_Mobile',
            'MobileNet_v3_large',
            'MobileNet_v3_small',
            'MNASNet_1_0',
            'MNASNet_0_5'
        ]
        
    def initModel(self, model_name):
        if model_name in self.model_names:
            model = torch.load(f'{self.project_path}models/{model_name}.pt')
        else:
            self.logger.error(f'[FAIL] Model {model_name} not available.')
            print(f'Please try one in {self.model_names}')
            return None
        
        model.name = model_name
        model = model.to(self.device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.debug(f'[CMPL] Initialized model {model.name} with {num_params} parameters, moved to {self.device}')
        return model