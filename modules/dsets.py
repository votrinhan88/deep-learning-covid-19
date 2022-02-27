import torch
import torchvision
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import PIL

from tqdm.auto import tqdm
import os
import logging
import modules.logconf

# Split dataset
# torch.tensor([126, 18, 40]), torch.tensor([3423, 489, 1086])
# (tensor([7.0000, 1.0000, 2.2222]), tensor([7.0000, 1.0000, 2.2209]))

fix = 47

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, subset, transform = None):
        super(ImageDataset, self).__init__()
        self.subset = subset
        self.root_dir = data_path + subset
        self.transform = transform
        self.logger = logging.getLogger(__name__)
        modules.logconf.initLogger(self.logger, project_path = self.root_dir)
        
        # Sets are defined in super-script
        self.class_names = os.listdir(os.path.join(self.root_dir))
        self.name_2_label = {'non': 0, 'covid': 1}
        self.data = []
        self.class_dist = {}
        
        for class_name in self.class_names:
            checkpoint_len = len(self.data)
            if ((self.subset == 'test') & (class_name == 'non')):
                # test/non/ has 13 categories, need to access deeper
                self.test_non_categories = os.listdir(os.path.join(self.root_dir, class_name))
                for index, category in enumerate(self.test_non_categories):
                    files = os.listdir(os.path.join(self.root_dir, class_name, category))
                    
                    self.data += list(zip(files, [class_name]*len(files), [category]*len(files)))
            else:
                files = os.listdir(os.path.join(self.root_dir, class_name))
                self.data += list(zip(files, [class_name]*len(files), [None]*len(files)))
                
            self.class_dist[class_name] = len(self.data) - checkpoint_len
        self.logger.info(f'[CMPL] Loaded {self.subset} subset')
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        image_path, class_name, category = self.data[index]
        
        # If test/non
        if ((self.subset == 'test') & (class_name == 'non')):
            root_and_dir = os.path.join(self.root_dir, class_name, category)
        else:
            root_and_dir = os.path.join(self.root_dir, class_name)
        
        # Convert to RGB channels using PIL. If use OpenCV:
        #image = cv2.cvtColor(cv2.imread(os.path.join(root_and_dir, image_path)), cv2.COLOR_BGR2RGB)
        image = np.array(PIL.Image.open(os.path.join(root_and_dir, image_path)).convert('RGB'))
        
        if self.transform is not None:
            augmentation = self.transform(image = image)
            image = augmentation["image"]
        
        label = torch.tensor(self.name_2_label[class_name])
        
        return image, label
        
    def getRawItem(self, index):
        image_path, class_name, category = self.data[index]
        
        # If test/non
        if ((self.subset == 'test') & (class_name == 'non')):
            root_and_dir = os.path.join(self.root_dir, class_name, category)
        else:
            root_and_dir = os.path.join(self.root_dir, class_name)
        
        # Convert to RGB channels using PIL. If use OpenCV:
        #image = cv2.cvtColor(cv2.imread(os.path.join(root_and_dir, image_path)), cv2.COLOR_BGR2RGB)
        image = np.array(PIL.Image.open(os.path.join(root_and_dir, image_path)).convert('RGB'))
        
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = torch.tensor(self.name_2_label[class_name])
        
        return image, label
        
    def getClassDistribution(self):
        return [self.class_dist['non'], self.class_dist['covid']]
    
class DatasetClass:
    def __init__(self, project_path, batch_size, num_workers, folder):
        # Init logger
        self.project_path = project_path
        
        if folder == 'raw':
            self.data_path = self.project_path + 'data/COVID-XRay-5K v3/data_upload_v3/'
        elif folder == 'pseudo':
            self.data_path = self.project_path + 'data/COVID-XRay-5K v3/data_upload_v3_pseudo/'
        elif folder == 'thesis':
            self.data_path = self.project_path + 'data/COVID-XRay-5K v3/data_upload_v3_thesis/'
            
        self.logger = logging.getLogger(__name__)
        modules.logconf.initLogger(logger = self.logger, project_path = self.project_path)
        
        # Handle parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.subsets = ['train', 'test']
        self.logger.info(f'Hyperparameters: batch_size = {self.batch_size}, num_workers = {self.num_workers}')
        self.logger.debug(f'[CMPL] Initialized DatasetClass {folder} - fix {fix}')
        
    def initTransform(self):
        # Transformation (image augmentation)
        # albu.Normalize: img = (img - mean * max_pixel_value) / (std * max_pixel_value)
        # ImageNet normalization: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
        self.transform = {
            'train': albu.Compose([
                albu.Resize(256, 256, p = 1),
                albu.RandomCrop(224, 224, p = 1),
                albu.Rotate(limit = 10, p = 1.0),
                albu.HorizontalFlip(p = 0.5),
                #albu.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p = 0.9),
                #albu.OneOf([
                #    albu.Blur(blur_limit = 3, p = 1.0),
                #    albu.ElasticTransform(p = 1.0),
                #    albu.GridDistortion(p = 1.0),
                #    #albu.ChannelShuffle(p = 1.0)
                #    #albu.ColorJitter(p = 0.5),
                #], p = 1.0),
                albu.Normalize(
                    mean = [0.6629, 0.6629, 0.6629],
                    std = [0.0675, 0.0674, 0.0673],
                    max_pixel_value = 255
                ),
                ToTensorV2()
            ]),
            'val': albu.Compose([
                albu.Resize(256, 256, p = 1),
                albu.CenterCrop(224, 224, p = 1),
                albu.Normalize(
                    mean = [0.6629, 0.6629, 0.6629],
                    std = [0.0675, 0.0674, 0.0673],
                    max_pixel_value = 255
                ),
                ToTensorV2()
            ]),
            'test': albu.Compose([
                albu.Resize(256, 256, p = 1),
                albu.CenterCrop(224, 224, p = 1),
                albu.Normalize(
                    mean = [0.6629, 0.6629, 0.6629],
                    std = [0.0675, 0.0674, 0.0673],
                    max_pixel_value = 255
                ),
                ToTensorV2()
            ])
        }
        self.logger.debug('[CMPL] Initialized normalized transforms')
    
    def loadDataset(self, subset):
        self.subset = subset
        self.logger.debug('Loading dataset...')
        
        self.initTransform()
        
        imageDataset    = ImageDataset(data_path    = self.data_path,
                                       subset       = self.subset,
                                       transform    = self.transform[subset])

        return imageDataset
    
    def initTransform_raw(self):
        self.transform_raw = {
            'train': albu.Compose([
                albu.Resize(256, 256, p = 1),
                albu.Normalize(
                    mean = [0, 0, 0],
                    std = [1, 1, 1],
                    max_pixel_value = 255
                ),
                ToTensorV2()
            ]),
            'val': albu.Compose([
                albu.Resize(256, 256, p = 1),
                albu.Normalize(
                    mean = [0, 0, 0],
                    std = [1, 1, 1],
                    max_pixel_value = 255
                ),
                ToTensorV2()
            ]),
            'test': albu.Compose([
                albu.Resize(256, 256, p = 1),
                albu.Normalize(
                    mean = [0, 0, 0],
                    std = [1, 1, 1],
                    max_pixel_value = 255
                ),
                ToTensorV2()
            ])
        }
        self.logger.debug('[CMPL] Initialized raw transforms')
        
    def getNormalizeParameters(self):
        self.logger.debug('[CMPL] Calculating normalize parameters ...')
        self.initTransform_raw()
        imageDatasets_raw = {}
        dataloaders_raw = {}
        channels_sum = torch.zeros(3)
        channels_squared_sum = torch.zeros(3)
        
        def worker_init_fn(worker_id):
            process_seed = torch.initial_seed()
            # Back out the base_seed so we can use all the bits.
            base_seed = process_seed - worker_id
            ss = np.random.SeedSequence([worker_id, base_seed])
            # More than 128 bits (4 32-bit words) would be overkill.
            np.random.seed(ss.generate_state(4))
        generator = torch.Generator()
        
        for subset in self.subsets:
            imageDatasets_raw[subset] = ImageDataset(root_dir = self.data_path + subset, transform = self.transform_raw[subset])
            dataloaders_raw[subset] = torch.utils.data.DataLoader(imageDatasets_raw[subset],
                                                                  batch_size     = self.batch_size,
                                                                  shuffle        = False,
                                                                  num_workers    = self.num_workers,
                                                                  worker_init_fn = worker_init_fn,
                                                                  generator      = generator)
            for image, _ in tqdm(dataloaders_raw[subset], desc = subset, unit = 'batch'):
                channels_sum += torch.sum(image, dim = [0, 2, 3])
                channels_squared_sum += torch.sum(image**2, dim = [0, 2, 3])
        
        total_pixels = sum(self.dataset_sizes.values()) * 224**2
        mean = channels_sum/total_pixels
        std = torch.sqrt((channels_squared_sum)/total_pixels - (mean**2))
        
        self.logger.debug('[CMPL] Calculated normalize parameters')
        #(tensor([0.6629, 0.6629, 0.6629]), tensor([0.0675, 0.0674, 0.0673]))
        return mean, std
        
    def show_augment(self, subset, num_examples, num_augments):
        self.initTransform_raw()
        self.initTransform()
        
        imageDataset_raw = ImageDataset(data_path = self.data_path,
                                        subset    = subset,
                                        transform = self.transform_raw[subset])
        imageDataset = ImageDataset(data_path = self.data_path,
                                    subset    = subset,
                                    transform = self.transform[subset])
        
        num_rows = num_examples
        num_cols = num_augments + 1
        
        augment = torch.zeros(num_rows, num_cols, 224, 224, 3)
        random_example = torch.randint(high = imageDataset.__len__(), size = (num_examples,))
        
        # Get images
        for example_id in range(num_rows):
            for augment_ind in range(num_cols):
                # Get raw image for first column and augmentations for others
                if (augment_ind == 0):
                    image, _ = imageDataset_raw.__getitem__(random_example[example_id])
                    # Resize raw item
                    image = torch.nn.functional.interpolate(image.unsqueeze(0), size = [224, 224])
                else:
                    image, _ = imageDataset.__getitem__(random_example[example_id])
                augment[example_id, augment_ind, :, :, :] = image.detach().squeeze().permute(1, 2, 0)
        # Preprocess to hide warnings
        augment = torch.clip(augment, min = 0, max = 1)
        # Plot
        figure, axis = plt.subplots(nrows        = num_rows,
                                    ncols        = num_cols,
                                    gridspec_kw  = {'wspace': -0.2, 'hspace':0.15},
                                    squeeze      = True,
                                    tight_layout = True,
                                    figsize      = (6, 6),
                                    dpi          = 120)
        # Row
        for example_id in range(num_rows):
            # Col
            for augment_ind in range(num_cols):
                axis[example_id, augment_ind].set_axis_off()
                axis[example_id, augment_ind].imshow(augment[example_id, augment_ind, :, :, :])
                axis[example_id, augment_ind].set_title(f'{subset[0:2].capitalize()}_E{int(random_example[example_id])}_A{augment_ind}', fontsize = 8, y = 0.98)

        plt.show()
        
        self.logger.debug(f'[CMPL] Performed {num_augments} augmentations of {num_examples} examples in {subset} subset')

    def testGetAndForward(self, subset, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        item_test_flags = torch.zeros(len(self.subsets), 3)

        for subset_ind, subset in enumerate(self.subsets):
            # Get
            image, label = self.imageDatasets[subset].__getitem__(torch.randint(self.dataset_sizes[subset], (1,)))
            # Forward
            output = model(image.unsqueeze(0).to(device))
            # Flags
            images_flag = (image.size() == torch.Size([3, 224, 224]))
            labels_flag = (label.size() == torch.Size([]))
            outputs_flag = (output.size() == torch.Size([1, 2]))
            item_test_flags[subset_ind, :] = torch.tensor([images_flag, labels_flag, outputs_flag])

        if (item_test_flags.prod() == True):
            self.logger.debug(f'[CMPL] Unit-tested __getitem__ and forward() for one item of {self.subsets} subsets')
        print(f'Example item outputs:\n{output}')

        num_batches = 5

        batch_test_flags = torch.zeros(len(self.subsets), num_batches, 3)

        for subset_ind, subset in enumerate(self.subsets):
            for ind_batch, test_images in enumerate(self.dataloaders[subset]):
                if ind_batch == num_batches:
                    break
                else:
                    # Forward
                    images = test_images[0].to(device)
                    labels = test_images[-1].to(device)
                    outputs = model(images)
                    # Flags
                    images_flag = (images.size() == torch.Size([self.batch_size, 3, 224, 224]))
                    labels_flag = (labels.size() == torch.Size([self.batch_size]))
                    outputs_flag = (outputs.size() == torch.Size([self.batch_size, 2]))
                    batch_test_flags[subset_ind, ind_batch, :] = torch.tensor([images_flag, labels_flag, outputs_flag])

        if (batch_test_flags.prod() == True):
            self.logger.debug(f'[CMPL] Tested dataloader and forward for {num_batches} batches of {self.subsets} subsets')

        print(f'Example batch outputs:\n{outputs}')

#
#
#
#