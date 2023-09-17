""" Dataloader for all datasets. """
import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class DatasetLoader(Dataset):
    """The class to load the dataset"""
    def __init__(self, setname, args, train_aug=False):
        # Set the path according to train, val and test        
        if setname=='train':
            THE_PATH = osp.join(args.dataset_dir, 'train')
            label_list = os.listdir(THE_PATH)
        elif setname=='test':
            THE_PATH = osp.join(args.dataset_dir, 'test')
            label_list = os.listdir(THE_PATH)
        elif setname=='val':
            THE_PATH = osp.join(args.dataset_dir, 'val')
            label_list = os.listdir(THE_PATH)
        else:
            raise ValueError('Wrong setname.') 

        self.setname = setname

        # Generate empty list for data and label           
        data = []
        label = []

        # Get folders' name
        folders = [osp.join(THE_PATH, the_label) for the_label in label_list if os.path.isdir(osp.join(THE_PATH, the_label))]

        # Get the images' paths and labels
        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        # Set data, label and class number to be accessable from outside
        self.data = data
        self.label = label
        self.num_class = len(set(label))

        # Transformation
        if train_aug:
            image_size = 224
            self.transform = transforms.Compose([
                transforms.Resize(225),
                transforms.RandomResizedCrop(image_size),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.0, 125.0, 125.0]]), 
                                     np.array([x / 255.0 for x in [62.5, 62.5, 62.5]]))])
        else:
            image_size = 224
            self.transform = transforms.Compose([
                transforms.Resize(225),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.0, 125.0, 125.0]]),
                                     np.array([x / 255.0 for x in [62.5, 62.5, 62.5]]))])

        self.transform_random = transforms.Compose([
            transforms.Resize(225),
            transforms.RandomResizedCrop(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(np.array([x / 255.0 for x in [125.0, 125.0, 125.0]]),
                                    np.array([x / 255.0 for x in [62.5, 62.5, 62.5]]))])



    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = Image.open(path).convert('RGB')
        if self.setname == 'train':
            img_0 = self.transform(image)
            img_1 = self.transform_random(image)
            img = [img_0, img_1]
        else:
            img = self.transform(image)
        return img, label
