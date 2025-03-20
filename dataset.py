import numpy
from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
from skimage import filters, restoration

class BilateralFilter(torch.nn.Module):
    def __init__(self, d, sigmaColor=None, sigmaSpace=1):
        super(BilateralFilter, self).__init__()
        self.d = d
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace

    def forward(self, x):
        """
        :param x: PIL Image
        :return: PIL Image
        """
        filtered = restoration.denoise_bilateral(np.array(x), self.d, self.sigmaColor, self.sigmaSpace, multichannel=True)
        return Image.fromarray((filtered*255).astype(np.uint8))

    def __repr__(self):
        return f"{self.__class__.__name__}(d={self.d}, sigmaColor={self.sigmaColor}, sigmaSpace={self.sigmaSpace})"

class WaveletFilter(torch.nn.Module):
    def __init__(self, wavelet='db1'):
        super(WaveletFilter, self).__init__()
        self.wavelet = wavelet

    def forward(self, x):
        """
        :param x: PIL Image
        :return: PIL Image
        """
        filtered = restoration.denoise_wavelet(np.array(x), wavelet=self.wavelet, multichannel=True)
        return Image.fromarray((filtered*255).astype(np.uint8))

    def __repr__(self):
        return f"{self.__class__.__name__}(wavelet={self.wavelet})"

class FrangiFilter(torch.nn.Module):
    def __init__(self):
        super(FrangiFilter, self).__init__()

    def forward(self, x):
        """
        :param x: PIL Image
        :return: PIL Image
        """
        filtered = filters.frangi(np.array(x))
        return Image.fromarray((filtered*255).astype(np.uint8))

    def __repr__(self):
        return f"{self.__class__.__name__}()"


def get_data_transforms(size, isize, mean_std=None, filter=None):
    if not mean_std:
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    assert len(mean_std) == 2, "mean_std should has mean and std."
    assert len(mean_std[0]) == 3, "mean should has 3 values."
    assert len(mean_std[1]) == 3, "std should has 3 values."

    if filter:
        data_transforms = transforms.Compose([
            transforms.Resize((size, size)),
            filter,
            transforms.ToTensor(),
            transforms.CenterCrop(isize),
            transforms.Normalize(mean=mean_std[0], std=mean_std[1])])
    else:
        data_transforms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.CenterCrop(isize),
            transforms.Normalize(mean=mean_std[0], std=mean_std[1])])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

def train_collate(batch):
    img = torch.stack([item[0] for item in batch])
    label = torch.tensor([item[2] for item in batch])
    return img, label

def path_format(path:str ) -> str:
    return path.replace('\\', '/')

class GFCDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_size, phase, transform=None, filter=None):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
            # self.train = True
        else:
            self.img_path = os.path.join(root, 'test')
            # self.train = False
        self.img_paths = root
        self.mean, self.std = None, None
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        if transform:
            self.transform, self.gt_transform = transform
        elif filter:
            self.transform, self.gt_transform = get_data_transforms(image_size, image_size, (self.mean, self.std), filter)
        else:
            self.transform, self.gt_transform = get_data_transforms(image_size, image_size, (self.mean, self.std))
        # load dataset

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_paths = list(map(path_format, img_paths))
                img_paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
                img_tot_paths.extend(img_paths)
                tot_labels.extend([0] * len(img_paths))
                tot_types += (['good'] * len(img_paths))
            elif defect_type == 'defect':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_paths = list(map(path_format, img_paths))
                img_paths.sort(key=lambda x: int(x.split('/')[-1].split('_')[0]))
                img_tot_paths.extend(img_paths)
                tot_labels.extend([1] * len(img_paths))
                # type_tot_paths = glob.glob(os.path.join(self.img_path, 'label') + "/*.txt")
                # for type_path in type_tot_paths:
                #     with open(type_path, 'r') as f:
                #         lines = f.readlines()
                #         types = '-'.join([line.strip() for line in lines])
                #     tot_types += [types]

        #     Compute mean and std
        img_list_px = []
        for img_path in img_tot_paths:
            img = Image.open(img_path).convert('RGB')
            img_list_px.extend(np.array(img).ravel())
        self.mean = round(np.mean(img_list_px) / 255, 3)
        self.std = round(np.std(img_list_px) / 255, 3)
        self.mean = [self.mean] * 3
        self.std = [self.std] * 3

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        # gt = self.gt_transform(img.copy())
        # img = self.transform(img)
        # gt = gt.float().fill_(float('nan'))
        gt = torch.tensor(float('nan'))
        # if self.train:
        #     return img, None, None, None

        # '.bmp'
        img_type = img_path[:-4].split('\\')[-1]
        typ_pos = img_type.find('_')
        # good img
        if typ_pos == -1:
            img_type = 'good'
        else:
            img_type = img_type[typ_pos + 1:]

        return img, gt, label, img_type

    def get_meta_data(self):
        if self.mean:
            return self.mean, self.std
        else:
            return None



class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_size, phase, transform=None, filter=None):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
            # self.train = True
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
            # self.train = False
        self.mean, self.std = None, None
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        if transform:
            self.transform, self.gt_transform = transform
        elif filter:
            self.transform, self.gt_transform = get_data_transforms(image_size, image_size, filter=filter)
        else:
            self.transform, self.gt_transform = get_data_transforms(image_size, image_size)

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_paths = list(map(path_format, img_paths))
                img_paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths = list(map(path_format, img_paths))
                gt_paths = list(map(path_format, gt_paths))
                img_paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
                gt_paths.sort(key=lambda x: int(x.split('/')[-1].split('_')[0]))
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        # if self.train:
        #     return img, None, None, None
        if gt == 0:
            if not self.train:
                gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_type

    def get_meta_data(self):
        if self.mean:
            return self.mean, self.std
        else:
            return None

def load_data(dataset_name='mnist',normal_class=0,batch_size='16'):

    if dataset_name == 'cifar10':
        img_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            #transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        os.makedirs("./Dataset/CIFAR10/train", exist_ok=True)
        dataset = CIFAR10('./Dataset/CIFAR10/train', train=True, download=True, transform=img_transform)
        print("Cifar10 DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/CIFAR10/test", exist_ok=True)
        test_set = CIFAR10("./Dataset/CIFAR10/test", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)

    elif dataset_name == 'mnist':
        img_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        os.makedirs("./Dataset/MNIST/train", exist_ok=True)
        dataset = MNIST('./Dataset/MNIST/train', train=True, download=True, transform=img_transform)
        print("MNIST DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/MNIST/test", exist_ok=True)
        test_set = MNIST("./Dataset/MNIST/test", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)

    elif dataset_name == 'fashionmnist':
        img_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        os.makedirs("./Dataset/FashionMNIST/train", exist_ok=True)
        dataset = FashionMNIST('./Dataset/FashionMNIST/train', train=True, download=True, transform=img_transform)
        print("FashionMNIST DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/FashionMNIST/test", exist_ok=True)
        test_set = FashionMNIST("./Dataset/FashionMNIST/test", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)


    elif dataset_name == 'retina':
        data_path = 'Dataset/OCT2017/train'

        orig_transform = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.ToTensor()
        ])

        dataset = ImageFolder(root=data_path, transform=orig_transform)

        test_data_path = 'Dataset/OCT2017/test'
        test_set = ImageFolder(root=test_data_path, transform=orig_transform)

    else:
        raise Exception(
            "You enter {} as dataset, which is not a valid dataset for this repository!".format(dataset_name))

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
    )

    return train_dataloader, test_dataloader
