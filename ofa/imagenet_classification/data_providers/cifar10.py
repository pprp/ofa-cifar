import warnings
import os
import math
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms.transforms import RandomCrop

from .base_provider import DataProvider
from ofa.utils.my_dataloader import MyRandomResizedCrop, MyDistributedSampler

import random
import numpy as np
import scipy
from scipy import ndimage
from PIL import Image, ImageEnhance, ImageOps
__all__ = ['CifarDataProvider']

class AutoAugment(object):
    def __init__(self):
        self.policies = [
            ['Invert', 0.1, 7, 'Contrast', 0.2, 6],
            ['Rotate', 0.7, 2, 'TranslateX', 0.3, 9],
            ['Sharpness', 0.8, 1, 'Sharpness', 0.9, 3],
            ['ShearY', 0.5, 8, 'TranslateY', 0.7, 9],
            ['AutoContrast', 0.5, 8, 'Equalize', 0.9, 2],
            ['ShearY', 0.2, 7, 'Posterize', 0.3, 7],
            ['Color', 0.4, 3, 'Brightness', 0.6, 7],
            ['Sharpness', 0.3, 9, 'Brightness', 0.7, 9],
            ['Equalize', 0.6, 5, 'Equalize', 0.5, 1],
            ['Contrast', 0.6, 7, 'Sharpness', 0.6, 5],
            ['Color', 0.7, 7, 'TranslateX', 0.5, 8],
            ['Equalize', 0.3, 7, 'AutoContrast', 0.4, 8],
            ['TranslateY', 0.4, 3, 'Sharpness', 0.2, 6],
            ['Brightness', 0.9, 6, 'Color', 0.2, 8],
            ['Solarize', 0.5, 2, 'Invert', 0.0, 3],
            ['Equalize', 0.2, 0, 'AutoContrast', 0.6, 0],
            ['Equalize', 0.2, 8, 'Equalize', 0.6, 4],
            ['Color', 0.9, 9, 'Equalize', 0.6, 6],
            ['AutoContrast', 0.8, 4, 'Solarize', 0.2, 8],
            ['Brightness', 0.1, 3, 'Color', 0.7, 0],
            ['Solarize', 0.4, 5, 'AutoContrast', 0.9, 3],
            ['TranslateY', 0.9, 9, 'TranslateY', 0.7, 9],
            ['AutoContrast', 0.9, 2, 'Solarize', 0.8, 3],
            ['Equalize', 0.8, 8, 'Invert', 0.1, 3],
            ['TranslateY', 0.7, 9, 'AutoContrast', 0.9, 1],
        ]

    def __call__(self, img):
        img = apply_policy(img, self.policies[random.randrange(len(self.policies))])
        return img


operations = {
    'ShearX': lambda img, magnitude: shear_x(img, magnitude),
    'ShearY': lambda img, magnitude: shear_y(img, magnitude),
    'TranslateX': lambda img, magnitude: translate_x(img, magnitude),
    'TranslateY': lambda img, magnitude: translate_y(img, magnitude),
    'Rotate': lambda img, magnitude: rotate(img, magnitude),
    'AutoContrast': lambda img, magnitude: auto_contrast(img, magnitude),
    'Invert': lambda img, magnitude: invert(img, magnitude),
    'Equalize': lambda img, magnitude: equalize(img, magnitude),
    'Solarize': lambda img, magnitude: solarize(img, magnitude),
    'Posterize': lambda img, magnitude: posterize(img, magnitude),
    'Contrast': lambda img, magnitude: contrast(img, magnitude),
    'Color': lambda img, magnitude: color(img, magnitude),
    'Brightness': lambda img, magnitude: brightness(img, magnitude),
    'Sharpness': lambda img, magnitude: sharpness(img, magnitude),
}

def apply_policy(img, policy):
    if random.random() < policy[1]:
        img = operations[policy[0]](img, policy[2])
    if random.random() < policy[4]:
        img = operations[policy[3]](img, policy[5])

    return img


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = offset_matrix @ matrix @ reset_matrix
    return transform_matrix


def shear_x(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-0.3, 0.3, 11)

    transform_matrix = np.array([[1, random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]), 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def shear_y(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-0.3, 0.3, 11)

    transform_matrix = np.array([[1, 0, 0],
                                 [random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]), 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def translate_x(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-150/331, 150/331, 11)

    transform_matrix = np.array([[1, 0, 0],
                                 [0, 1, img.shape[1]*random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def translate_y(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-150/331, 150/331, 11)

    transform_matrix = np.array([[1, 0, img.shape[0]*random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])],
                                 [0, 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def rotate(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-30, 30, 11)
    theta = np.deg2rad(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    transform_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def auto_contrast(img, magnitude):
    img = ImageOps.autocontrast(img)
    return img


def invert(img, magnitude):
    img = ImageOps.invert(img)
    return img


def equalize(img, magnitude):
    img = ImageOps.equalize(img)
    return img


def solarize(img, magnitude):
    magnitudes = np.linspace(0, 256, 11)
    img = ImageOps.solarize(img, random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    return img


def posterize(img, magnitude):
    magnitudes = np.linspace(4, 8, 11)
    img = ImageOps.posterize(img, int(round(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))))
    return img


def contrast(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    img = ImageEnhance.Contrast(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    return img


def color(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    img = ImageEnhance.Color(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    return img


def brightness(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    return img


def sharpness(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    return img

class CifarDataProvider(DataProvider):
	DEFAULT_PATH = '~/datasets/cifar10'

	def __init__(self, save_path=None, train_batch_size=64, test_batch_size=128, valid_size=None, n_worker=4,
	             resize_scale=0.08, distort_color=None, image_size=32,
	             num_replicas=None, rank=None):

		warnings.filterwarnings('ignore')
		self._save_path = save_path

		self.image_size = image_size  # int or list of int
		# self.distort_color = 'None' if distort_color is None else distort_color
		self.resize_scale = resize_scale

		self._valid_transform_dict = {}
		if not isinstance(self.image_size, int):
			from ofa.utils.my_dataloader import MyDataLoader
			assert isinstance(self.image_size, list)
			self.image_size.sort()  # e.g., 160 -> 32
			MyRandomResizedCrop.IMAGE_SIZE_LIST = self.image_size.copy()
			MyRandomResizedCrop.ACTIVE_SIZE = max(self.image_size)

			for img_size in self.image_size:
				self._valid_transform_dict[img_size] = self.build_valid_transform(img_size)
			self.active_img_size = max(self.image_size)  # active resolution for test
			valid_transforms = self._valid_transform_dict[self.active_img_size]
			train_loader_class = MyDataLoader  # randomly sample image size for each batch of training image
		else:
			self.active_img_size = self.image_size
			valid_transforms = self.build_valid_transform()
			train_loader_class = torch.utils.data.DataLoader

		train_dataset = self.train_dataset(self.build_train_transform())

		if valid_size is not None:
			if not isinstance(valid_size, int):
				assert isinstance(valid_size, float) and 0 < valid_size < 1
				valid_size = int(len(train_dataset) * valid_size)

			valid_dataset = self.train_dataset(valid_transforms)
			train_indexes, valid_indexes = self.random_sample_valid_set(len(train_dataset), valid_size)

			if num_replicas is not None:
				train_sampler = MyDistributedSampler(train_dataset, num_replicas, rank, True, np.array(train_indexes))
				valid_sampler = MyDistributedSampler(valid_dataset, num_replicas, rank, True, np.array(valid_indexes))
			else:
				train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
				valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)

			self.train = train_loader_class(
				train_dataset, batch_size=train_batch_size, sampler=train_sampler,
				num_workers=n_worker, pin_memory=True,
			)
			self.valid = torch.utils.data.DataLoader(
				valid_dataset, batch_size=test_batch_size, sampler=valid_sampler,
				num_workers=n_worker, pin_memory=True,
			)
		else:
			if num_replicas is not None:
				train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas, rank)
				self.train = train_loader_class(
					train_dataset, batch_size=train_batch_size, sampler=train_sampler,
					num_workers=n_worker, pin_memory=True
				)
			else:
				self.train = train_loader_class(
					train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=n_worker, pin_memory=True,
				)
			self.valid = None

		test_dataset = self.test_dataset(valid_transforms)
		if num_replicas is not None:
			test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas, rank)
			self.test = torch.utils.data.DataLoader(
				test_dataset, batch_size=test_batch_size, sampler=test_sampler, num_workers=n_worker, pin_memory=True,
			)
		else:
			self.test = torch.utils.data.DataLoader(
				test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=n_worker, pin_memory=True,
			)

		if self.valid is None:
			self.valid = self.test

	@staticmethod
	def name():
		return 'cifar10'

	@property
	def data_shape(self):
		return 3, self.active_img_size, self.active_img_size  # C, H, W

	@property
	def n_classes(self):
		return 10

	@property
	def save_path(self):
		if self._save_path is None:
			self._save_path = self.DEFAULT_PATH
			if not os.path.exists(self._save_path):
				self._save_path = os.path.expanduser('~/data/cifar10')
		return self._save_path

	@property
	def data_url(self):
		raise ValueError('unable to download %s' % self.name())

	def train_dataset(self, _transforms):
		return datasets.CIFAR10(self.train_path, train=True, transform=_transforms,download=True)

	def test_dataset(self, _transforms):
		return datasets.CIFAR10(self.valid_path, train=False, transform=_transforms,download=True)

	@property
	def train_path(self):
		return os.path.join(self.save_path, 'train')

	@property
	def valid_path(self):
		return os.path.join(self.save_path, 'val')

	@property
	def normalize(self):
		return transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

	def build_train_transform(self, image_size=None, print_log=True):
		if image_size is None:
			image_size = self.image_size

		# random_resize_crop -> random_horizontal_flip
		train_transforms = [
			transforms.RandomCrop(32,padding=4),
			transforms.RandomHorizontalFlip(),
			# AutoAugment(),
		]
		
		train_transforms += [
			transforms.ToTensor(),
			self.normalize,
		]

		train_transforms = transforms.Compose(train_transforms)
		return train_transforms

	def build_valid_transform(self, image_size=None):
		if image_size is None:
			image_size = self.active_img_size

		return transforms.Compose([
			transforms.ToTensor(),
			self.normalize,
		])

	def assign_active_img_size(self, new_img_size):
		self.active_img_size = new_img_size
		if self.active_img_size not in self._valid_transform_dict:
			self._valid_transform_dict[self.active_img_size] = self.build_valid_transform()
		# change the transform of the valid and test set
		self.valid.dataset.transform = self._valid_transform_dict[self.active_img_size]
		self.test.dataset.transform = self._valid_transform_dict[self.active_img_size]

	def build_sub_train_loader(self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None):
		# used for resetting BN running statistics
		if self.__dict__.get('sub_train_%d' % self.active_img_size, None) is None:
			if num_worker is None:
				num_worker = self.train.num_workers

			n_samples = len(self.train.dataset)
			g = torch.Generator()
			g.manual_seed(DataProvider.SUB_SEED)
			rand_indexes = torch.randperm(n_samples, generator=g).tolist()

			new_train_dataset = self.train_dataset(
				self.build_train_transform(image_size=self.active_img_size, print_log=False))
			chosen_indexes = rand_indexes[:n_images]
			if num_replicas is not None:
				sub_sampler = MyDistributedSampler(new_train_dataset, num_replicas, rank, True, np.array(chosen_indexes))
			else:
				sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
			sub_data_loader = torch.utils.data.DataLoader(
				new_train_dataset, batch_size=batch_size, sampler=sub_sampler,
				num_workers=num_worker, pin_memory=True,
			)
			self.__dict__['sub_train_%d' % self.active_img_size] = []
			for images, labels in sub_data_loader:
				self.__dict__['sub_train_%d' % self.active_img_size].append((images, labels))
		return self.__dict__['sub_train_%d' % self.active_img_size]
