import glob
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

class ImageFolder(Dataset):
    '''三个函数：构造函数、len、getitem函数'''
    def __init__(
        self,
        dataset,#文件名
        data_dir,#数据路径
        transform#图像预处理
    ):
        super(ImageFolder, self).__init__()
        if dataset == 'ImageNet':
            self.fnames = list(glob.glob(data_dir + '/train/*/*.JPEG'))
        elif dataset == 'COCO':
            self.fnames = list(glob.glob(data_dir + '/train2017/*.jpg'))
        elif dataset == 'COCOplus':
            self.fnames = list(glob.glob(data_dir + '/train2017/*.jpg')) + list(glob.glob(data_dir + '/unlabeled2017/*.jpg'))
        elif dataset == 'COCOval':
            self.fnames = list(glob.glob(data_dir + '/val2017/*.jpg'))
        elif dataset == 'CLEVRtest':
            print('loading')
            self.fnames = list(glob.glob(data_dir + '/CLEVR_v1.0/images/1/*.png'))
        elif dataset == 'CLEVRval':
            self.fnames = list(glob.glob(data_dir + '/CLEVR_v1.0/images/val/*.png'))
        else:
            raise NotImplementedError

        self.fnames = np.array(self.fnames) # to avoid memory leak
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        #基于索引返回样本，用ids得到对应的image
        fpath = self.fnames[idx]
        image = Image.open(fpath).convert('RGB')
        return self.transform(image)#输出预处理后的image
