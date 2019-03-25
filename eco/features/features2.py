import mxnet as mx
import numpy as np
import pickle
import os
import cv2
from mxnet.gluon.model_zoo import vision
from mxnet.gluon.nn import AvgPool2D
import mxnet as mx

from ..config import config
from . import _gradient


def mround(x):
    '''四舍五入'''
    x_ = x.copy()
    idx = (x - np.floor(x)) >= 0.5
    # np.floor返回不大于x的最大整数
    x_[idx] = np.floor(x[idx]) + 1
    idx = ~idx
    x_[idx] = np.floor(x[idx])
    return x_

class Feature:
    def init_size(self, img_sample_sz, cell_size=None):
        if cell_size is not None:
            max_cell_size = max(cell_size)
            new_img_sample_sz = (1 + 2 * mround(img_sample_sz / ( 2 * max_cell_size))) * max_cell_size
            feature_sz_choices = np.array([(new_img_sample_sz.reshape(-1, 1) + np.arange(0, max_cell_size).reshape(1, -1)) // x for x in cell_size])
            num_odd_dimensions = np.sum((feature_sz_choices % 2) == 1, axis=(0,1))
            best_choice = np.argmax(num_odd_dimensions.flatten())
            img_sample_sz = mround(new_img_sample_sz + best_choice)

        self.sample_sz = img_sample_sz
        self.data_sz = [img_sample_sz // self._cell_size]
        return img_sample_sz

    def _sample_patch(self, im, pos, sample_sz, output_sz):
        '''获取目标的图像'''
        pos = np.floor(pos)
        sample_sz = np.maximum(mround(sample_sz), 1)
        xs = np.floor(pos[1]) + np.arange(0, sample_sz[1]+1) - np.floor((sample_sz[1]+1)/2)
        ys = np.floor(pos[0]) + np.arange(0, sample_sz[0]+1) - np.floor((sample_sz[0]+1)/2)
        # 获得框的每一个位置的x和y
        # xs为x坐标的范围
        # ys为y坐标的范围
        xmin = max(0, int(xs.min()))
        xmax = min(im.shape[1], int(xs.max()))
        ymin = max(0, int(ys.min()))
        ymax = min(im.shape[0], int(ys.max()))
        # 将超出边界的地方压到边界上
        # extract image
        im_patch = im[ymin:ymax, xmin:xmax, :]
        # 提取目标图像
        left = right = top = down = 0
        if xs.min() < 0:
            left = int(abs(xs.min()))
        if xs.max() > im.shape[1]:
            right = int(xs.max() - im.shape[1])
        if ys.min() < 0:
            top = int(abs(ys.min()))
        if ys.max() > im.shape[0]:
            down = int(ys.max() - im.shape[0])
        if left != 0 or right != 0 or top != 0 or down != 0:
            im_patch = cv2.copyMakeBorder(im_patch, top, down, left, right, cv2.BORDER_REPLICATE)
            # 将刚才超出边界裁掉的部分填充回来，cv2.BORDER_REPLICATE填充方式，即边界像素向外延申
        # im_patch = cv2.resize(im_patch, (int(output_sz[0]), int(output_sz[1])))
        im_patch = cv2.resize(im_patch, (int(output_sz[0]), int(output_sz[1])), cv2.INTER_CUBIC)
        # 修改为期望输出的大小
        if len(im_patch.shape) == 2:
            im_patch = im_patch[:, :, np.newaxis]
        return im_patch

    def _feature_normalization(self, x):
        if hasattr(config, 'normalize_power') and config.normalize_power > 0:
            if config.normalize_power == 2:
                x = x * np.sqrt((x.shape[0]*x.shape[1]) ** config.normalize_size * (x.shape[2]**config.normalize_dim) / (x**2).sum(axis=(0, 1, 2)))
            else:
                x = x * ((x.shape[0]*x.shape[1]) ** config.normalize_size) * (x.shape[2]**config.normalize_dim) / ((np.abs(x) ** (1. / config.normalize_power)).sum(axis=(0, 1, 2)))

        if config.square_root_normalization:
            x = np.sign(x) * np.sqrt(np.abs(x))
        return x.astype(np.float32)

class CNNFeature(Feature):
    def _forward(self, x):
        pass

    def get_features(self, img, pos, sample_sz, scales):
        feat1 = []
        feat2 = []
        if img.shape[2] == 1:
            img = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2RGB)
        if not isinstance(scales, list) and not isinstance(scales, np.ndarray):
            scales = [scales]
        patches = []
        for scale in scales:
            patch = self._sample_patch(img, pos, sample_sz*scale, sample_sz)
            patch = mx.nd.array(patch / 255., ctx=self._ctx)
            normalized = mx.image.color_normalize(patch,
                                                  mean=mx.nd.array([0.485, 0.456, 0.406], ctx=self._ctx),
                                                  std=mx.nd.array([0.229, 0.224, 0.225], ctx=self._ctx))
            normalized = normalized.transpose((2, 0, 1)).expand_dims(axis=0)
            patches.append(normalized)
        patches = mx.nd.concat(*patches, dim=0)
        # patches初始化时(1,3,240,240)
        # patches在使用vgg16特征时为(5,3,240,240)
        print(patches.shape,'patches')
        f1, f2 = self._forward(patches)
        # 神经网络正向输出结果
        print(f1.shape,'f1')
        print(f2.shape,'f2')
        f1 = self._feature_normalization(f1)
        f2 = self._feature_normalization(f2)
        return f1, f2

class ResNet50Feature(CNNFeature):
    def __init__(self, fname, compressed_dim):
        self._ctx = mx.gpu(config.gpu_id) if config.use_gpu else mx.cpu(0)
        self._resnet50 = vision.resnet50_v2(pretrained=True, ctx = self._ctx)
        self._compressed_dim = compressed_dim
        self._cell_size = [4, 16]
        self.penalty = [0., 0.]
        self.min_cell_size = np.min(self._cell_size)

    def init_size(self, img_sample_sz, cell_size=None):
        # only support img_sample_sz square
        img_sample_sz = img_sample_sz.astype(np.int32)
        feat1_shape = np.ceil(img_sample_sz / 4)
        feat2_shape = np.ceil(img_sample_sz / 16)
        desired_sz = feat2_shape + 1 + feat2_shape % 2
        # while feat1_shape[0] % 2 == 0 or feat2_shape[0] % 2 == 0:
        #     img_sample_sz += np.array([1, 0])
        #     feat1_shape = np.ceil(img_sample_sz / 4)
        #     feat2_shape = np.ceil(img_sample_sz / 16)
        # while feat1_shape[1] % 2 == 0 or feat2_shape[1] % 2 == 0:
        #     img_sample_sz += np.array([0, 1])
        #     feat1_shape = np.ceil(img_sample_sz / 4)
        #     feat2_shape = np.ceil(img_sample_sz / 16)
        img_sample_sz = desired_sz * 16
        self.num_dim = [64, 1024]
        self.sample_sz = img_sample_sz
        self.data_sz = [np.ceil(img_sample_sz / 4),
                        np.ceil(img_sample_sz / 16)]
        return img_sample_sz

    def _forward(self, x):
        # stage1
        bn0 = self._resnet50.features[0].forward(x)
        conv1 = self._resnet50.features[1].forward(bn0)     # x2
        bn1 = self._resnet50.features[2].forward(conv1)
        relu1 = self._resnet50.features[3].forward(bn1)
        pool1 = self._resnet50.features[4].forward(relu1)   # x4
        # stage2
        stage2 = self._resnet50.features[5].forward(pool1)  # x4
        stage3 = self._resnet50.features[6].forward(stage2) # x8
        stage4 = self._resnet50.features[7].forward(stage3) # x16
        return [pool1.asnumpy().transpose(2, 3, 1, 0),
                stage4.asnumpy().transpose(2, 3, 1, 0)]

class VGG16Feature(CNNFeature):
    def __init__(self, fname, compressed_dim):
        self._ctx = mx.gpu(config.gpu_id) if config.use_gpu else mx.cpu(0)
        self._vgg16 = vision.vgg16(pretrained=True, ctx=self._ctx)
        self._compressed_dim = compressed_dim
        self._cell_size = [4, 16]
        self.penalty = [0., 0.]
        self.min_cell_size = np.min(self._cell_size)
        self._avg_pool2d = AvgPool2D()

    def init_size(self, img_sample_sz, cell_size=None):
        img_sample_sz = img_sample_sz.astype(np.int32)
        feat1_shape = np.ceil(img_sample_sz / 4)
        feat2_shape = np.ceil(img_sample_sz / 16)
        desired_sz = feat2_shape + 1 + feat2_shape % 2
        img_sample_sz = desired_sz * 16
        self.num_dim = [64, 512]
        self.sample_sz = img_sample_sz
        self.data_sz = [np.ceil(img_sample_sz / 4),
                        np.ceil(img_sample_sz / 16)]
        return img_sample_sz

    def _forward(self, x):
        # stage1
        conv1_1 = self._vgg16.features[0].forward(x)
        relu1_1 = self._vgg16.features[1].forward(conv1_1)
        conv1_2 = self._vgg16.features[2].forward(relu1_1)
        relu1_2 = self._vgg16.features[3].forward(conv1_2)
        pool1 = self._vgg16.features[4].forward(relu1_2) # x2
        pool_avg = self._avg_pool2d(pool1)
        # stage2
        conv2_1 = self._vgg16.features[5].forward(pool1)
        relu2_1 = self._vgg16.features[6].forward(conv2_1)
        conv2_2 = self._vgg16.features[7].forward(relu2_1)
        relu2_2 = self._vgg16.features[8].forward(conv2_2)
        pool2 = self._vgg16.features[9].forward(relu2_2) # x4
        # stage3
        conv3_1 = self._vgg16.features[10].forward(pool2)
        relu3_1 = self._vgg16.features[11].forward(conv3_1)
        conv3_2 = self._vgg16.features[12].forward(relu3_1)
        relu3_2 = self._vgg16.features[13].forward(conv3_2)
        conv3_3 = self._vgg16.features[14].forward(relu3_2)
        relu3_3 = self._vgg16.features[15].forward(conv3_3)
        pool3 = self._vgg16.features[16].forward(relu3_3) # x8
        # stage4
        conv4_1 = self._vgg16.features[17].forward(pool3)
        relu4_1 = self._vgg16.features[18].forward(conv4_1)
        conv4_2 = self._vgg16.features[19].forward(relu4_1)
        relu4_2 = self._vgg16.features[20].forward(conv4_2)
        conv4_3 = self._vgg16.features[21].forward(relu4_2)
        relu4_3 = self._vgg16.features[22].forward(conv4_3)
        pool4 = self._vgg16.features[23].forward(relu4_3) # x16
        return [pool_avg.asnumpy().transpose(2, 3, 1, 0),
                pool4.asnumpy().transpose(2, 3, 1, 0)]


def fhog(I, bin_size=8, num_orients=9, clip=0.2, crop=False):
    soft_bin = -1
    M, O = _gradient.gradMag(I.astype(np.float32), 0, True)
    H = _gradient.fhog(M, O, bin_size, num_orients, soft_bin, clip)
    return H


class FHogFeature(Feature):
    def __init__(self, fname, cell_size=6, compressed_dim=10, num_orients=9, clip=.2):
        self.fname = fname
        self._cell_size = cell_size
        self._compressed_dim = [compressed_dim]
        self._soft_bin = -1
        self._bin_size = cell_size
        self._num_orients = num_orients
        self._clip = clip

        self.min_cell_size = self._cell_size
        self.num_dim = [3 * num_orients + 5 - 1]
        self.penalty = [0.]

    def get_features(self, img, pos, sample_sz, scales):
        '''获取图像的Fhog特征'''
        feat = []
        if not isinstance(scales, list) and not isinstance(scales, np.ndarray):
            scales = [scales]
        for scale in scales:
            patch = self._sample_patch(img, pos, sample_sz*scale, sample_sz)
            # h, w, c = patch.shape
            M, O = _gradient.gradMag(patch.astype(np.float32), 0, True)
            H = _gradient.fhog(M, O, self._bin_size, self._num_orients, self._soft_bin, self._clip)
            # drop the last dimension
            H = H[:, :, :-1]
            feat.append(H)
        feat = self._feature_normalization(np.stack(feat, axis=3))
        return [feat]


class TableFeature(Feature):
    def __init__(self, fname, compressed_dim, table_name, use_for_color, cell_size=1):
        self.fname = fname
        self._table_name = table_name
        self._color = use_for_color
        self._cell_size = cell_size
        self._compressed_dim = [compressed_dim]
        self._factor = 32
        self._den = 8
        # load table
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # os.path.realpath返回当前运行文件真实目录
        # os.path.dirname返回当前目录的上级文件夹目录

        self._table = pickle.load(open(os.path.join(dir_path, "lookup_tables", self._table_name+".pkl"), "rb"))
        # 加载pickle
        # 其中CNnorm.pkl中包含(32768, 10)大小的矩阵

        self.num_dim = [self._table.shape[1]]
        # num_dim为10
        self.min_cell_size = self._cell_size
        # min_cell_size为4
        self.penalty = [0.]
        self.sample_sz = None
        self.data_sz = None


    def integralVecImage(self, img):
        w, h, c = img.shape
        intImage = np.zeros((w+1, h+1, c), dtype=img.dtype)
        intImage[1:, 1:, :] = np.cumsum(np.cumsum(img, 0), 1)
        # np.cumsum(array,axis)将array按照axis的方向叠加数值
        return intImage

    def average_feature_region(self, features, region_size):
        region_area = region_size ** 2
        if features.dtype == np.float32:
            maxval = 1.
        else:
            maxval = 255
        intImage = self.integralVecImage(features)
        i1 = np.arange(region_size, features.shape[0]+1, region_size).reshape(-1, 1)
        i2 = np.arange(region_size, features.shape[1]+1, region_size).reshape(1, -1)
        region_image = (intImage[i1, i2, :] - intImage[i1, i2-region_size,:] - intImage[i1-region_size, i2, :] + intImage[i1-region_size, i2-region_size, :])  / (region_area * maxval)
        return region_image

    def get_features(self, img, pos, sample_sz, scales):
        '''获取图像的cn特征'''
        feat = []
        if not isinstance(scales, list) and not isinstance(scales, np.ndarray):
            scales = [scales]
        for scale in scales:
            patch = self._sample_patch(img, pos, sample_sz*scale, sample_sz)
            # _sample_patch()返回裁剪过后的目标图像，尺寸为sample_sz大小
            h, w, c = patch.shape
            if c == 3:
                # RGB图
                RR = patch[:, :, 0].astype(np.int32)
                GG = patch[:, :, 1].astype(np.int32)
                BB = patch[:, :, 2].astype(np.int32)
                index = RR // self._den + (GG // self._den) * self._factor + (BB // self._den) * self._factor * self._factor
                features = self._table[index.flatten()].reshape((h, w, self._table.shape[1]))
                # 从读取的CN.pkl中拿特征
            else:
                features = self._table[patch.flatten()].reshape((h, w, self._table.shape[1]))
            if self._cell_size > 1:
                features = self.average_feature_region(features, self._cell_size)
            feat.append(features)
        feat = self._feature_normalization(np.stack(feat, axis=3))
        # 归一化特征
        return [feat]

