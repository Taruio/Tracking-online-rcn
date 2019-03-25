import numpy as np
import cupy as cp
import cv2
import scipy
import time
# from numpy.fft import fftshift
# from pyfftw.interfaces.numpy_fft import fftshift
import matplotlib.pyplot as plt
from .config import config
from .features import FHogFeature, TableFeature, mround, ResNet50Feature, VGG16Feature
from .fourier_tools import cfft2, interpolate_dft, shift_sample, full_fourier_coeff,\
        cubic_spline_fourier, compact_fourier_coeff, ifft2, fft2
from .optimize_score import optimize_score
from .sample_space_model import GMM
from .train import train_joint, train_filter
from .scale_filter import ScaleFilter


class ECOTracker:
    def __init__(self, is_color, session, network):
        self._is_color = is_color
        self._frame_num = 0
        self._frames_since_last_train = 0
        self.nets = network
        self.sess = session
        if config.use_gpu:
            cp.cuda.Device(config.gpu_id).use()

    def build_features(self):
        fhog_params = {'fname': 'fhog',
                       'num_orients': 9,
                       'cell_size': 4,
                       'compressed_dim': 10,
                       # 'nDim': 9 * 3 + 5 -1
                       }
        cnn_params = {'fname': "cnn-vgg16",
                      'compressed_dim': [16, 64],
                      'session': self.sess,
                      'network': self.nets
                      }
        return [fhog_params, cnn_params]

    def _cosine_window(self, size):
        """
            生成size大小的余弦分布（即中心强，四周弱的分布）
            get the cosine window
        """
        cos_window = np.hanning(int(size[0]+2))[:, np.newaxis].dot(np.hanning(int(size[1]+2))[np.newaxis, :])
        cos_window = cos_window[1:-1, 1:-1][:, :, np.newaxis, np.newaxis].astype(np.float32)
        if config.use_gpu:
            cos_window = cp.asarray(cos_window)
        return cos_window

    def _get_interp_fourier(self, sz):
        """
            compute the fourier series of the interpolation function.
        """
        f1 = np.arange(-(sz[0]-1) / 2, (sz[0]-1)/2+1, dtype=np.float32)[:, np.newaxis] / sz[0]
        interp1_fs = np.real(cubic_spline_fourier(f1, config.interp_bicubic_a) / sz[0])
        f2 = np.arange(-(sz[1]-1) / 2, (sz[1]-1)/2+1, dtype=np.float32)[np.newaxis, :] / sz[1]
        interp2_fs = np.real(cubic_spline_fourier(f2, config.interp_bicubic_a) / sz[1])
        if config.interp_centering:
            f1 = np.arange(-(sz[0]-1) / 2, (sz[0]-1)/2+1, dtype=np.float32)[:, np.newaxis]
            interp1_fs = interp1_fs * np.exp(-1j*np.pi / sz[0] * f1)
            f2 = np.arange(-(sz[1]-1) / 2, (sz[1]-1)/2+1, dtype=np.float32)[np.newaxis, :]
            interp2_fs = interp2_fs * np.exp(-1j*np.pi / sz[1] * f2)

        if config.interp_windowing:
            win1 = np.hanning(sz[0]+2)[:, np.newaxis]
            win2 = np.hanning(sz[1]+2)[np.newaxis, :]
            interp1_fs = interp1_fs * win1[1:-1]
            interp2_fs = interp2_fs * win2[1:-1]
        if not config.use_gpu:
            return (interp1_fs[:, :, np.newaxis, np.newaxis],
                    interp2_fs[:, :, np.newaxis, np.newaxis])
        else:
            return (cp.asarray(interp1_fs[:, :, np.newaxis, np.newaxis]),
                    cp.asarray(interp2_fs[:, :, np.newaxis, np.newaxis]))

    def _get_reg_filter(self, sz, target_sz, reg_window_edge):
        """
            compute the spatial regularization function and drive the
            corresponding filter operation used for optimization
        """
        if config.use_reg_window:
            # normalization factor
            reg_scale = 0.5 * target_sz

            # construct grid
            wrg = np.arange(-(sz[0]-1)/2, (sz[1]-1)/2+1, dtype=np.float32)
            wcg = np.arange(-(sz[0]-1)/2, (sz[1]-1)/2+1, dtype=np.float32)
            wrs, wcs = np.meshgrid(wrg, wcg)

            # construct the regularization window
            reg_window = (reg_window_edge - config.reg_window_min) * (np.abs(wrs/reg_scale[0])**config.reg_window_power + \
                            np.abs(wcs/reg_scale[1])**config.reg_window_power) + config.reg_window_min

            # compute the DFT and enforce sparsity
            reg_window_dft = fft2(reg_window) / np.prod(sz)
            reg_window_dft[np.abs(reg_window_dft) < config.reg_sparsity_threshold* np.max(np.abs(reg_window_dft.flatten()))] = 0

            # do the inverse transform, correct window minimum
            reg_window_sparse = np.real(ifft2(reg_window_dft))
            reg_window_dft[0, 0] = reg_window_dft[0, 0] - np.prod(sz) * np.min(reg_window_sparse.flatten()) + config.reg_window_min
            reg_window_dft = np.fft.fftshift(reg_window_dft).astype(np.complex64)

            # find the regularization filter by removing the zeros
            row_idx = np.logical_not(np.all(reg_window_dft==0, axis=1))
            col_idx = np.logical_not(np.all(reg_window_dft==0, axis=0))
            mask = np.outer(row_idx, col_idx)
            reg_filter = np.real(reg_window_dft[mask]).reshape(np.sum(row_idx), -1)
        else:
            # else use a scaled identity matrix
            reg_filter = config.reg_window_min
        if not config.use_gpu:
            return reg_filter.T
        else:
            return cp.asarray(reg_filter.T)

    def _init_proj_matrix(self, init_sample, compressed_dim, proj_method):
        """
            init the projection matrix
        """
        xp = cp.get_array_module(init_sample[0])
        x = [xp.reshape(x, (-1, x.shape[2])) for x in init_sample]
        x = [z - z.mean(0) for z in x]
        proj_matrix_ = []
        if config.proj_init_method == 'pca':
            for x_, compressed_dim_  in zip(x, compressed_dim):
                proj_matrix, _, _ = xp.linalg.svd(x_.T.dot(x_))
                proj_matrix = proj_matrix[:, :compressed_dim_]
                proj_matrix_.append(proj_matrix)
        elif config.proj_init_method == 'rand_uni':
            for x_, compressed_dim_ in zip(x, compressed_dim):
                proj_matrix = xp.random.uniform(size=(x_.shape[1], compressed_dim_))
                proj_matrix /= xp.sqrt(xp.sum(proj_matrix**2, axis=0, keepdims=True))
                proj_matrix_.append(proj_matrix)
        return proj_matrix_

    def _proj_sample(self, x, P):
        xp = cp.get_array_module(x[0])
        return [xp.matmul(P_.T, x_) for x_, P_ in zip(x, P)]



    def init(self, frame, bbox, total_frame=np.inf):
        """
            frame -- image
            bbox -- need xmin, ymin, width, height
        """
        self._pos = np.array([bbox[1]+(bbox[3]-1)/2., bbox[0]+(bbox[2]-1)/2.], dtype=np.float32)
        # _pos表示y中心、x中心
        self._target_sz = np.array([bbox[3], bbox[2]])
        # _target_sz表示目标框的h、w
        self._num_samples = min(config.num_samples, total_frame)
        # 存储的最大训练样本（默认为30）
        self.last_score = np.zeros((10,1))
        self.flag = 0
        self.plot_score = []
        self.count = 0

        xp = cp if config.use_gpu else np

        # calculate search area and initial scale factor
        search_area = np.prod(self._target_sz * config.search_area_scale)
        # 检查面积默认为目标大小的4倍，np.prob计算面积
        # max_image_sample_size为200*200
        # min_image_sample_size为150*150
        if search_area > config.max_image_sample_size:
            self._current_scale_factor = np.sqrt(search_area / config.max_image_sample_size)
        elif search_area < config.min_image_sample_size:
            self._current_scale_factor = np.sqrt(search_area / config.min_image_sample_size)
        else:
            self._current_scale_factor = 1.

        # target size at the initial scale
        self._base_target_sz = self._target_sz / self._current_scale_factor

        # target size, taking padding into account
        if config.search_area_shape == 'proportional':
            self._img_sample_sz = np.floor(self._base_target_sz * config.search_area_scale)
        elif config.search_area_shape == 'square':
            self._img_sample_sz = np.ones((2), dtype=np.float32) * np.sqrt(np.prod(self._base_target_sz * config.search_area_scale))
            # _img_sample_sz为2维
        else:
            raise("unimplemented")
        # 计算标准化图片的面积大小
        deep_features = self.build_features()
        features = [feature for feature in deep_features
                if ("use_for_color" in feature and feature["use_for_color"] == self._is_color) or
                    "use_for_color" not in feature]
        # 在输入彩色图像（_is_color == True）时，使用fhog和cn特征
        # 在输入灰度图像（_is_color == False）时，使用fhog和ic特征

        self._features = []
        cnn_feature_idx = -1
        for idx, feature in enumerate(features):
            if feature['fname'] == 'cn' or feature['fname'] == 'ic':
                self._features.append(TableFeature(**feature))
                # TableFeature读取了CNnorm.pkl或ic.pkl
            elif feature['fname'] == 'fhog':
                self._features.append(FHogFeature(**feature))
                # FHogFeature初始化了计算HOG的类

            elif feature['fname'].startswith('cnn'):
                cnn_feature_idx = idx
                netname = feature['fname'].split('-')[1]
                if netname == 'resnet50':
                    self._features.append(ResNet50Feature(**feature))
                elif netname == 'vgg16':
                    self._features.append(VGG16Feature(**feature))
                # 利用神经网络

            else:
                raise("unimplemented features")

        self._features = sorted(self._features, key=lambda x:x.min_cell_size)
        # 根据features利用的min_cell_size排序
        # CN和ic为4，Fhog为6

        # calculate image sample size
        if cnn_feature_idx >= 0:
            self._img_sample_sz = self._features[cnn_feature_idx].init_size(self._img_sample_sz)
            # 用cnn的情况。。。
        else:
            cell_size = [x.min_cell_size for x in self._features]
            # cell_size = [4,6]
            self._img_sample_sz = self._features[0].init_size(self._img_sample_sz, cell_size)
            # self._features[0].init_size计算图像的初始size

        for idx, feature in enumerate(self._features):
            if idx != cnn_feature_idx:
                feature.init_size(self._img_sample_sz)
                # feature.init_size使用cell_size = None的情况，仅保存self.sample_sz和self.data_sz

        if config.use_projection_matrix:
            sample_dim = [ x for feature in self._features for x in feature._compressed_dim ]
            # cn特征为3，ic特征为4，Fhog特征为10
            # sample_dim = [3,10]
        else:
            sample_dim = [ x for feature in self._features for x in feature.num_dim ]

        feature_dim = [ x for feature in self._features for x in feature.num_dim ]
        # cn的num_dim为10，Fhog的num_dim为31

        feature_sz = np.array([x for feature in self._features for x in feature.data_sz ], dtype=np.int32)
        # 两个特征的self.data_sz = [img_sample_sz // self._cell_size]

        # number of fourier coefficients to save for each filter layer, this will be an odd number
        filter_sz = feature_sz + (feature_sz + 1) % 2
        # filter_sz = [[41,41], [27,27]]
        # 每个滤波器要保存的傅里叶变换系数的个数（滤波器尺寸）

        # the size of the label function DFT. equal to the maximum filter size
        self._k1 = np.argmax(filter_sz, axis=0)[0]
        self._output_sz = filter_sz[self._k1]
        # 找到最大滤波器的尺寸作为输出尺寸

        self._num_feature_blocks = len(feature_dim)
        # 使用特征的个数，2

        # get the remaining block indices
        self._block_inds = list(range(self._num_feature_blocks))
        self._block_inds.remove(self._k1)
        # _block_inds保留小的滤波器尺寸的标号

        # how much each feature block has to be padded to the obtain output_sz
        self._pad_sz = [((self._output_sz - filter_sz_) / 2).astype(np.int32) for filter_sz_ in filter_sz]
        # 计算padding的数目

        # compute the fourier series indices and their transposes
        self._ky = [np.arange(-np.ceil(sz[0]-1)/2, np.floor((sz[0]-1)/2)+1, dtype=np.float32)
                        for sz in filter_sz]
        self._kx = [np.arange(-np.ceil(sz[1]-1)/2, 1, dtype=np.float32)
                        for sz in filter_sz]
        # 计算傅里叶变换用的y和x的指数？
        # ky对于每个特征有sz[0]个，kx对于每个特征有sz[0]/2个

        # construct the gaussian label function using poisson formula
        sig_y = np.sqrt(np.prod(np.floor(self._base_target_sz))) * config.output_sigma_factor * (self._output_sz / self._img_sample_sz)
        yf_y = [np.sqrt(2 * np.pi) * sig_y[0] / self._output_sz[0] * np.exp(-2 * (np.pi * sig_y[0] * ky_ / self._output_sz[0])**2)
                    for ky_ in self._ky]
        yf_x = [np.sqrt(2 * np.pi) * sig_y[1] / self._output_sz[1] * np.exp(-2 * (np.pi * sig_y[1] * kx_ / self._output_sz[1])**2)
                    for kx_ in self._kx]
        self._yf = [yf_y_.reshape(-1, 1) * yf_x_ for yf_y_, yf_x_ in zip(yf_y, yf_x)]
        # 构建高斯函数的参数。。。
        if config.use_gpu:
            self._yf = [cp.asarray(yf) for yf in self._yf]
            self._ky = [cp.asarray(ky) for ky in self._ky]
            self._kx = [cp.asarray(kx) for kx in self._kx]

        # construct cosine window
        self._cos_window = [self._cosine_window(feature_sz_) for feature_sz_ in feature_sz]
        # 生成期望的余弦分布图

        # compute fourier series of interpolation function
        self._interp1_fs = []
        self._interp2_fs = []
        for sz in filter_sz:
            interp1_fs, interp2_fs = self._get_interp_fourier(sz)
            self._interp1_fs.append(interp1_fs)
            self._interp2_fs.append(interp2_fs)
        # 计算傅里叶变换的插值函数

        # get the reg_window_edge parameter
        reg_window_edge = []
        for feature in self._features:
            if hasattr(feature, 'reg_window_edge'):
                reg_window_edge.append(feature.reg_window_edge)
            else:
                reg_window_edge += [config.reg_window_edge for _ in range(len(feature.num_dim))]
        # 10+31一共41个10e-3

        # construct spatial regularization filter
        self._reg_filter = [self._get_reg_filter(self._img_sample_sz, self._base_target_sz, reg_window_edge_)
                                for reg_window_edge_ in reg_window_edge]
        # 利用空间正则化函数构建滤波器

        # compute the energy of the filter (used for preconditioner)
        if not config.use_gpu:
            self._reg_energy = [np.real(np.vdot(reg_filter.flatten(), reg_filter.flatten()))
                            for reg_filter in self._reg_filter]
            # np.real(val)返回val的实部
            # np.vdot(a,b)计算a与b的点积
        else:
            self._reg_energy = [cp.real(cp.vdot(reg_filter.flatten(), reg_filter.flatten()))
                            for reg_filter in self._reg_filter]
        # 计算滤波器的能量

        if config.use_scale_filter:
            # 应用尺度自适应的滤波器
            self._scale_filter = ScaleFilter(self._target_sz)
            self._num_scales = self._scale_filter.num_scales
            # _num_scales = 17
            self._scale_step = self._scale_filter.scale_step
            # _scale_step = 1.02
            self._scale_factor = self._scale_filter.scale_factors
            # _scale_factor = np.array([1])
        else:
            # use the translation filter to estimate the scale
            self._num_scales = config.number_of_scales
            self._scale_step = config.scale_step
            scale_exp = np.arange(-np.floor((self._num_scales-1)/2), np.ceil((self._num_scales-1)/2)+1)
            self._scale_factor = self._scale_step**scale_exp

        if self._num_scales > 0:
            # force reasonable scale changes
            self._min_scale_factor = self._scale_step ** np.ceil(np.log(np.max(5 / self._img_sample_sz)) / np.log(self._scale_step))
            self._max_scale_factor = self._scale_step ** np.floor(np.log(np.min(frame.shape[:2] / self._base_target_sz)) / np.log(self._scale_step))
            # 转换尺寸比例。。。

        # set conjugate gradient options
        init_CG_opts = {'CG_use_FR': True,
                        'tol': 1e-6,
                        'CG_standard_alpha': True
                       }
        self._CG_opts = {'CG_use_FR': config.CG_use_FR,
                         'tol': 1e-6,
                         'CG_standard_alpha': config.CG_standard_alpha
                        }
        # config.CG_use_FR默认为False
        # config.CG_standard_alpha默认为True
        if config.CG_forgetting_rate == np.inf or config.learning_rate >= 1:
            # config.CG_forgetting_rate为50
            # config.learning_rate为0.025
            self._CG_opts['init_forget_factor'] = 0.
        else:
            self._CG_opts['init_forget_factor'] = (1 - config.learning_rate) ** config.CG_forgetting_rate
            # 0.2819881023409169

        # init ana allocate
        self._gmm = GMM(self._num_samples)
        #  初始化GMM
        self._samplesf = [[]] * self._num_feature_blocks
        # _num_feature_blocks为2， _samplesf = [[], []]

        for i in range(self._num_feature_blocks):
            if not config.use_gpu:
                self._samplesf[i] = np.zeros((int(filter_sz[i, 0]), int((filter_sz[i, 1]+1)/2),
                    sample_dim[i], config.num_samples), dtype=np.complex64)
                # cn特征的_samplesf大小为[41，21，3，30]
                # Fhog特征的_samplesf大小为[27，14，10，30]
            else:
                self._samplesf[i] = cp.zeros((int(filter_sz[i, 0]), int((filter_sz[i, 1]+1)/2),
                    sample_dim[i], config.num_samples), dtype=cp.complex64)

        # allocate
        self._num_training_samples = 0

        # extract sample and init projection matrix
        sample_pos = mround(self._pos)
        # sample_pos保存了整数的的目标中心y，x坐标
        sample_scale = self._current_scale_factor
        xl = [x for feature in self._features
                for x in feature.get_features(frame, sample_pos, self._img_sample_sz, self._current_scale_factor) ]
        # get_features
        # cn特征的get_features得到从读取的cn特征矩阵中得到的特征
        # fhog特征从_gradient.cpp中计算

        if config.use_gpu:
            xl = [cp.asarray(x) for x in xl]
        # 进行第一次滤波器的训练
        xlw = [x * y for x, y in zip(xl, self._cos_window)]                                                          # do windowing
        xlf = [cfft2(x) for x in xlw]                                                                                # fourier series
        xlf = interpolate_dft(xlf, self._interp1_fs, self._interp2_fs)                                               # interpolate features,
        xlf = compact_fourier_coeff(xlf)                                                                             # new sample to be added
        shift_sample_ = 2 * np.pi * (self._pos - sample_pos) / (sample_scale * self._img_sample_sz)
        xlf = shift_sample(xlf, shift_sample_, self._kx, self._ky)
        self._proj_matrix = self._init_proj_matrix(xl, sample_dim, config.proj_init_method)
        xlf_proj = self._proj_sample(xlf, self._proj_matrix)
        merged_sample, new_sample, merged_sample_id, new_sample_id = \
            self._gmm.update_sample_space_model(self._samplesf, xlf_proj, self._num_training_samples)
        self._num_training_samples += 1

        if config.update_projection_matrix:
            for i in range(self._num_feature_blocks):
                self._samplesf[i][:, :, :, new_sample_id:new_sample_id+1] = new_sample[i]

        # train_tracker
        self._sample_energy = [xp.real(x * xp.conj(x)) for x in xlf_proj]

        # init conjugate gradient param
        self._CG_state = None
        if config.update_projection_matrix:
            init_CG_opts['maxit'] = np.ceil(config.init_CG_iter / config.init_GN_iter)
            self._hf = [[[]] * self._num_feature_blocks for _ in range(2)]
            feature_dim_sum = float(np.sum(feature_dim))
            proj_energy = [2 * xp.sum(xp.abs(yf_.flatten())**2) / feature_dim_sum * xp.ones_like(P)
                    for P, yf_ in zip(self._proj_matrix, self._yf)]
        else:
            self._CG_opts['maxit'] = config.init_CG_iter
            self._hf = [[[]] * self._num_feature_blocks]

        # init the filter with zeros
        for i in range(self._num_feature_blocks):
            self._hf[0][i] = xp.zeros((int(filter_sz[i, 0]), int((filter_sz[i, 1]+1)/2),
                int(sample_dim[i]), 1), dtype=xp.complex64)

        if config.update_projection_matrix:
            # init Gauss-Newton optimization of the filter and projection matrix
            self._hf, self._proj_matrix = train_joint(
                                                  self._hf,
                                                  self._proj_matrix,
                                                  xlf,
                                                  self._yf,
                                                  self._reg_filter,
                                                  self._sample_energy,
                                                  self._reg_energy,
                                                  proj_energy,
                                                  init_CG_opts)
            # re-project and insert training sample
            xlf_proj = self._proj_sample(xlf, self._proj_matrix)
            # self._sample_energy = [np.real(x * np.conj(x)) for x in xlf_proj]
            for i in range(self._num_feature_blocks):
                self._samplesf[i][:, :, :, 0:1] = xlf_proj[i]

            # udpate the gram matrix since the sample has changed
            if config.distance_matrix_update_type == 'exact':
                # find the norm of the reprojected sample
                new_train_sample_norm = 0.
                for i in range(self._num_feature_blocks):
                    new_train_sample_norm += 2 * xp.real(xp.vdot(xlf_proj[i].flatten(), xlf_proj[i].flatten()))
                self._gmm._gram_matrix[0, 0] = new_train_sample_norm
        self._hf_full = full_fourier_coeff(self._hf)

        if config.use_scale_filter and self._num_scales > 0:
            self._scale_filter.update(frame, self._pos, self._base_target_sz, self._current_scale_factor)
        self._frame_num += 1

    def update(self, frame, train=True):
        # target localization step
        xp = cp if config.use_gpu else np
        pos = self._pos
        # 上次目标位置
        old_pos = np.zeros((2))
        for _ in range(config.refinement_iterations):
            # if np.any(old_pos != pos):
            if not np.allclose(old_pos, pos):
                old_pos = pos.copy()
                # extract features at multiple resolutions
                sample_pos = mround(pos)
                sample_scale = self._current_scale_factor * self._scale_factor
                xt = [x for feature in self._features
                        for x in feature.get_features(frame, sample_pos, self._img_sample_sz, sample_scale) ]  # get features
                if config.use_gpu:
                    xt = [cp.asarray(x) for x in xt]
                # xt是列表,其中三项,第一项为FHOG反馈特征,第二三项为vgg16反馈的第一层卷积特征f1和第四层卷积特征f2
                # [60,60,31,5],[60,60,64,5],[15,15,512,5]
                xt_proj = self._proj_sample(xt, self._proj_matrix)                                             # project sample
                # 函数将xt中的特征处理,降维表示
                # 原hog特征[60,60,31,5]降为[60,60,10,5]
                # 原深度特征f1 [60,60,64,5]降为[60,60,16,5]
                # 原深度特征f2 [15,15,512,5]降为[15,15,64,5]
                # 其中_proj_matrix为原滤波器维度Ｄ转换到新特征维度Ｃ的权重矩阵
                xt_proj = [feat_map_ * cos_window_
                        for feat_map_, cos_window_ in zip(xt_proj, self._cos_window)]                          # do windowing
                # 每一个特征与其对应的余弦窗函数相乘
                xtf_proj = [cfft2(x) for x in xt_proj]                                                         # compute the fourier series
                # 对特征做二维傅里叶变换，其中使用了fftshift将０转换到正中
                # 若输入矩阵的宽高为偶数，则增加一个维度变为奇数
                # [60,60,10,5]→[61,61,10,5],[60,60,16,5]→[61,61,16,5],[15,15,64,5]不变
                xtf_proj = interpolate_dft(xtf_proj, self._interp1_fs, self._interp2_fs)                       # interpolate features to continuous domain
                #　执行隐式插值，将样本转化到连续空间域上

                # compute convolution for each feature block in the fourier domain, then sum over blocks
                scores_fs_feat = [[]] * self._num_feature_blocks
                # 创建特征得分的存放列表，共三项，用来存放３种特征
                scores_fs_feat[self._k1] = xp.sum(self._hf_full[self._k1] * xtf_proj[self._k1], 2)
                # _hf_full为滤波器，将滤波器与特征相乘，获得响应图，将ＦＨＯＧ特征的十个维度得到的响应图叠加
                # 第一特征FHOG
                scores_fs = scores_fs_feat[self._k1]

                # scores_fs_sum shape: height x width x num_scale
                # 深度两个特征的分辨率不同
                for i in self._block_inds:
                    scores_fs_feat[i] = xp.sum(self._hf_full[i] * xtf_proj[i], 2)
                    scores_fs[self._pad_sz[i][0]:self._output_sz[0]-self._pad_sz[i][0],
                              self._pad_sz[i][1]:self._output_sz[0]-self._pad_sz[i][1]] += scores_fs_feat[i]
                # 对于深度特征中的f1，其大小与FHOG特征相同，直接叠加
                # 对于深度特征中的f2，其代表了低分辨率特征，将其叠加到得分的中间区域[23:38,23:38]处

                # optimize the continuous score function with newton's method.
                # 利用scores判断目标的下一位置
                trans_row, trans_col, scale_idx , self.last_score, self.flag= optimize_score(scores_fs, config.newton_iterations, self.last_score, self.flag)
                # self.plot_score.append(self.last_score[9][0])
                # if len(self.plot_score) >= 50:
                #     xx = np.arange(0,50,1)
                #     plt.plot(xx,self.plot_score,'r',)
                #     plt.savefig(r'/home/lhc/Desktop/pyECO-master/'+'300'+str(self.count)+r'.jpg')
                #     plt.close()
                #     self.count += 1
                #     self.plot_score = []
                # 绘图判断目标丢失时的分数变化情况

                # compute the translation vector in pixel-coordinates and round to the cloest integer pixel
                translation_vec = np.array([trans_row, trans_col]) * (self._img_sample_sz / self._output_sz) * \
                                    self._current_scale_factor * self._scale_factor[scale_idx]
                scale_change_factor = self._scale_factor[scale_idx]

                # udpate position
                pos = sample_pos + translation_vec
                # 更新目标的下一位置

                if config.clamp_position:
                    pos = np.maximum(np.array(0, 0), np.minimum(np.array(frame.shape[:2]), pos))

                # do scale tracking with scale filter
                if self._num_scales > 0 and config.use_scale_filter:
                    scale_change_factor = self._scale_filter.track(frame, pos, self._base_target_sz,
                           self._current_scale_factor)

                # udpate the scale
                self._current_scale_factor *= scale_change_factor

                # adjust to make sure we are not to large or to small
                if self._current_scale_factor < self._min_scale_factor:
                    self._current_scale_factor = self._min_scale_factor
                elif self._current_scale_factor > self._max_scale_factor:
                    self._current_scale_factor = self._max_scale_factor

        # model udpate step
        if config.learning_rate > 0:
            # use the sample that was used for detection
            sample_scale = sample_scale[scale_idx]
            xlf_proj = [xf[:, :(xf.shape[1]+1)//2, :, scale_idx:scale_idx+1] for xf in xtf_proj]

            # shift the sample so that the target is centered
            shift_sample_ = 2 * np.pi * (pos - sample_pos) / (sample_scale * self._img_sample_sz)
            xlf_proj = shift_sample(xlf_proj, shift_sample_, self._kx, self._ky)

        # update the samplesf to include the new sample. The distance matrix, kernel matrix and prior weight are also updated
        merged_sample, new_sample, merged_sample_id, new_sample_id = \
                self._gmm.update_sample_space_model(self._samplesf, xlf_proj, self._num_training_samples)

        if self._num_training_samples < self._num_samples:
            self._num_training_samples += 1

        if config.learning_rate > 0:
            for i in range(self._num_feature_blocks):
                if merged_sample_id >= 0:
                    self._samplesf[i][:, :, :, merged_sample_id:merged_sample_id+1] = merged_sample[i]
                if new_sample_id >= 0:
                    self._samplesf[i][:, :, :, new_sample_id:new_sample_id+1] = new_sample[i]

        # training filter
        if self._frame_num < config.skip_after_frame or \
                self._frames_since_last_train >= config.train_gap:
            # print("Train filter: ", self._frame_num)
            new_sample_energy = [xp.real(xlf * xp.conj(xlf)) for xlf in xlf_proj]
            self._CG_opts['maxit'] = config.CG_iter
            self._sample_energy = [(1 - config.learning_rate)*se + config.learning_rate*nse
                                for se, nse in zip(self._sample_energy, new_sample_energy)]

            # do conjugate gradient optimization of the filter
            self._hf, self._CG_state = train_filter(
                                                 self._hf,
                                                 self._samplesf,
                                                 self._yf,
                                                 self._reg_filter,
                                                 self._gmm.prior_weights,
                                                 self._sample_energy,
                                                 self._reg_energy,
                                                 self._CG_opts,
                                                 self._CG_state)
            # reconstruct the ful fourier series
            self._hf_full = full_fourier_coeff(self._hf)
            self._frames_since_last_train = 0
        else:
            self._frames_since_last_train += 1
        if config.use_scale_filter:
            self._scale_filter.update(frame, pos, self._base_target_sz, self._current_scale_factor)

        # udpate the target size
        self._target_sz = self._base_target_sz * self._current_scale_factor

        # save position and calculate fps
        bbox = (pos[1] - self._target_sz[1]/2, # xmin
                pos[0] - self._target_sz[0]/2, # ymin
                pos[1] + self._target_sz[1]/2, # xmax
                pos[0] + self._target_sz[0]/2) # ymax
        self._pos = pos
        self._frame_num += 1
        return bbox
