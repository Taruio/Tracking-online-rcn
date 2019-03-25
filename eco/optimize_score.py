from .fourier_tools import sample_fs
import numpy as np
import cupy as cp
from sklearn import linear_model
"""
    code no problem
"""

def optimize_score(scores_fs, iterations, last_sc, flag):
    """
        Maximizes the continuous convolution response (classification scores)
    """
    xp = cp.get_array_module(scores_fs)
    if len(scores_fs.shape) == 2:
        scores_fs = scores_fs[:, :, xp.newaxis]
    output_sz = scores_fs.shape[:2]
    # scores_fs为前面用特征与滤波器计算得到的响应得分，在跟踪阶段[61,61,5]对应了５个尺度的响应得分

    # do the grid search step by finding the maximum in the sampled response for each scale
    sampled_scores = sample_fs(scores_fs)
    # sample_scores通过傅里叶反变换得到实数分数,(61,61,5)
    init_max_score = xp.max(sampled_scores, axis=(0, 1))
    # 获得score中每个尺度滤波器下的响应峰值

    max_idx = xp.reshape(sampled_scores, (-1, sampled_scores.shape[2])).argmax(axis=0)
    # 找到每个尺度滤波器出现响应峰值的位置
    max_pos = xp.column_stack(xp.unravel_index(max_idx, sampled_scores[:,:,0].shape))
    # 将位置换算成得分矩阵中的坐标
    row = max_pos[:, 0:1]
    col = max_pos[:, 1:2]
    # 最大响应分数出现的行和列(在５个尺度下)

    # shift and rescale the coordinate system to [-pi, -pi]
    trans_row = (row + np.floor((output_sz[0] - 1)/2)) % output_sz[0] - np.floor((output_sz[1]-1)/2)
    trans_col = (col + np.floor((output_sz[1] - 1)/2)) % output_sz[1] - np.floor((output_sz[1]-1)/2)
    init_pos_y = 2 * np.pi * trans_row / output_sz[0]
    init_pos_x = 2 * np.pi * trans_col / output_sz[1]
    # 将上面的最大响应分数出现的行和列的位置映射到[-pi, -pi]上

    max_pos_y = init_pos_y
    max_pos_x = init_pos_x

    # construct grid
    ky = xp.arange(- np.ceil((output_sz[0] - 1)/2), np.floor(output_sz[0]-1)/2 + 1).reshape(1, -1)
    kx = xp.arange(- np.ceil((output_sz[1] - 1)/2), np.floor(output_sz[1]-1)/2 + 1).reshape(-1, 1)

    exp_iky = xp.exp(1j * max_pos_y * ky)[:, xp.newaxis, :].astype(xp.complex64)
    exp_ikx = xp.exp(1j * kx * max_pos_x.T).transpose()[:, :, xp.newaxis].astype(xp.complex64)

    ky2 = ky * ky
    kx2 = kx * kx

    max_pos_y = max_pos_y[:, :, xp.newaxis]
    max_pos_x = max_pos_x[:, :, xp.newaxis]

    init_pos_y = init_pos_y[:, :, xp.newaxis]
    init_pos_x = init_pos_x[:, :, xp.newaxis]

    scores_fs = scores_fs.transpose(2, 0, 1)
    # scores_fs大小[5,61,61],序数
    # 5为scale尺寸缩放?
    for _ in range(iterations):
        # compute gradient
        ky_exp_ky = ky * exp_iky
        kx_exp_kx = kx * exp_ikx
        y_resp = xp.matmul(exp_iky, scores_fs)
        resp_x = xp.matmul(scores_fs, exp_ikx)
        grad_y = -xp.imag(xp.matmul(ky_exp_ky, resp_x))
        grad_x = -xp.imag(xp.matmul(y_resp, kx_exp_kx))

        # compute hessian
        ival = 1j * xp.matmul(exp_iky, resp_x)
        H_yy = xp.real(-xp.matmul(ky2 * exp_iky, resp_x) + ival)
        H_xx = xp.real(-xp.matmul(y_resp, kx2 * exp_ikx) + ival)
        H_xy = xp.real(-xp.matmul(ky_exp_ky, xp.matmul(scores_fs, kx_exp_kx)))
        det_H = H_yy * H_xx - H_xy * H_xy

        # compute new position using newtons method
        max_pos_y = max_pos_y - (H_xx * grad_y - H_xy * grad_x) / det_H
        max_pos_x = max_pos_x - (H_yy * grad_x - H_xy * grad_y) / det_H

        # evaluate maximum
        exp_iky = xp.exp(1j * ky * max_pos_y).astype(xp.complex64)
        exp_ikx = xp.exp(1j * kx * max_pos_x).astype(xp.complex64)

    max_score = xp.real(xp.matmul(xp.matmul(exp_iky, scores_fs), exp_ikx)).flatten()
    # 经过上面的ｆｏｒ循环迭代后，max_pos_y和max_pos_x代表了最终迭代出的目标相对位置偏差
    # check for scales that have not increased in score
    idx = max_score < init_max_score
    max_score[idx] = init_max_score[idx]
    max_pos_y[idx] = init_pos_y[idx]
    max_pos_x[idx] = init_pos_x[idx]
    # idx这些行没什么卵用


    real_max_score = xp.max(max_score)
    scale_idx = xp.argmax(max_score)
    max_scale_response = max_score[scale_idx]
    disp_row = ((max_pos_y[scale_idx][0][0] + np.pi) % (2 * np.pi) - np.pi) / (2 * np.pi) * output_sz[0]
    disp_col = ((max_pos_x[scale_idx][0][0] + np.pi) % (2 * np.pi) - np.pi) / (2 * np.pi) * output_sz[1]

    real_sample_score = sampled_scores[:,:,scale_idx]
    result = side_mean(output_sz,max_pos,real_sample_score,real_max_score,scale_idx)
    print(result)

    # 利用flag判断分数小于之前分数的次数(判断是否跟丢)
    # if last_sc >= real_max_score:
    #     flag += 1
    # else:
    #     flag = flag - 1 if flag >= 1 else 0
    # last_sc = real_max_score
    # print(flag)
    # print(last_sc)

    # 利用相邻10个得分值的拟合直线斜率判断,斜率小于零时,分数有下降趋势,认为跟丢
    # 存在问题:1.在跟踪初期分数会有一个明显的峰值,随后下降,会被误判为丢失.2.跟踪过程中波动会导致误判为丢失
    # 改进措施:需要结合斜率和当前分数阈值判断,需要研究初始几帧出现峰值的原因
    # last_sc[0:9] = last_sc[1:10]
    # last_sc[9][0] = real_max_score
    # regr = linear_model.LinearRegression()
    # regr.fit(last_sc,np.arange(0,10,1).reshape((-1,1)))
    # coef = regr.coef_
    # if (coef < 0):
    #     flag = 1
    # else:
    #     flag = 0

    # 利用(最大分数-均值)/方差进行计算,无效
    # final_scores = sampled_scores[:,:,scale_idx].flatten()
    # scores_mean = final_scores.mean()
    # scores_var = final_scores.var()
    # print((real_max_score-scores_mean)/scores_var)


    if xp is np:
        return disp_row, disp_col, scale_idx, last_sc, flag
    else:
        return disp_row.get(), disp_col.get(), scale_idx.get(), last_sc, flag

def side_mean(size,max_pos,score,real_max_score,scale_idx):
    xp = cp.get_array_module(score)
    up_lim = 5
    down_lim = size[0] - 5
    left_lim = 5
    right_lim = size[1] - 5
    max_position = max_pos[scale_idx]
    max_row = max_position[0]
    max_col = max_position[1]
    if up_lim <= max_row <= down_lim and left_lim <= max_col <= right_lim:
        pos_up = max_row - 5
        pos_down = max_row + 5
        pos_left = max_col - 5
        pos_right = max_col + 5
        final_score = xp.hstack((score[0:pos_up, :].flatten(),
                                score[pos_down + 1:size[0], :].flatten(),
                                score[pos_up:pos_down + 1, 0:pos_left].flatten(),
                                score[pos_up:pos_down + 1, pos_right + 1:size[1]].flatten()))
        mean_final = final_score.mean()
        std_final = final_score.std()
        result = (real_max_score - mean_final) / std_final
    else:
        if max_row < up_lim:
            temp_score = xp.zeros(size)
            temp_score[0:5,:] = score[size[0]-5:size[0],:]
            temp_score[5:size[0],:] = score[0:size[0]-5,:]
            score = temp_score
            max_row += 5
        if max_row > down_lim:
            temp_score = xp.zeros(size)
            temp_score[size[0]-5:size[0],:] = score[0:5,:]
            temp_score[0:size[0]-5,:] = score[5:size[0],:]
            score = temp_score
            max_row -= 5
        if max_col < left_lim:
            temp_score = xp.zeros(size)
            temp_score[:,0:5] = score[:,size[1]-5:size[1]]
            temp_score[:,5:size[1]] = score[:,0:size[1]-5]
            score = temp_score
            max_col += 5
        if max_col > right_lim:
            temp_score = xp.zeros(size)
            temp_score[:,size[1]-5:size[1]] = score[:,0:5]
            temp_score[:,0:size[1]-5] = score[:,5:size[1]]
            score = temp_score
            max_col -= 5
        pos_up = max_row - 5
        pos_down = max_row + 5
        pos_left = max_col - 5
        pos_right = max_col + 5
        final_score = xp.hstack((score[0:pos_up, :].flatten(),
                                score[pos_down + 1:size[0], :].flatten(),
                                score[pos_up:pos_down + 1, 0:pos_left].flatten(),
                                score[pos_up:pos_down + 1, pos_right + 1:size[1]].flatten()))
        mean_final = final_score.mean()
        std_final = final_score.std()
        result = (real_max_score - mean_final) / std_final
    return result