import numpy as np
import os
import math
import cv2
from sklearn import mixture
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
import matplotlib.pyplot as plt

def readFloat(name):
    f = open(name, 'rb')

    if(f.readline().decode("utf-8"))  != 'float\n':
        raise Exception('float file %s did not contain <float> keyword' % name)

    dim = int(f.readline())
    dims = []
    count = 1
    for i in range(0, dim):
        d = int(f.readline())
        dims.append(d)
        count *= d

    dims = list(reversed(dims))
    data = np.fromfile(f, np.float32, count).reshape(dims)
    if dim == 2:
        data = np.transpose(data, (0, 1))
    elif dim == 3:
        data = np.transpose(data, (1, 2, 0))
    elif dim == 4:
        data = np.transpose(data, (2, 3, 1, 0))
    else:
        raise Exception('bad float file dimension: %d' % dim)

    return data

def decode_img(file_path, width=None, height=None):
    img = cv2.imread(file_path)
    img = img / 255.0
    img = np.subtract(img, 0.4)
    if width is not None and height is not None:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
    img = np.expand_dims(img, 0)
    img = np.transpose(img, (0, 3, 1, 2))
    return img

def decode_semantic(file_path, width=None, height=None):
    img = cv2.imread(file_path)
    img = img / 255.0
    if width is not None and height is not None:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
    img = np.expand_dims(img, 0)
    img = np.transpose(img, (0, 3, 1, 2))
    return img

def decode_obj(file_path, id, coeff_x, coeff_y):
    object = np.expand_dims(np.expand_dims(np.expand_dims(readFloat(file_path)[id], 0), 0), 3).astype(np.float32)
    x_tl = object[:, :, 0:1, :] / coeff_x
    y_tl = object[:, :, 1:2, :] / coeff_y
    width = object[:, :, 2:3, :] / coeff_x
    height = object[:, :, 3:4, :] / coeff_y
    object = np.concatenate((x_tl, y_tl, width, height, object[:, :, 4:6, :]), axis=2)
    return object

def decode_obj_gt(file_path, id):
    object = np.expand_dims(np.expand_dims(np.expand_dims(readFloat(file_path)[id], 0), 0), 3).astype(np.float32)
    x_tl = object[:, :, 0:1, :]
    y_tl = object[:, :, 1:2, :]
    width = object[:, :, 2:3, :]
    height = object[:, :, 3:4, :]
    x_center = x_tl + width / 2
    y_center = y_tl + height / 2
    object = np.concatenate((x_center, y_center, width, height), axis=2)
    object = np.transpose(object, (0, 2, 1, 3))
    return object

def decode_ego(file_path):
    ego = np.expand_dims(np.expand_dims(readFloat(file_path), 0), 0).astype(np.float32)
    return ego


def resample_hyps(hyps, coeff_x, coeff_y):
    resampled_hyps = []
    for h in hyps:
        x_center = h[:, 0:1, :, :] / coeff_x
        y_center = h[:, 1:2, :, :] / coeff_y
        width = h[:, 2:3, :, :] / coeff_x
        height = h[:, 3:4, :, :] / coeff_y
        resampled_hyps.append(np.concatenate((x_center, y_center, width, height), axis=1))
    return resampled_hyps

def draw_hyps(img_path, hyps, random_color=False):
    img = cv2.imread(img_path)
    for h in hyps:
        x1 = int(h[0, 0, 0, 0] - h[0, 2, 0, 0] / 2)
        y1 = int(h[0, 1, 0, 0] - h[0, 3, 0, 0] / 2)
        x2 = int(h[0, 0, 0, 0] + h[0, 2, 0, 0] / 2)
        y2 = int(h[0, 1, 0, 0] + h[0, 3, 0, 0] / 2)
        if random_color:
            color = list(np.random.random(size=3) * 256)
        else:
            color = (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img

def draw_heatmap(img_path, means, sigmas, weights, width, height, output_path, gt=None):
    def transparent_cmap(cmap, N=255):
        "Copy colormap and set alpha values"
        mycmap = cmap
        mycmap._init()
        mycmap._lut[:, -1] = np.clip(np.linspace(0, 1.0, N + 4), 0, 1.0)
        return mycmap

    # map from tl_x, tl_y, br_x, br_y to c_x, c_y, w, h
    mapped_means = []
    mapped_sigmas = []
    for i in range(len(means)):
        center_mean = (means[i][:, 0:2, :, :] + means[i][:, 2:4, :, :]) / 2
        scale_mean = means[i][:, 2:4, :, :] - means[i][:, 0:2, :, :]
        mapped_means.append(np.concatenate((center_mean, scale_mean), axis=1))
        center_sigma = (sigmas[i][:, 0:2, :, :] + sigmas[i][:, 2:4, :, :]) / 2
        scale_sigma = sigmas[i][:, 2:4, :, :] - sigmas[i][:, 0:2, :, :]
        mapped_sigmas.append(np.concatenate((center_sigma, scale_sigma), axis=1))


    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    if gt is not None:
        x1 = int(gt[0, 0, 0, 0] - gt[0, 2, 0, 0] / 2)
        y1 = int(gt[0, 1, 0, 0] - gt[0, 3, 0, 0] / 2)
        x2 = int(gt[0, 0, 0, 0] + gt[0, 2, 0, 0] / 2)
        y2 = int(gt[0, 1, 0, 0] + gt[0, 3, 0, 0] / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    # draw means as bounding boxes
    for h in mapped_means:
        x1 = int(h[0, 0, 0, 0] - h[0, 2, 0, 0] / 2)
        y1 = int(h[0, 1, 0, 0] - h[0, 3, 0, 0] / 2)
        x2 = int(h[0, 0, 0, 0] + h[0, 2, 0, 0] / 2)
        y2 = int(h[0, 1, 0, 0] + h[0, 3, 0, 0] / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # construct the GMM
    c_means = np.stack([mapped_means[i][0,0:2,0,0] for i in range(len(mapped_means))], axis=0)  # (4,2)
    c_sigmas = np.stack([mapped_sigmas[i][0,0:2,0,0] for i in range(len(mapped_sigmas))], axis=0)  # (4,2)
    c_weights = np.stack(weights, axis=0)  # (4,1)
    clfs = []
    for i in range(len(mapped_means)):
        clf = mixture.GaussianMixture(n_components=1, covariance_type='diag')
        var = c_sigmas[i:i + 1, :] * c_sigmas[i:i + 1, :]
        precisions_cholesky = _compute_precision_cholesky(var, 'diag')
        clf.weights_ = c_weights[i]
        clf.means_ = c_means[i:i + 1, :]
        clf.precisions_cholesky_ = precisions_cholesky
        clf.covariances_ = var
        clfs.append(clf)

    all_z = []
    for i in range(len(clfs)):
        Z = np.exp(clfs[i].score_samples(XX))
        Z = Z.reshape(X.shape)
        all_z.append(Z)
    Z_stacked = np.stack(all_z, axis=0)
    Z = np.max(Z_stacked, axis=0)
    vmax = np.max(Z)
    vmin = np.min(Z)
    plt.imshow(img)
    plt.contourf(X, Y, Z, cmap=transparent_cmap(plt.cm.jet), vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.clf()

def compute_nll(pred_means, pred_sigmas, pred_weights, gt):
    num_hyps = len(pred_means)
    # Transform gt and pred_hyps from c_x,c_y,w,h to tl_x,tl_y,br_x,br_y
    gt_transformed = np.concatenate((gt[:, 0:2, :, :] - gt[:, 2:4, :, :] / 2, gt[:, 0:2, :, :] + gt[:, 2:4, :, :] / 2), axis=1)

    sum_likelihood_tl_x = None
    sum_likelihood_tl_y = None
    sum_likelihood_br_x = None
    sum_likelihood_br_y = None
    eps = 1e-5 / 2.0
    for i in range(num_hyps):
        diff = np.subtract(gt_transformed, pred_means[i])  # (batch,4,1,1)
        diff2 = np.square(diff)
        sigma_sq_tl_x = np.square(pred_sigmas[i][:,0:1,:,:])
        sigma_sq_tl_y = np.square(pred_sigmas[i][:,1:2,:,:])
        sigma_sq_br_x = np.square(pred_sigmas[i][:,2:3,:,:])
        sigma_sq_br_y = np.square(pred_sigmas[i][:,3:4,:,:])
        sigma_sq_inv_tl_x = np.power(2 * sigma_sq_tl_x + eps, -1)
        sigma_sq_inv_tl_y = np.power(2 * sigma_sq_tl_y + eps, -1)
        sigma_sq_inv_br_x = np.power(2 * sigma_sq_br_x + eps, -1)
        sigma_sq_inv_br_y = np.power(2 * sigma_sq_br_y + eps, -1)
        c_tl_x = np.multiply(diff2[:,0:1,:,:], sigma_sq_inv_tl_x)
        c_tl_y = np.multiply(diff2[:,1:2,:,:], sigma_sq_inv_tl_y)
        c_br_x = np.multiply(diff2[:,2:3,:,:], sigma_sq_inv_br_x)
        c_br_y = np.multiply(diff2[:,3:4,:,:], sigma_sq_inv_br_y)
        c_exp_tl_x = np.exp(-1 * c_tl_x)
        c_exp_tl_y = np.exp(-1 * c_tl_y)
        c_exp_br_x = np.exp(-1 * c_br_x)
        c_exp_br_y = np.exp(-1 * c_br_y)
        sigma_inv_tl_x = np.power(np.sqrt(2 * 3.14 * sigma_sq_tl_x) + eps, -1)
        sigma_inv_tl_y = np.power(np.sqrt(2 * 3.14 * sigma_sq_tl_y) + eps, -1)
        sigma_inv_br_x = np.power(np.sqrt(2 * 3.14 * sigma_sq_br_x) + eps, -1)
        sigma_inv_br_y = np.power(np.sqrt(2 * 3.14 * sigma_sq_br_y) + eps, -1)
        likelihood_tl_x = np.multiply(c_exp_tl_x, sigma_inv_tl_x)
        likelihood_tl_y = np.multiply(c_exp_tl_y, sigma_inv_tl_y)
        likelihood_br_x = np.multiply(c_exp_br_x, sigma_inv_br_x)
        likelihood_br_y = np.multiply(c_exp_br_y, sigma_inv_br_y)
        likelihood_tl_x_weighted = np.multiply(likelihood_tl_x, pred_weights[i])
        likelihood_tl_y_weighted = np.multiply(likelihood_tl_y, pred_weights[i])
        likelihood_br_x_weighted = np.multiply(likelihood_br_x, pred_weights[i])
        likelihood_br_y_weighted = np.multiply(likelihood_br_y, pred_weights[i])
        if i == 0:
            sum_likelihood_tl_x = likelihood_tl_x_weighted
            sum_likelihood_tl_y = likelihood_tl_y_weighted
            sum_likelihood_br_x = likelihood_br_x_weighted
            sum_likelihood_br_y = likelihood_br_y_weighted
        else:
            sum_likelihood_tl_x = sum_likelihood_tl_x + likelihood_tl_x_weighted
            sum_likelihood_tl_y = sum_likelihood_tl_y + likelihood_tl_y_weighted
            sum_likelihood_br_x = sum_likelihood_br_x + likelihood_br_x_weighted
            sum_likelihood_br_y = sum_likelihood_br_y + likelihood_br_y_weighted

    sum_likelihood_tl_x = sum_likelihood_tl_x + eps
    sum_likelihood_tl_y = sum_likelihood_tl_y + eps
    sum_likelihood_br_x = sum_likelihood_br_x + eps
    sum_likelihood_br_y = sum_likelihood_br_y + eps
    nll_tl_x = -1 * np.log(sum_likelihood_tl_x)
    nll_tl_y = -1 * np.log(sum_likelihood_tl_y)
    nll_br_x = -1 * np.log(sum_likelihood_br_x)
    nll_br_y = -1 * np.log(sum_likelihood_br_y)
    nll = nll_tl_x + nll_tl_y + nll_br_x + nll_br_y #(1,1,1,1)

    return nll[0,0,0,0]

def get_best_hyp(hyps, gt):
    num_hyps = len(hyps)
    gts = np.stack([gt for i in range(0, num_hyps)], axis=1)  # n,num,c,1,1
    hyps = np.stack(hyps, axis=1)  # n,num,c,1,1

    def spatial_error(hyps, gts):
        diff = np.square(hyps - gts) # n,num,c,1,1
        channels_sum = np.sum(diff, axis=2) # n,num,1,1
        spatial_epes = np.sqrt(channels_sum) # n,num,1,1
        return np.expand_dims(spatial_epes, axis=2) # n,num,1,1,1

    def get_best(hypotheses, errors, num_hyps):
        indices = np.argmin(errors, axis=1) # n,1,1,1
        shape = indices.shape
        # compute one-hot encoding
        encoding = np.zeros((shape[0],num_hyps,shape[1],shape[2],shape[3]))
        encoding[np.arange(shape[0]),indices,np.arange(shape[1]),np.arange(shape[2]),np.arange(shape[3])] = 1 # n,num,1,1,1

        hyps_channels = hypotheses.shape[2]
        encoding = np.concatenate([encoding for i in range(hyps_channels)], axis=2) # n,num,c,1,1
        reduced = hypotheses * encoding # n,num,c,1,1
        reduced = np.sum(reduced, axis=1) # n,c,1,1
        return reduced

    errors = spatial_error(hyps, gts) # n,num,1,1,1
    best = get_best(hyps, errors, num_hyps) #n,c,1,1
    return best


def get_FDE(hyp, gt):
    diff = np.square(hyp[:, 0:2, :, :] - gt[:, 0:2, :, :])
    channels_sum = np.sum(diff, axis=1)
    spatial_epe = np.sqrt(channels_sum)
    fde = np.mean(spatial_epe)
    return fde

def compute_oracle_FDE(hyps, gt):
    best_hyp = get_best_hyp(hyps, gt)
    return get_FDE(best_hyp, gt)


def compute_oracle_IOU(hyps, gt):
    # convert from c_x,c_y,w,h to tl_x,tl_y,br_x,br_y
    gt_box_reformat = np.concatenate([gt[:, 0:2, 0, 0] - gt[:, 2:4, 0, 0] / 2, gt[:, 0:2, 0, 0] + gt[:, 2:4, 0, 0] / 2],
                                     axis=1)  # 1,4
    hyps_reformat = [
        np.concatenate([h[:, 0:2, 0, 0] - h[:, 2:4, 0, 0] / 2, h[:, 0:2, 0, 0] + h[:, 2:4, 0, 0] / 2], axis=1) for h in
        hyps]  # list of 1,4

    hyps_stacked = np.concatenate(hyps_reformat, axis=0)  # num,4
    gt_box_tiled = np.tile(gt_box_reformat, [len(hyps), 1])  # num,4

    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = np.split(gt_box_tiled, 4, axis=1)  # list of num,1
    b2_y1, b2_x1, b2_y2, b2_x2 = np.split(hyps_stacked, 4, axis=1)
    y1 = np.maximum(b1_y1, b2_y1)  # num,1
    x1 = np.maximum(b1_x1, b2_x1)
    y2 = np.minimum(b1_y2, b2_y2)
    x2 = np.minimum(b1_x2, b2_x2)
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)  # num,1

    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)  # num,1
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection  # num,1

    # 4. Compute IoU
    iou = intersection / union  # num,1
    max_overlap = np.max(iou)  # 1
    return max_overlap

