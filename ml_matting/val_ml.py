import os
import time
import sys
sys.path.append("/home/gsq")

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pymatting.util.util import load_image, save_image, stack_images
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from PIL import Image

import progbar
import metric
import logger

np.set_printoptions(suppress=True)


class TimeAverager(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self._cnt = 0
        self._total_time = 0
        self._total_samples = 0

    def record(self, usetime, num_samples=None):
        self._cnt += 1
        self._total_time += usetime
        if num_samples:
            self._total_samples += num_samples

    def get_average(self):
        if self._cnt == 0:
            return 0
        return self._total_time / float(self._cnt)

    def get_ips_average(self):
        if not self._total_samples or self._cnt == 0:
            return 0
        return float(self._total_samples) / self._total_time


def calculate_eta(remaining_step, speed):
    if remaining_step < 0:
        remaining_step = 0
    remaining_time = int(remaining_step * speed)
    result = "{:0>2}:{:0>2}:{:0>2}"
    arr = []
    for i in range(2, -1, -1):
        arr.append(int(remaining_time / 60**i))
        remaining_time %= 60**i
    return result.format(*arr)


def save_alpha_pred(alpha, path):
    """
    The value of alpha is range [0, 1], shape should be [h,w]
    """
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    alpha = (alpha).astype('uint8')
    cv2.imwrite(path, alpha)


# def reverse_transform(alpha, trans_info):
#     """recover pred to origin shape"""
#     for item in trans_info[::-1]:
#         if item[0][0] == 'resize':
#             h, w = int(item[1][0]), int(item[1][1])
#             alpha = cv2.resize(alpha, dsize=(w, h))
#         elif item[0][0] == 'padding':
#             h, w = int(item[1][0]), int(item[1][1])
#             alpha = alpha[0:h, 0:w]
#         else:
#             raise Exception("Unexpected info '{}' in im_info".format(item[0]))
#     return alpha


def evaluate_ml(model,
                eval_dataset,
                num_workers=0,
                print_detail=True,
                save_dir='output/results',
                save_results=True):

    # 使用 torch.utils.data.DataLoader 加载数据集
    loader = DataLoader(
        eval_dataset,
        batch_size=1,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=None,
        pin_memory=False,
        shuffle=False
    )

    total_iters = len(loader)
    mse_metric = metric.MSE()
    sad_metric = metric.SAD()
    grad_metric = metric.Grad()
    conn_metric = metric.Conn()

    if print_detail:
        logger.info("Start evaluating (total_samples: {}, total_iters: {})...".
                    format(len(eval_dataset), total_iters))
    progbar_val = progbar.Progbar(target=total_iters, verbose=1)
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()

    img_name = ''
    i = 0
    ignore_cnt = 0
    for iter, data in enumerate(loader):
        reader_cost_averager.record(time.time() - batch_start)

        image_rgb_chw = data['img'].numpy()[0]
        image_rgb_hwc = np.transpose(image_rgb_chw, (1, 2, 0))
        trimap = data['trimap'].numpy().squeeze()
        image = image_rgb_hwc * 0.5 + 0.5  # reverse normalize (x/255 - mean) / std

        is_fg = trimap >= 0.9
        is_bg = trimap <= 0.1

        if is_fg.sum() == 0 or is_bg.sum() == 0:
            ignore_cnt += 1
            logger.info(str(iter))
            continue
        alpha_pred = model(image, trimap)

        # alpha_pred = reverse_transform(alpha_pred, data['trans_info'])

        alpha_gt = data['alpha'].numpy().squeeze() * 255
        trimap = data['ori_trimap'].numpy().squeeze()

        alpha_pred = np.round(alpha_pred * 255)
        trimap = trimap * 255
        mse = mse_metric.update(alpha_pred, alpha_gt, trimap)
        sad = sad_metric.update(alpha_pred, alpha_gt, trimap)
        grad = grad_metric.update(alpha_pred, alpha_gt, trimap)
        conn = conn_metric.update(alpha_pred, alpha_gt, trimap)
        if sad > 1000:
            print(data['img_name'][0])

        if save_results:
            alpha_pred_one = alpha_pred
            alpha_pred_one[trimap == 255] = 255
            alpha_pred_one[trimap == 0] = 0

            save_name = data['img_name'][0]
            name, ext = os.path.splitext(save_name)
            if save_name == img_name:
                save_name = name + '_' + str(i) + ext
                i += 1
            else:
                img_name = save_name
                save_name = name + '_' + str(0) + ext
                i = 1
            save_alpha_pred(alpha_pred_one, os.path.join(save_dir, save_name))

        batch_cost_averager.record(
            time.time() - batch_start, num_samples=len(alpha_gt))
        batch_cost = batch_cost_averager.get_average()
        reader_cost = reader_cost_averager.get_average()

        if print_detail:
            progbar_val.update(iter + 1,
                               [('SAD', sad), ('MSE', mse), ('Grad', grad),
                                ('Conn', conn), ('batch_cost', batch_cost),
                                ('reader cost', reader_cost)])

        reader_cost_averager.reset()
        batch_cost_averager.reset()
        batch_start = time.time()

    mse = mse_metric.evaluate()
    sad = sad_metric.evaluate()
    grad = grad_metric.evaluate()
    conn = conn_metric.evaluate()

    logger.info('[EVAL] SAD: {:.4f}, MSE: {:.4f}, Grad: {:.4f}, Conn: {:.4f}'.
                format(sad, mse, grad, conn))
    logger.info('{}'.format(ignore_cnt))

    return sad, mse, grad, conn


class MLDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.fgr_path = os.path.join(root, 'fgr')
        self.pha_path = os.path.join(root, 'pha')
        self.trimap_path = os.path.join(root, 'trimap')
        self.img_name = list(map(lambda x: x.split('.jpg')[0], os.listdir(self.fgr_path)))
        self.fgr = list(map(lambda x: os.path.join(self.fgr_path, x + '.jpg'), self.img_name))
        self.pha = list(map(lambda x: os.path.join(self.pha_path, x + '.png'), self.img_name))
        self.trimap = list(map(lambda x: os.path.join(self.trimap_path, x + '.png'), self.img_name))


    def __len__(self):
        return len(self.fgr)

    def __getitem__(self, idx):
        t = transforms.ToTensor()

        fgr = Image.open(self.fgr[idx])
        # pha = Image.open(self.pha[idx])
        pha = Image.open(self.pha[idx]).convert('L')
        trimap = Image.open(self.trimap[idx]).convert('L')
        # trimap = Image.open(self.trimap[idx])

        fgr = t(fgr)
        pha = t(pha)
        trimap = t(trimap)

        return {
            'img': fgr,
            'alpha': pha,
            'trimap': trimap,
            'ori_trimap': trimap,
            'img_name': os.listdir(self.fgr_path)[idx],
            'trans_info': 'padding'
        }


if __name__ == '__main__':
    """
    CloseFormMatting am2k:
        SAD: 13.8519, MSE: 0.0203, Grad: 11.3624, Conn: 12.4828
    KNNMatting am2k:
        SAD: 18.1329, MSE: 0.0310, Grad: 15.6497, Conn: 16.9102
    LearningBasedMatting am2k:
        SAD: 13.9161, MSE: 0.0206, Grad: 12.4212, Conn: 12.7240
    FastMatting am2k:
        SAD: 17.8814, MSE: 0.0273, Grad: 14.3139, Conn: 16.4083
    RandomWalksMatting am2k:
        SAD: 29.3014, MSE: 0.0574, Grad: 24.5867, Conn: 28.8489

    LearningBasedMatting p3m10k:
    

    KNNMatting p3m10k:
        SAD: 13.8264, MSE: 0.0280, Grad: 15.1780, Conn: 12.4538
    CloseFormMatting p3m10k:
        SAD: 9.4703, MSE: 0.0171, Grad: 11.3417, Conn: 8.3186
    FastMatting p3m10k
        SAD: 12.5025, MSE: 0.0235, Grad: 14.4931, Conn: 11.1326
    """
    from method import CloseFormMatting, KNNMatting, LearningBasedMatting, FastMatting, RandomWalksMatting
    model = LearningBasedMatting()
    root = "/home/gsq/test/P3M-10k/val/P3M-500-NP"
    dataset = MLDataset(root)
    evaluate_ml(
        model,
        dataset,
        num_workers=16,
        save_dir=f'./output/{model.__class__.__name__}_p3m10k',
        save_results=True,
    )