import copy
import torch
import numpy as np

def peak_detection(ori_signal, count, threshold = 0.3):

    signal = copy.deepcopy(ori_signal)
    idx = 0
    peak_list = []
    freeze_value = -1e9
    while True:
        peak = np.max(signal)
        peak_index = np.argmax(signal)
        if peak == freeze_value:
            break
        windowsize = 5 #peak is 5, [0,10]
        left_satis = False
        right_satis = False
        left = max(0, peak_index - windowsize)
        right = min(signal.shape[0] - 1, peak_index + windowsize)

        # [0,10]
        while True:

            if signal[left] <= threshold * peak:
                left_satis = True

            if signal[right] <= threshold * peak:
                right_satis = True

            if left == 0:
                left_satis = True

            if right == signal.shape[0] - 1:
                right_satis = True

            if not left_satis:
                left -= 1

            if not right_satis:
                right += 1

            if left_satis and right_satis:
                break

        #peak_info = [left, peak_index, right]
        signal[left : right + 1] = freeze_value
        peak_list.append(peak_index)
        idx += 1
        if idx == count:
            break

    return peak_list

def genGausKernel1D(length, sigma = 16):
    x = np.linspace(-(length - 1) / 2.0, (length - 1) / 2.0, length)
    gaussian = (1.0 / np.sqrt(2 * np.pi * (sigma ** 2))) * np.exp(-(x ** 2) / (2 * (sigma ** 2)))
    sum = np.sum(gaussian)
    kernel_1d = np.true_divide(gaussian, sum)
    kernel_1d = kernel_1d.reshape((1, -1))

    return kernel_1d

def GenerateDensityMap(q, count):
    mean_density = np.zeros(q.shape[0])
    for dim in range(q.shape[1]):
        target = q[:, dim]
        peak_list = peak_detection(target, count)
        density_map = np.zeros(target.shape[0])
        peak_list.sort()
        for i in range(len(peak_list)):
            # First peak
            peak_index = peak_list[i]
            if i == 0:
                previous_peak_index = 0
            else:
                previous_peak_index = peak_list[i - 1]

            if i == len(peak_list) - 1:
                next_peak_index = target.shape[0] - 1
            else:
                next_peak_index = peak_list[i + 1]
            #print(previous_peak_index, peak_index, next_peak_index)
            #print(peak_list)
            gaussian_width = max(peak_index - previous_peak_index, next_peak_index - peak_index)
            # print(gaussian_width)
            gaussian_mask = genGausKernel1D(2 * gaussian_width + 1).squeeze()
            if peak_index - gaussian_width < 0:
                left_size = peak_index - 0
            else:
                left_size = gaussian_width

            if peak_index + gaussian_width > target.shape[0] - 1:
                right_size = target.shape[0] - 1 - peak_index
            else:
                right_size = gaussian_width
            #print(gaussian_width)
            #print(density_map.shape)
            #print(gaussian_mask.shape)
            density_map[peak_index - left_size: peak_index + right_size + 1] += gaussian_mask[
                                                                                gaussian_width - left_size: gaussian_width + right_size + 1]
        mean_density += density_map
    mean_density /= 6

    return mean_density