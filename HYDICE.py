# codin = utf-8
"""
Author: Ninghuyan
Email: huyanning@stu.xidian.edu.cn
Time 2018-07-18
Please email me if you find bugs, or have suggestions or questions!

Huyan N, Zhang X, Zhou H, et al. Hyperspectral Anomaly Detection via Background and
Potential Anomaly Dictionaries Construction[J].
IEEE Transactions on Geoscience and Remote Sensing, 2018.
"""


import numpy as np
# import HyperProTool as hyper
import scipy.io as sio
from LRSR import LRSR
#from LRSR_1 import LRSR
from dic_constr import dic_constr
from result_show import result_show
from ROC_AUC import ROC_AUC
import HyperProTool as hyper

# data pre-precessing
data = sio.loadmat("C:/Users/DRASHTI/OneDrive/Desktop/Academics/ML Project/Dataset/HYDICE_data.mat")
data3d = np.array(data["data"], dtype=float)
data3d = data3d[0:80, 0:100, :]
# Global scaling
data3d = (data3d - np.min(data3d)) / (np.max(data3d) - np.min(data3d))
# remove_bands = np.hstack((range(1, 4, 1), range(76), range(87), range(101, 111, 1), range(136, 153, 1)))
# data3d = np.delete(data3d, remove_bands, axis=2)
rows, cols, bands = data3d.shape
groundtruthfile = sio.loadmat("C:/Users/DRASHTI/OneDrive/Desktop/Academics/ML Project/Dataset/HYDICE_GT.mat")
groundtruth = np.array(groundtruthfile["map"])
rows, cols, bands = data3d.shape
from scipy import ndimage
# Apply a small Gaussian filter to each band
for i in range(data3d.shape[2]):
    data3d[:,:,i] = ndimage.gaussian_filter(data3d[:,:,i], sigma=0.5)
# background and anomaly dictionary construction
data2d, bg_dic, tg_dic,bg_dic_label, tg_dic_label = dic_constr(data3d, groundtruth, 5, 12, 8, 0.02, 300)

# low rank and sparse representaion
Z, E, S = LRSR(bg_dic, tg_dic, data2d, 0.005, 0.05)

# result visualization
background2d, target2d = result_show(bg_dic, tg_dic, Z, S, E, rows, cols, bands, bg_dic_label, tg_dic_label)
threshold = 0.15  # Adjust based on experimentation
target2d = np.where(target2d > threshold, target2d, 0)
# After getting target2d from result_show
from scipy import ndimage
# Remove small isolated detections
target2d = ndimage.median_filter(target2d, size=3)
# Enhance anomaly regions
target2d = ndimage.maximum_filter(target2d, size=2)

# ROC curve show
auc = ROC_AUC(target2d, groundtruth)
print("The AUC is: {0}".format(auc))

