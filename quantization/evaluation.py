from torcheval.metrics import PeakSignalNoiseRatio
from typing import List, Union
from skimage.metrics import structural_similarity as ssim

def calculatePSNR(preds, targets, device):
    metric = PeakSignalNoiseRatio(device=device)
    for pred, target in zip(preds, targets):
        metric.update(pred, target)
    return metric.compute()

def calcSSIM(preds, targets, device):
    vals = []
    for pred, target in zip(preds, targets):
        pred = pred.squeeze()
        target = target.squeeze()
        pred = pred.detach().to('cpu').numpy()
        target = target.detach().to('cpu').numpy()
        print(pred.shape, target.shape)
        vals.append(ssim(pred, target, channel_axis=0, gaussian_weights=True, data_range=(target.max()-target.min())))
    mean_ssim = sum(vals) / len(vals)
    return mean_ssim
    




        
