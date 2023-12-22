import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor


def run_ours_point(image, pts_sampled, model):
    img_tensor = ToTensor()(image)
    pts_sampled = torch.reshape(torch.tensor(pts_sampled), [1, 1, -1, 2])
    max_num_pts = pts_sampled.shape[2]
    pts_labels = torch.ones(1, 1, max_num_pts)

    predicted_logits, predicted_iou = model(
        img_tensor[None, ...],
        pts_sampled,
        pts_labels,
    )
    predicted_logits = predicted_logits.cpu()
    all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy()
    predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()

    max_predicted_iou = -1
    selected_mask_using_predicted_iou = None
    for m in range(all_masks.shape[0]):
        curr_predicted_iou = predicted_iou[m]
        if (
            curr_predicted_iou > max_predicted_iou
            or selected_mask_using_predicted_iou is None
        ):
            max_predicted_iou = curr_predicted_iou
            selected_mask_using_predicted_iou = all_masks[m]
    return max_predicted_iou, selected_mask_using_predicted_iou

start = time.time()
model = torch.jit.load('efficientsam_ti_cpu.jit')

time_1 = time.time() - start
add_mask = np.array([[[[500, 1100], [200, 1000],[300, 750], [700, 600], [300, 1350], [350, 1300]]]])
remove_area = np.array([[[[250, 1400], [300, 1400], [300, 1350], [300, 1349], [300, 1348], [300, 1347], [300, 1346], [300, 1345]]]])
input_label = np.array([1])
image_path = './target.jpeg'

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

mask_confidence, mask = run_ours_point(image, add_mask, model)
remove_area_confidence, remove_area = run_ours_point(image, remove_area, model)

true_mask = (mask_confidence * mask) > (remove_area_confidence * remove_area)
print(mask_confidence, remove_area_confidence)
image_rgba[:, :, 3] = true_mask.astype(np.uint8) * 255

plt.figure(figsize=(10, 10))
plt.imshow(image_rgba)
plt.gca().scatter(250, 1400, color='green', marker="*")
plt.show()
