import numpy as np
import torch
from torchvision import transforms
import cv2 as cv
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral
from models import UANet
from sateImageDataset import get_datasets
import constants
from typing import List


def mask_preprocessing(mask_tensor: torch.Tensor):
    np_image = mask_tensor.argmax(dim=1).numpy().astype(np.uint8)

    img = cv.cvtColor(np_image.transpose(1, 2, 0), cv.COLOR_GRAY2BGR)

    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def image_preprocessing(image_tensor: torch.Tensor):
    np_image = image_tensor[0].numpy().astype(np.uint8)

    img = np.transpose(np_image, (1, 2, 0))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def mask_postprocessing(mask: np.ndarray, rgb_values: List[List[int]] = constants.CLASS_RGB_VALUES):
    class_map = []
    for color in rgb_values:
        eq = np.equal(mask, color)
        class_map.append(np.all(eq, axis=-1))

    return transforms.ToTensor()(np.stack(class_map, axis=-1)).float().unsqueeze(0)


def polymerization(mask_tensor: torch.Tensor):
    cv_mask = mask_preprocessing(mask_tensor)

    contours, _ = cv.findContours(cv_mask, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1)

    smooth = torch.zeros(512, 512).numpy()

    cv.drawContours(smooth, contours, -1, (255, 255, 255), thickness=cv.FILLED)

    smooth = cv.cvtColor(smooth, cv.COLOR_GRAY2RGB)

    return mask_postprocessing(smooth)


def blur(mask_tensor: torch.Tensor):
    cv_mask = mask_preprocessing(mask_tensor)

    blurred_mask = cv.GaussianBlur(cv_mask, (3, 3), 0)
    blurred_mask = cv.medianBlur(blurred_mask, 3)
    blurred_mask = cv.bilateralFilter(blurred_mask, 9, 75, 75)

    blurred_mask = cv.cvtColor(blurred_mask * 255, cv.COLOR_GRAY2RGB)

    return mask_postprocessing(blurred_mask)


def CRP(image_tensor: torch.Tensor, mask_tensor: torch.Tensor):
    cv_image = image_preprocessing(image_tensor)

    dcrf_model = dcrf.DenseCRF2D(cv_image.shape[0], cv_image.shape[1], 2)
    unary = unary_from_softmax(mask_tensor.numpy())
    unary = np.ascontiguousarray(unary)

    dcrf_model.setUnaryEnergy(unary.reshape(2, -1))

    dcrf_model.addPairwiseEnergy(create_pairwise_gaussian(sdims=(10, 10), shape=cv_image.shape[:2]),
                                 compat=3,
                                 kernel=dcrf.DIAG_KERNEL,
                                 normalization=dcrf.NORMALIZE_SYMMETRIC)

    dcrf_model.addPairwiseEnergy(create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20), img=cv_image, chdim=2),
                                 compat=10,
                                 kernel=dcrf.DIAG_KERNEL,
                                 normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = dcrf_model.inference(10)
    refined_mask = np.argmax(Q, axis=0).reshape((cv_image.shape[0], cv_image.shape[1])).astype(np.uint8)
    refined_mask = cv.cvtColor(refined_mask * 255, cv.COLOR_GRAY2RGB)
    return mask_postprocessing(refined_mask)


if __name__ == "__main__":
    CHECKPOINT = "Checkpoints/UANet/UANet_ResNet50_K_1/2024-07-13 20_13_50/model_49.pt"
    DEVICE = "cuda" if torch.cuda.is_available() else "mps"

    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('mps'))

    model = UANet().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    _, _, test_dataset = get_datasets(1)

    inverse_normalise = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])

    with torch.no_grad():
        for i in range(10):

            X, Y = test_dataset.__getitem__(i)
            X, Y = X.to(DEVICE).unsqueeze(0), Y.to(DEVICE).unsqueeze(0)

            pred = model(X)
            pred_poly = polymerization(pred.cpu().detach().clone())
            pred_blur = blur(pred.cpu().detach().clone())
            pred_crp = CRP(X.cpu().detach().clone(), pred.cpu().detach().clone())

            fig, ax = plt.subplots(5, 1, figsize=(20, 20), dpi=100)

            ax[0].imshow(transforms.ToPILImage()(Y.argmax(dim=1).float()), cmap="gray")
            ax[0].set_title("Ground Truth")
            ax[0].axis("off")

            ax[1].imshow(transforms.ToPILImage()(pred.argmax(dim=1).float()), cmap="gray")
            ax[1].set_title("Prediction")
            ax[1].axis("off")

            ax[2].imshow(transforms.ToPILImage()(pred_poly.argmax(dim=1).float()), cmap="gray")
            ax[2].set_title("Prediction + Polygonization")
            ax[2].axis("off")

            ax[3].imshow(transforms.ToPILImage()(pred_blur.argmax(dim=1).float()), cmap="gray")
            ax[3].set_title("Prediction + Blur")
            ax[3].axis("off")

            ax[4].imshow(transforms.ToPILImage()(pred_crp.argmax(dim=1).float()), cmap="gray")
            ax[4].set_title("Prediction + CRP")
            ax[4].axis("off")

            #plt.show()
            plt.savefig(f'example_{i}.png', dpi=100)
