import cv2


def colorize_img(img, max_val=None):
    """
    Colorize image for visualization.
    :param img: 1-channel image (H, W)
    :return: Colorized image (H, W, 3)
    """
    if max_val is None:
        max_val = max(img.max(), 1e-6)  # Avoid division by zero
    img_vis = cv2.convertScaleAbs(img, alpha=255.0 / max_val)
    img_vis = cv2.applyColorMap(img_vis, cv2.COLORMAP_JET)
    return img_vis
