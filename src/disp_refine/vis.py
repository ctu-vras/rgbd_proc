import cv2


def colorize_disp(disp, max_disp=None):
    """
    Colorize disparity map for visualization.
    :param disp: Disparity map (H, W)
    :return: Colorized disparity map (H, W, 3)
    """
    if max_disp is None:
        max_disp = max(disp.max(), 1e-6)  # Avoid division by zero
    disp_vis = cv2.convertScaleAbs(disp, alpha=255.0 / max_disp)
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    return disp_vis
