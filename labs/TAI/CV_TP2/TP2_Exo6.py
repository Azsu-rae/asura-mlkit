
import cv2

from pathlib import Path
imgpath_lr_128 = Path.home() / "Data" / "Images" / "LR_128X128"
imgpath_lr_256 = Path.home() / "Data" / "Images" / "LR_256X256"
imgpath_hr_gt_512 = Path.home() / "Data" / "Images" / "HR_GT_512X512"

for i in range(2, 24):

    lr_128 = cv2.imread("{}/LR_128_{}.jpg".format(imgpath_lr_128, i))
    if lr_128 is None:
        raise ValueError("No image!")

    lr_256 = cv2.imread("{}/LR_256_{}.jpg".format(imgpath_lr_256, i))
    if lr_256 is None:
        raise ValueError("No image!")

    hr_gt_512 = cv2.imread("{}/HR_GT_{}.jpg".format(imgpath_hr_gt_512, i))
    if hr_gt_512 is None:
        raise ValueError("No image!")

    res_128_256_lin = cv2.resize(lr_128, (256, 256), cv2.INTER_LINEAR)
    res_128_256_cub = cv2.resize(lr_128, (256, 256), cv2.INTER_CUBIC)

    res_128_512_lin = cv2.resize(res_128_256_lin, (512, 512), cv2.INTER_LINEAR)
    res_128_512_cub = cv2.resize(res_128_256_cub, (512, 512), cv2.INTER_CUBIC)

    cv2.imshow("Original 128", lr_128)

    cv2.imshow("LINEAR 256", res_128_256_lin)
    cv2.imshow("CUBIC 256", res_128_256_cub)
    cv2.imshow("Original 256", lr_256)

    cv2.imshow("LINEAR 512", res_128_512_lin)
    cv2.imshow("CUBIC 512", res_128_512_cub)
    cv2.imshow("Original 512", hr_gt_512)

    print(f"lr_256 with res_128_256_lin: {cv2.PSNR(lr_256, res_128_256_lin, 255.0)}")
    print(f"lr_256 with res_128_256_cub: {cv2.PSNR(lr_256, res_128_256_cub, 255.0)}")

    print(f"hr_gt_512 with res_128_512_lin: {cv2.PSNR(hr_gt_512, res_128_512_lin, 255.0)}")
    print(f"hr_gt_512 with res_128_512_cub: {cv2.PSNR(hr_gt_512, res_128_512_cub, 255.0)}")

    cv2.waitKey(0)
