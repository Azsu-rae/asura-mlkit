
import numpy as np
import cv2

im1 = cv2.imread("Images/Cerveau_AVC1.jpg", cv2.IMREAD_GRAYSCALE)
if im1 is None:
    raise ValueError("No Image")

imb = cv2.GaussianBlur(im1, (5, 5), 1.4)

neighbors = [
        (-1, -1), (-1, 0), (+1, +1),
        (0, -1), (0, +1),
        (+1, -1), (+1, 0), (-1, +1),
    ]


def sequential_region_growing(imb, seuil=15):
    out = np.zeros(imb.shape, dtype=np.uint32)
    h, w = imb.shape[:2]
    reg_mean = []
    reg_id = 1
    for i in range(h):
        for j in range(w):

            if out[i, j] == 0:
                reg_mean.append(region_growing(imb, (i, j), out, reg_id))
                reg_id += 1

    print(f"found {reg_id} regions in a {imb.shape} image!")

    output = imb.copy().astype(np.float32)
    for reg in range(len(reg_mean)):
        output[out == reg+1] = reg_mean[reg]

    return output.astype(np.uint8)


def region_growing(imb, seed_point, out, reg_id, seuil=15):

    points = []
    points.append(seed_point)

    region = []
    region.append(imb[seed_point])

    region_sum = 0
    region_sum += float(imb[seed_point])
    region_count = 1

    out[seed_point] = reg_id

    h, w = imb.shape[:2]
    visited = np.zeros((h, w))

    while points:

        curr = points.pop(0)
        for n in neighbors:
            di, dj = n
            i, j = curr
            i += di
            j += dj
            if i >= 0 and j >= 0 and i < h and j < w:

                if visited[i, j] or out[i, j] != 0:
                    continue
                visited[i, j] = True

                current_mean = region_sum / region_count
                if abs(imb[i, j] - current_mean) <= seuil:
                    points.append((i, j))
                    region.append(imb[i, j])
                    out[i, j] = reg_id
                    region_sum += float(imb[i, j])
                    region_count += 1

        # critere d'homoginite?

    return np.mean(region)


cv2.imshow("test zone", sequential_region_growing(imb))
cv2.imshow("original", imb)
cv2.waitKey(0)
