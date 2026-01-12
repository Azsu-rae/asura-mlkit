
import numpy as np
import cvkit


# iterate through an image (leaving space as a padding) to 'do' something
def standard(im, ker, do):

    if ker % 2 == 0:
        raise ValueError('Kernel size must be an odd number!')

    pad = ker // 2
    h, w = im.shape[:2]

    output = np.copy(im)
    for i in range(pad, h-pad):
        for j in range(pad, w-pad):
            output[i][j] = do(i, j)

    return output

# ----------------- filtre moyenneur -----------------


def average(im, i, j, ker):
    pad = ker // 2
    avr = im[i-pad:i+pad+1, j-pad:j+pad+1].sum() / ker**2
    return np.astype(avr, np.uint8)


def mean(im, ker):
    return standard(im, ker, lambda i, j: average(im, i, j, ker))

# ----------------- filtre median -----------------


def middle(im, i, j, pad):
    return np.median(im[i-pad:i+pad+1, j-pad:j+pad+1])


def median(im, ker):
    return standard(im, ker, lambda i, j: middle(im, i, j, ker // 2))

# ----------------- gaussian -----------------


def gaussian_convolution(im, i, j, ker, sig):

    pad = ker // 2
    region = im[i-pad:i+pad+1, j-pad:j+pad+1]
    gaussian_ker = cvkit.utils.gaussian_kernel(ker, sig)

    return (region * gaussian_ker).sum() / gaussian_ker.sum()


def gaussian(im, ker, sigma):
    return standard(im, ker, lambda i, j: gaussian_convolution(im, i, j, ker, sigma))

# ----------------- bilateral -----------------


def bi_weighted_convolution(image, i, j, kernel, sigma_s, sigma_r):

    pad = kernel // 2
    sum_weights = 0.0
    sum_weighted_values = 0.0

    for k in range(-pad, pad+1):
        neighbor_i = i+k
        for l in range(-pad, pad+1):
            neighbor_j = j+l

            # TODO: DOUBLE CHECK THIS

            spatial_dist = np.sqrt(k*k + l*l)
            spatial_weight = np.exp(-spatial_dist / 2 * sigma_s**2)

            color_dist = abs(image[i, j].astype(int)-image[neighbor_i, neighbor_j].astype(int))
            color_weight = np.exp(-color_dist / 2 * sigma_r**2)

            weight = spatial_weight * color_weight

            sum_weights += weight
            sum_weighted_values += weight * image[neighbor_i, neighbor_j]

    resultat = (sum_weighted_values // sum_weights)

    return resultat


def bilateral(im, ker, sigma_s, sigma_r):
    return standard(im, ker, lambda i, j: bi_weighted_convolution(im, i, j, ker, sigma_s, sigma_r))

# ----------------- nagao -----------------


# pour recuperer touts les pixels d'une region donner
def region_projection(image, region, i, j):

    pixels = []
    for coordoner in region:
        k, l = coordoner
        pixels.append(image[i+k, j+l])

    return np.array(pixels)


# faires des iterations sur toutes les regions, recuperer la meiller, et finalement finir par recuperer la moyenne
def choose_region(im, i, j, ker):

    regions = cvkit.utils.nagao_regions()
    good_region = 0
    min_var = 0

    for k, r in enumerate(regions):
        var = np.var(region_projection(im, r, i, j))
        if min_var < var:
            min_variance = var
            good_region = k

    return region_projection(im, regions[good_region], i, j).sum() / len(regions[good_region])


def nagao(image, kernel):
    return standard(image, kernel, choose_region)


#
def min_max_scalling(im, ymin, ymax, seuil):

    histo = cvkit.utils.histo_gray(im)

    xmin = 0.0
    xmax = 255.0

    for i in range(256):
        if histo[i] > seuil:
            xmin = histo[i]
            break

    for i in range(255, -1, -1):
        if histo[i] > seuil:
            xmax = histo[i]
            break

    alpha = ((ymin * xmax) - (ymax * xmin)) / xmax - xmin
    beta = (ymax - ymin) / (xmax - xmin)

    h, w = im.shape[:2]
    output = np.copy(im)
    for i in range(h):
        for j in range(w):
            output[i, j] = alpha + beta * im[i, j]
            if output[i, j] < 0 or output[i, j] > 255:
                output[i, j] = im[i, j]

    for i in range(h):
        for j in range(w):
            output[i, j] = (im[i, j] - mn) / (mx - mn) * 255

    return output.astype(np.uint8)
