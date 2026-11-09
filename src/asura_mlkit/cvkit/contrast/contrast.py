
#
def min_max_scalling(im, ymin, ymax, seuil):

    histo = cvkit.util.histo_gray(im)

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

#    for i in range(h):
#        for j in range(w):
#            output[i, j] = (im[i, j] - mn) / (mx - mn) * 255

#    return output.astype(np.uint8)
