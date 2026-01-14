
def histo_gray(im):

    h, w = im.shape[:2]

    histo = np.zeros(256)
    for i in range(h):
        for j in range(w):
            niveaux_gray = im[i, j]
            histo[niveaux_gray] += 1

    return histo


def histo_color(im):

    h, w = im.shape[:2]

    print(im[0, 0])
    histo = np.zeros((3, 255))
    for i in range(h):
        for j in range(w):

            blue = im[i, j][0]
            green = im[i, j][1]
            red = im[i, j][2]

            histo[0][blue] += 1
            histo[1][green] += 1
            histo[2][red] += 1

    return histo


def print_gray_histo(histo):
    print("niveaux_gray   occurences")
    for niveaux_gray, occurences in enumerate(histo):
        print(f"     {niveaux_gray}             {occurences}")


def normalization(im, threshold=0, Y_min=0, Y_max=255):

    histo = histo_gray(im)

    X_min = -1
    X_max = -1
    for i in range(len(histo)):
        if histo[i] > threshold and X_min == -1:
            X_min = i
        if histo[255-i] > threshold and X_max == -1:
            X_max = 255-i
        if X_min != -1 and X_max != -1:
            break

    h, w = im.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            ratio = ((im[i, j] - X_min) / (X_max - X_min))
            out[i, j] = (255 * ratio).astype(np.uint8)

    return out


def equalization(im):

    histo = histo_gray(im)
    cdf = histo.cumsum()

    no_pxls = histo.sum()
    h, w = im.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            level = im[i, j]
            out[i, j] = ((cdf[level] - cdf.min()) / (no_pxls - cdf.min())) * 255

    return out


def display_gray_histo(histo):

    x = range(0, 256)
    y = histo

    plt.plot(x, y)
    plt.xlabel("gray levels")
    plt.ylabel("occurences")
    plt.title("histogram of image")
    plt.show()
