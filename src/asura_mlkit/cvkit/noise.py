
import numpy as np


def salt_pepper(image, prob):

    random_mask = np.random.random(image.shape[:2])
    prob_pepper = prob / 2
    prob_salt = prob / 2

    # prob is between [0, 1)
    # random() generates evenly distributed random values between [0, 1)

    output = np.copy(image)
    output[random_mask < prob_salt] = 255
    output[random_mask > (1 - prob_pepper)] = 0

    # for prob = 0.1:
    # all < 0.05: 255
    # all > 0.95: 0

    return output


def gaussian(image, mean, variance):
    noise = np.random.normal(mean, variance, image.shape)
    noised_image = image + noise
    return np.clip(noised_image, 0, 255).astype(np.uint8)
