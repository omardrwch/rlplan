import numpy as np


def masked_argmax(array, mask):
    """
    :param array: 1d numpy array
    :param mask:  1d list
    :return: index in array of np.argmax(array[mask])
    """
    mask_argmax = np.argmax(array[mask])
    array_argmax = np.arange(array.shape[0])[mask][mask_argmax]
    return array_argmax
