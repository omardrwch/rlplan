import pytest
import numpy as np
from rlplan.utils import masked_argmax

array_mask_argmax_params = [
                            (np.array([5, 10, 20, 30, 40, 50]), [1, 3, 5], 5),
                            (np.array([100, 10, 20, 30, 40, 50]), [0, 1], 0)
                           ]


@pytest.mark.parametrize("array,mask,argmax", array_mask_argmax_params)
def test_masked_armax(array, mask, argmax):
    assert masked_argmax(array, mask) == argmax
