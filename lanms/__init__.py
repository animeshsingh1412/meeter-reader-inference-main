import subprocess
import os
import numpy as np

__pdoc__ = {
    'adaptor': False
}

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
    raise RuntimeError('Cannot compile lanms: {}'.format(BASE_DIR))

def merge_quadrangle_n9(polys, thres=0.3, precision=10000):
    """
    Merge polygons that are nearby into a single polygon using local nonmax-suppression.
    Args:
        polys: `float32` A numpy array of shape `(N, 9)`. `N` is number of polygons. `[[X0, Y0, X1, Y1, X2, Y2, X3, Y3, score]]`.
        thres: Polygons with score less than `thres` value are ignored for local nonmax-suppression.
        precision: A multiplier for polygon coordinates to improve the acccuracy when comapring floating point coordinates.
    Returns:
        `float32` List of merged polygons. Shape is `(M, 9)`. `M` is number of polygons. `[[X0, Y0, X1, Y1, X2, Y2, X3, Y3, score]]`.
    """
    from .adaptor import merge_quadrangle_n9 as nms_impl
    if len(polys) == 0:
        return np.array([], dtype='float32')
    p = polys.copy()
    p[:,:8] *= precision
    ret = np.array(nms_impl(p, thres), dtype='float32')
    ret[:,:8] /= precision
    return ret

