import os
from PIL import Image
import numpy as np

def image2array(path, threshhold=0):
    r"""
    Converts an image to an occupancy-grid array where

    - 0 represents free space,
    - 1 represents an obstacle

    The returned array is transposed, so indexing works via worldMap[x,y].

    Parameters
    ----------
    path : *str*
        A (valid)path to an image.
    threshhold : *int*
        Any pixelvalue below /threshhold/ will be counted as occupied space.

    Returns
    -------
    worldMap : *array_like*
        Transposed Occupancy-Grid of the converted image.

    """
    if not os.path.exists(path):
        raise IOError("Invalid Path: "+path)
    try:
        worldMap = np.asarray(Image.open(path).convert("L"))
        worldMap.setflags(write=1)
        obstacles = np.where(worldMap <= threshhold)
        worldMap[:,:] = 0
        worldMap[obstacles] = 1
        return worldMap
    except IOError:
        msg = "Cannot convert image ",path
        raise IOError(msg)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    path = './maps/mensa_example.png'
    path = './maps/rightWall_6x4.png'
    res = image2array(path)
    print(res.shape)
    # plt.imshow(res)
    # plt.show()
