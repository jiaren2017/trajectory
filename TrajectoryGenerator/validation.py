r"""Validation of traces/points against an Occupancy-grid."""
import wx
import warnings

def validate_trace(trace_x, trace_y, worldMap, batch):
    r"""Check if a trace is valid against an Occupancy-grid.

    Parameters
    ----------
    trace_x, trace_y : *array_like*/*list*
        Lists containing the individual values of the trace.
    worldMap : *array*
        Occupancy-grid from `Sillywalks.sillywalks.mapCreation.image2array`.
    batch : *boolean*
        Indicates if the code is running in batch-mode. Necessary for
        error-indication since in this case a `wx.MessageBox` is not
        wanted.

    Returns
    -------
    valid : *bool*
        *True* if all Trace-coordinates lie in free space, *False* if otherwise.

    """
    assert (len(trace_x) > 0) and (len(trace_y) > 0) and (len(trace_x) == len(trace_y))
    valid = True
    for x, y in zip(trace_x, trace_y):
        if validate_point(x, y, worldMap, batch) == False:
            valid = False
    return valid

def validate_point(x, y, worldMap, batch):
    r"""Check if a pixel in the Occupancy-grid is occupied.

    If Coordinates fall outside of the grid-map, True is returned!
    Parameters
    ----------
    x, y : *int*/*int*
        Pixel-coordinates
    worldMap : *array_like*
        Occupancy-grid from `Sillywalks.sillywalks.mapCreation.image2array`.
    batch : *boolean*
        Indicates if the code is running in batch-mode. Necessary for
        error-indication since in this case a `wx.MessageBox` is not
        wanted.

    Returns
    -------
    valid : *bool*
        *True* if Input-coordinates lie in free space, *False* if otherwise.

    """
    try:
        x = int(x)
        y = int(y)
    except:
        msg = "Could not convert Trace-positions to integer: "+str(x)+" - "+str(y)
        if batch:
            print(msg)
        else:
            wx.MessageBox(msg, 'Error', \
                          wx.OK | wx.ICON_ERROR)
        raise ValueError(msg)
    ysize, xsize = worldMap.shape
    if not (0 <= x < xsize):
        msg = "Invalid Trace-X-Position: "+str(x)+" with Image-X-Size: "+str(xsize)
        msg += "\nMaybe modify the trace-points or reduce noise to create a valid trace."
        if batch:
            warnings.warn(msg)
        else:
            wx.MessageBox(msg, 'Error', \
                          wx.OK | wx.ICON_ERROR)
            return True
    if not (0 <= y < xsize):
        msg = "Invalid Trace-Y-Position: "+str(y)+" with Image-Y-Size: "+str(ysize)
        msg += "\nMaybe modify the trace-points or reduce noise to create a valid trace."
        if batch:
            warnings.warn(msg)
        else:
            wx.MessageBox(msg, 'Error', \
                          wx.OK | wx.ICON_ERROR)
        return True
    if worldMap[y][x] == 1:
        return False
    else:
        return True
