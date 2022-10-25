""" Function for calculating timestamp differences for the given pair of
pixels.

This file can also be imported as a module and contains the following
functions:

    * calculate differences - calculate timestamp differences for a given
    pair of pixels. Returns an array of differences.

"""


def calc_diff(
    data_pair,
    timestamps: int = 512,
    range_left: float = -2.5e3,
    range_right: float = 2.5e3,
):
    """
    Function for calculating timestamp differences for a given pair of pixels.

    Parameters
    ----------
    data_pair : array-like
        Data from 2 pixels.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition window. The default is
        512.
    range_left : float, optional
        Left limit for the timestamp differences. Values below that are not
        taken into account and the differences are not returned. The default
        is -2.5e3.
    range_right : float, optional
        Right limit for the timestamp differences. Values above that are not
        taken into account and the differences are not returned. The default
        is 2.5e3.

    Returns
    -------
    output : array-like
        An array of timestamp differences for the given pair of pixels.

    """

    minuend = len(data_pair)
    timestamps_total = len(data_pair[0])
    subtrahend = len(data_pair)

    output = []

    for i in range(minuend):
        acq = 0  # number of acq cycle
        for j in range(timestamps_total):
            if j % timestamps == 0:
                acq = acq + 1  # next acq cycle
            if data_pair[i][j] == -1:
                continue
            for k in range(subtrahend):
                if k <= i:
                    continue  # to avoid repetition: 2-1, 53-45
                for p in range(timestamps):
                    n = timestamps * (acq - 1) + p
                    if data_pair[k][n] == -1:
                        continue
                    elif data_pair[i][j] - data_pair[k][n] > range_right:
                        continue
                    elif data_pair[i][j] - data_pair[k][n] < range_left:
                        continue
                    else:
                        output.append(data_pair[i][j] - data_pair[k][n])
    return output

# TODO: finish the code!
def calc_diff_df(
    data_1,
    data_2,
    timestamps: int = 512,
    range_left: float = -2.5e3,
    range_right: float = 2.5e3,
):
    """
    Function for calculating timestamp differences for a given pair of pixels.

    Parameters
    ----------
    data_1 : pandas.Series
        Data from the 1st pixel of the pair.
    data_2 : pandas.Series
        Data from the 2nd pixel of the pair.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition window. The default is
        512.
    range_left : float, optional
        Left limit for the timestamp differences. Values below that are not
        taken into account and the differences are not returned. The default
        is -2.5e3.
    range_right : float, optional
        Right limit for the timestamp differences. Values above that are not
        taken into account and the differences are not returned. The default
        is 2.5e3.

    Returns
    -------
    output : array-like
        An array of timestamp differences for the given pair of pixels.

    """

    minuend = 1
    # timestamps_total = len(data_pair[0])
    subtrahend = 2

    # output = []

    # for i in range(minuend):
    #     acq = 0  # number of acq cycle
    #     for j in range(timestamps_total):
    #         if j % timestamps == 0:
    #             acq = acq + 1  # next acq cycle
    #         for k in range(subtrahend):
    #             if k <= i:
    #                 continue  # to avoid repetition: 2-1, 53-45
    #             for p in range(timestamps):
    #                 n = timestamps * (acq - 1) + p
    #                 if data_pair[i][j] - data_pair[k][n] > range_right:
    #                     continue
    #                 elif data_pair[i][j] - data_pair[k][n] < range_left:
    #                     continue
    #                 else:
    #                     output.append(data_pair[i][j] - data_pair[k][n])
    # return output