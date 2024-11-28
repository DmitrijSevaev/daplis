"""Module for computing timestamp differences.

Compares all timestamps from the same cycle for the given pair of pixels
against a given value (delta_window). Differences in that window are saved
and returned as a list.

This file can also be imported as a module and contains the following
functions:

    TODO remove
    * calculate_differences_2212 - calculate timestamp differences for
    the given pair of pixels. Works only with firmware version '2212'.
    
    * calculate_differences_2212_fast - calculate timestamp differences for
    the given pair of pixels. Works only with firmware version '2212'.
    Uses a faster algorithm than the function above.

"""

from typing import List
from warnings import warn

import numpy as np
import pandas as pd
from numpy import ndarray

from src.daplis.functions import utils


# TODO remove - deprecated
def calculate_differences_2212(
    data: List[float],
    pixels: List[int] | List[List[int]],
    pix_coor,
    delta_window: float = 50e3,
):
    """Calculate timestamp differences for firmware version 2212.

    Calculate timestamp differences for the given pixels and LinoSPAD2
    firmware version 2212.

    Parameters
    ----------
    data : list of array-like
        List of data arrays, each corresponding to a different TDC.
    pixels : List[int] | List[List[int]]
        List of pixel numbers for which the timestamp differences should
        be calculated or list of two lists with pixel numbers for peak
        vs. peak calculations.
    pix_coor : array-like
        Array for transforming the pixel address in terms of TDC (0 to 3)
        to pixel number in terms of half of the sensor (0 to 255).
    delta_window : float, optional
        Width of the time window for counting timestamp differences.
        The default is 50e3 (50 ns).

    Returns
    -------
    deltas_all : dict
        Dictionary containing timestamp differences for each pair of pixels.

    """

    # TODO: remove
    warn(
        "This function is deprecated. Use" "'calculate_differences_2212_fast'",
        DeprecationWarning,
        stacklevel=2,
    )
    # Dictionary for the timestamp differences, where keys are the
    # pixel numbers of the requested pairs
    deltas_all = {}

    pixels_left, pixels_right = utils.pixel_list_transform(pixels)

    # Find ends of cycles
    cycle_ends = np.argwhere(data[0].T[0] == -2)
    cycle_ends = np.insert(cycle_ends, 0, 0)

    for q in pixels_left:
        # First pixel in the pair
        tdc1, pix_c1 = np.argwhere(pix_coor == q)[0]
        pix1 = np.where(data[tdc1].T[0] == pix_c1)[0]
        for w in pixels_right:
            if w <= q:
                continue
            deltas_all[f"{q},{w}"] = []

            # Second pixel in the pair
            tdc2, pix_c2 = np.argwhere(pix_coor == w)[0]
            pix2 = np.where(data[tdc2].T[0] == pix_c2)[0]

            # Go over cycles, getting data for the appropriate cycle
            # only
            for cyc in range(len(cycle_ends) - 1):
                slice_from = cycle_ends[cyc]
                slice_to = cycle_ends[cyc + 1]
                pix1_slice = pix1[(pix1 >= slice_from) & (pix1 < slice_to)]
                if not np.any(pix1_slice):
                    continue
                pix2_slice = pix2[(pix2 >= slice_from) & (pix2 < slice_to)]
                if not np.any(pix2_slice):
                    continue

                # Calculate delta t
                tmsp1 = data[tdc1].T[1][pix1_slice]
                tmsp1 = tmsp1[tmsp1 > 0]
                tmsp2 = data[tdc2].T[1][pix2_slice]
                tmsp2 = tmsp2[tmsp2 > 0]
                for t1 in tmsp1:
                    deltas = tmsp2 - t1
                    ind = np.where(np.abs(deltas) < delta_window)[0]
                    deltas_all[f"{q},{w}"].extend(deltas[ind])

    return deltas_all


def calculate_differences_2212_fast(
    data: ndarray,
    pixels: List[int] | List[List[int]],
    pix_coor: ndarray,
    delta_window: float = 50e3,
    cycle_length: float = 4e9,
):
    """Calculate timestamp differences for firmware version 2212.

    Calculate timestamp differences for the given pixels and LinoSPAD2
    firmware version 2212.

    Parameters
    ----------
    data : ndarray
        Matrix of timestamps, where rows correspond to the TDCs.
    pixels : List[int] | List[List[int]]
        List of pixel numbers for which the timestamp differences should
        be calculated or list of two lists with pixel numbers for peak
        vs. peak calculations.
    pix_coor : ndarray
        Array for transforming the pixel address in terms of TDC (0 to 3)
        to pixel number in terms of half of the sensor (0 to 255).
    delta_window : float, optional
        Width of the time window for counting timestamp differences.
        The default is 50e3 (50 ns).
    cycle_length : float, optional
        Length of each acquisition cycle. The default is 4e9 (4 ms).

    Returns
    -------
    deltas_all : dict
        Dictionary containing timestamp differences for each pair of pixels.

    """

    # Dictionary for the timestamp differences, where keys are the
    # pixel numbers of the requested pairs
    deltas_all = {}

    pixels_left, pixels_right = utils.pixel_list_transform(pixels)

    # Find ends of cycles
    cycle_ends = np.argwhere(data[0].T[0] == -2)
    cycle_ends = np.insert(cycle_ends, 0, 0)

    for q in pixels_left:
        # First pixel in the pair
        tdc1, pix_c1 = np.argwhere(pix_coor == q)[0]
        pix1 = np.where(data[tdc1].T[0] == pix_c1)[0]
        for w in pixels_right:
            if w <= q:
                continue
            deltas_all[f"{q},{w}"] = []

            timestamps_1 = []
            timestamps_2 = []

            # Second pixel in the pair
            tdc2, pix_c2 = np.argwhere(pix_coor == w)[0]
            pix2 = np.where(data[tdc2].T[0] == pix_c2)[0]

            # Go over cycles, shifting the timestamps from each next
            # cycle by lengths of cycles before (e.g., for the 4th cycle
            # add 12 ms)
            for i, _ in enumerate(cycle_ends[:-1]):
                slice_from = cycle_ends[i]
                slice_to = cycle_ends[i + 1]
                pix1_slice = pix1[(pix1 >= slice_from) & (pix1 < slice_to)]
                if not np.any(pix1_slice):
                    continue
                pix2_slice = pix2[(pix2 >= slice_from) & (pix2 < slice_to)]
                if not np.any(pix2_slice):
                    continue

                # Shift timestamps by cycle length
                tmsp1 = data[tdc1].T[1][pix1_slice]
                tmsp1 = tmsp1[tmsp1 > 0]
                tmsp1 = tmsp1 + cycle_length * i

                tmsp2 = data[tdc2].T[1][pix2_slice]
                tmsp2 = tmsp2[tmsp2 > 0]
                tmsp2 = tmsp2 + cycle_length * i

                timestamps_1.extend(tmsp1)
                timestamps_2.extend(tmsp2)

            timestamps_1 = np.array(timestamps_1)
            timestamps_2 = np.array(timestamps_2)

            # Indicators for each pixel: 0 for timestamps from one pixel
            # 1 - from the other
            pix1_ind = np.zeros(len(timestamps_1), dtype=np.int32)
            pix2_ind = np.ones(len(timestamps_2), dtype=np.int32)

            pix1_data = np.vstack((pix1_ind, timestamps_1))
            pix2_data = np.vstack((pix2_ind, timestamps_2))

            # Dataframe for each pixel with pixel indicator and
            # timestamps
            df1 = pd.DataFrame(
                pix1_data.T, columns=["Pixel_index", "Timestamp"]
            )
            df2 = pd.DataFrame(
                pix2_data.T, columns=["Pixel_index", "Timestamp"]
            )

            # Combine the two dataframes
            df_combined = pd.concat((df1, df2), ignore_index=True)

            # Sort the timestamps
            df_combined.sort_values("Timestamp", inplace=True)

            # Subtract pixel indicators of neighbors; values of 0
            # correspond to timestamp differences for the same pixel
            # '-1' and '1' - to differences from different pixels
            df_combined["Pixel_index_diff"] = df_combined["Pixel_index"].diff()

            # Calculate timestamp difference between neighbors
            df_combined["Timestamp_diff"] = df_combined["Timestamp"].diff()

            # Get the correct timestamp difference sign
            df_combined["Timestamp_diff"] = (
                df_combined["Timestamp_diff"] * df_combined["Pixel_index_diff"]
            )

            # Collect timestamp differences where timestamps are from
            # different pixels
            filtered_df = df_combined[
                abs(df_combined["Pixel_index_diff"]) == 1
            ]

            # Save only timestamps differences in the requested window
            delta_ts = filtered_df[
                abs(filtered_df["Timestamp_diff"]) < delta_window
            ]["Timestamp_diff"].values

            deltas_all[f"{q},{w}"].extend(delta_ts)

    return deltas_all
