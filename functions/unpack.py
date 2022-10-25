"""Unpack data from LinoSPAD2

Functions for unpacking either 'txt' of 'dat' data files of LinoSPAD2.
Functions for either 10, 512 or a given number of timestamps per acquisition
cycle per pixel are available.

This file can also be imported as a module and contains the following
functions:

    * unpack_txt_512 - unpacks the 'txt' data files with 512 timestamps
    per acquisition cycle
    * unpack_txt_10 - unpacks the 'txt' data files with 10 timestamps per
    acquisition cycle
    * unpack_binary_10 - unpacks the 'dat' data files with 10 timestamps
    per acquisition cycle
    * unpack_binary_512 - unpack the 'dat' data files with 512 timestamps
    per acquisition point
    * unpack_binary_flex - unpacks the 'dat' data files with a given number of
    timestamps per acquisition cycle
    * unpack_binary_df - unpacks the 'dat' data files with a given number of
    timestamps per acquisition cycle into a pandas dataframe. Unlike the
    others, this functions does not write '-1' nonvalid timestamps which makes
    this the fastest approach compared to other. The dataframe output allows
    faster plots using seaborn.

"""

from struct import unpack
import numpy as np
import pandas as pd


def unpack_txt_512(filename):
    """Unpacks the 'txt' data files with 512 timestamps per acquistion
    cycle.

    Parameters
    ----------
    filename : str
        File with data from LinoSPAD2 in which precisely 512 timestamps
        per acquistion cycle is written.

    Returns
    -------
    data_matrix : array_like
        2D matrix (256 pixels by 512*number-of-cycles) of data points.

    """

    data = np.genfromtxt(filename)
    data_matrix = np.zeros((256, int(len(data) / 256)))  # rows=#pixels,
    # cols=#cycles

    noc = len(data) / 512 / 256  # number of cycles, 512 data lines per pixel per
    # cycle, 256 pixels

    # =========================================================================
    # Unpack data from the txt, which is a Nx1 array, into a 256x#cycles matrix
    # using two counters. When i (number of the pixel) reaches the 255
    # (256th pixel), move k one step further - second data acquisition cycle,
    # when the algorithm goes back to the 1st pixel and writes the data right
    # next to the data from previous cycle.
    # =========================================================================
    k = 0
    while k != noc:
        i = 0
        while i < 256:
            data_matrix[i][k * 512 : k * 512 + 512] = (
                data[(i + 256 * k) * 512 : (i + 256 * k) * 512 + 512] - 2 ** 31
            )
            i = i + 1
        k = k + 1
    data_matrix = data_matrix * 17.857  # 2.5 ns from TDC 400 MHz clock read
    # out 140 bins from 35 elements of the delay line - average bin size
    # is 17.857 ps

    return data_matrix


def unpack_txt_10(filename):
    """Unpacks the 'txt' data files with 10 timestamps per acquistion
    cycle.

    Parameters
    ----------
    filename : str
        File with data from LinoSPAD2 in which precisely 10 timestamps
        per acquistion cycle is written.

    Returns
    -------
    data_matrix : array_like
        2D matrix (256 pixels by 10*number-of-cycles) of data points.

    """

    data = np.genfromtxt(filename)
    data_matrix = np.zeros((256, int(len(data) / 256)))  # rows=#pixels,
    # cols=#cycles

    noc = len(data) / 10 / 256  # number of cycles, 10 data lines per pixel per
    # cycle, 256 pixels

    # =====================================================================
    # Unpack data from the txt, which is a Nx1 array, into a 256x#cycles
    # matrix using two counters. When i (number of the pixel) reaches the
    # 255 (256th pixel), move k one step further - second data acquisition
    # cycle, when the algorithm goes back to the 1st pixel and writes the
    # data right next to the data from previous cycle.
    # =====================================================================
    k = 0
    while k != noc:
        i = 0
        while i < 256:
            data_matrix[i][k * 10 : k * 10 + 10] = (
                data[(i + 256 * k) * 10 : (i + 256 * k) * 10 + 10] - 2 ** 31
            )
            i = i + 1
        k = k + 1
    data_matrix = data_matrix * 17.857  # 2.5 ns from TDC 400 MHz clock read
    # out 140 bins from 35 elements of the delay line - average bin size
    # is 17.857 ps

    return data_matrix


def unpack_binary_10(filename):
    """Unpacks the 'dat' data files with 10 timestamps per acquistion
    cycle.

    Parameters
    ----------
    filename : str
        File with data from LinoSPAD2 in which precisely 10 timestamps
        per acquistion cycle is written.

    Returns
    -------
    data_matrix : array_like
        2D matrix (256 pixels by 10*number-of-cycles) of data points.

    """

    timestamp_list = []
    address_list = []

    with open(filename, "rb") as f:
        while True:
            rawpacket = f.read(4)  # read 32 bits
            if not rawpacket:
                break  # stop when the are no further 4 bytes to readout
            packet = unpack("<I", rawpacket)
            if (packet[0] >> 31) == 1:  # check validity bit: if 1
                # - timestamp is valid
                timestamp = packet[0] & 0xFFFFFFF  # cut the higher bits,
                # leave only timestamp ones
                # 2.5 ns from TDC 400 MHz clock read out 140 bins from 35
                # elements of the delay line - average bin sizу is 17.857 ps
                timestamp = timestamp * 17.857  # in ps
            else:
                timestamp = -1
            timestamp_list.append(timestamp)
            address = (packet[0] >> 28) & 0x3  # gives away only zeroes -
            # not in this firmware??
            address_list.append(address)
    # rows=#pixels, cols=#cycles
    data_matrix = np.zeros((256, int(len(timestamp_list) / 256)))

    noc = len(timestamp_list) / 10 / 256  # number of cycles, 10 data lines per
    # pixel per cycle, 256 pixels

    # pack the data from a 1D array into a 2D matrix
    k = 0
    while k != noc:
        i = 0
        while i < 256:
            data_matrix[i][k * 10 : k * 10 + 10] = timestamp_list[
                (i + 256 * k) * 10 : (i + 256 * k) * 10 + 10
            ]
            i = i + 1
        k = k + 1
    return data_matrix


def unpack_binary_512(filename):
    """Unpacks the 'dat' data files with 512 timestamps per acquistion
    cycle.

    Parameters
    ----------
    filename : str
        File with data from LinoSPAD2 in which precisely 512 timestamps
        per acquistion cycle is written.

    Returns
    -------
    data_matrix : array_like
        2D matrix (256 pixels by 512*number-of-cycles) of data points.

    """

    timestamp_list = []
    address_list = []

    with open(filename, "rb") as f:
        while True:
            rawpacket = f.read(4)  # read 32 bits
            if not rawpacket:
                break  # stop when the are no further 4 bytes to readout
            packet = unpack("<I", rawpacket)
            if (packet[0] >> 31) == 1:  # check validity bit: if 1
                # - timestamp is valid
                timestamp = packet[0] & 0xFFFFFFF  # cut the higher bits,
                # leave only timestamp ones
                # 2.5 ns from TDC 400 MHz clock read out 140 bins from 35
                # elements of the delay line - average bin sizу is 17.857 ps
                timestamp = timestamp * 17.857  # in ps
            else:
                timestamp = -1
            timestamp_list.append(timestamp)
            address = (packet[0] >> 28) & 0x3  # gives away only zeroes -
            # not in this firmware??
            address_list.append(address)
    # rows=#pixels, cols=#cycles
    data_matrix = np.zeros((256, int(len(timestamp_list) / 256)))

    noc = len(timestamp_list) / 512 / 256  # number of cycles, 512 data lines per
    # pixel per cycle, 256 pixels

    # pack the data from a 1D array into a 2D matrix
    k = 0
    while k != noc:
        i = 0
        while i < 256:
            data_matrix[i][k * 512 : k * 512 + 512] = timestamp_list[
                (i + 256 * k) * 512 : (i + 256 * k) * 512 + 512
            ]
            i = i + 1
        k = k + 1
    return data_matrix


def unpack_binary_flex(filename, lines_of_data: int = 512):
    """Unpacks the 'dat' data files with certain timestamps per acquistion
    cycle.

    Parameters
    ----------
    filename : str
        File with data from LinoSPAD2 in which precisely lines_of_data lines
        of data per acquistion cycle is written.
    lines_of_data: int, optional
        Number of binary-encoded timestamps in the 'dat' file. The default
        value is 512.

    Returns
    -------
    data_matrix : array_like
        A 2D matrix (256 pixels by lines_of_data X number-of-cycles) of
        timestamps.

    """

    timestamp_list = []
    address_list = []

    with open(filename, "rb") as f:
        while True:
            rawpacket = f.read(4)  # read 32 bits
            if not rawpacket:
                break  # stop when the are no further 4 bytes to readout
            packet = unpack("<I", rawpacket)
            if (packet[0] >> 31) == 1:  # check validity bit: if 1
                # - timestamp is valid
                timestamp = packet[0] & 0xFFFFFFF  # cut the higher bits,
                # leave only timestamp ones
                # 2.5 ns from TDC 400 MHz clock read out 140 bins from 35
                # elements of the delay line - average bin size is 17.857 ps
                timestamp = timestamp * 17.857  # in ps
            else:
                timestamp = -1
            timestamp_list.append(timestamp)
            address = (packet[0] >> 28) & 0x3  # gives away only zeroes -
            # not in this firmware??
            address_list.append(address)
    # rows=#pixels, cols=#cycles
    data_matrix = np.zeros((256, int(len(timestamp_list) / 256)))

    noc = len(timestamp_list) / lines_of_data / 256  # number of cycles,
    # lines_of_data data lines per pixel per cycle, 256 pixels

    # pack the data from a 1D array into a 2D matrix
    k = 0
    while k != noc:
        i = 0
        while i < 256:
            data_matrix[i][
                k * lines_of_data : k * lines_of_data + lines_of_data
            ] = timestamp_list[
                (i + 256 * k) * lines_of_data : (i + 256 * k) * lines_of_data
                + lines_of_data
            ]
            i = i + 1
        k = k + 1
    return data_matrix


def unpack_binary_df(
    filename, lines_of_data: int = 512, apply_mask: bool = True, cut_empty: bool = True
):
    """ Unpacks the 'dat' files with a given number of timestamps per acquisition cycle
    per pixel into a pandas dataframe. The fastest unpacking compared to others.

    Parameters
    ----------
    filename : str
        The 'dat' binary-encoded datafile.
    lines_of_data : int, optional
        Number of timestamps per acquisition cycle per pixel. The default is 512.
    apply_mask : bool, optional
        Switch for masking the warm/hot pixels. Default is True.
    cut_empty : bool, optional
        Switch for appending '-1', or either non-valid or empty timestamps, to the
        output. Default is True.

    Returns
    -------
    timestamps : pandas.DataFrame
        A dataframe with two columns: first is the pixel number, second are the
        timestamps.

    """

    mask = [
        15,
        16,
        29,
        39,
        40,
        50,
        52,
        66,
        73,
        93,
        95,
        96,
        98,
        101,
        109,
        122,
        127,
        196,
        210,
        231,
        236,
        238,
    ]

    timestamp_list = list()
    pixels = list()
    i = 0
    cycles = lines_of_data * 256

    with open(filename, "rb") as f:
        while True:
            i += 1
            if i == cycles:
                i = 0
            rawpacket = f.read(4)
            if not rawpacket:
                break
            packet = unpack("<I", rawpacket)
            if (packet[0] >> 31) == 1:
                # timestamps cover last 28 bits of the total 32
                timestamp = packet[0] & 0xFFFFFFF
                # 17.857 ps - average bin size
                timestamp = timestamp * 17.857
                pixels.append(int(i / 512) + 1)
                timestamp_list.append(timestamp)
            elif cut_empty is False:
                timestamp_list.append(-1)
                pixels.append(int(i / 512 + 1))
    dic = {"Pixel": pixels, "Timestamp": timestamp_list}
    timestamps = pd.DataFrame(dic)
    timestamps = timestamps[~timestamps["Pixel"].isin(mask)]

    return timestamps
