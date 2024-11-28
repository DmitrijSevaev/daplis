"""Module for working with the calibration data.

Functions in this module can be used to analyze the LinoSPAD2 data
to collect either a calibration matrix for compensating TDC 
nonlinearities or a calibration matrix for offset calibration. First
nonlinearity is introduced by nonequal bins of the 140-bin long TDC line
while the second - from the different-length electrical path in the PCB.

The calibration matrices (for TDC calibration) and arrays (for offset
calibration) should be put into ~/LinoSPAD2/src/LinoSPAD2/params
/calibration_data folder, where it will be pulled from by other
functions during analysis.

This file can also be imported as a module and contains the following
functions:

    * calibrate_and_save_TDC_calibration - calculate a calibration
    matrix of the TDC calibrations and save it as a '.csv' table.

    * unpack_data_for_offset_calibration - unpack binary data applying
    TDC calibration in the process. Used for calculations of the offset
    calibration.

    * save_offset_timestamp_differences - calculate and save timestamps
    differences for pairs of pixels 0, 4 to 255, and 1 to 3 for finding
    the delta t peaks for calculating the offset calibration.

    * calculate_and_save_offset_calibration - calculate and save the 256
    offset compensations for all pixels of the given LinoSPAD2 sensor
    half. The output is saved as a .npy file for later use.

    * load_calibration_data - load the calibration matrix from a '.csv'
    table.

"""

import glob
import os
import sys
import time
from math import ceil

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm


def calibrate_and_save_TDC_data(
    path: str,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int = 1000,
):
    """Calculate and save calibration data for TDC nonlinearities.

    Function for calculating the calibration matrix and saving it into a
    '.csv' file. The data files used for the calculation should be taken
    with the sensor uniformly illuminated by ambient light.

    Parameters
    ----------
    path : str
        Path to the folder with the '.dat' data files.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard number, including the "#".
    firmware_version : str
        LinoSPAD2 firmware version. Versions "2208", "2212s" (skip), and
        "2212b" (block) are recognized.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The
        default is 1000.

    Returns
    -------
    None.

    Raises
    ------
    TypeError
        If 'daughterboard_number', 'motherboard_number', or
        'firmware_version' parameters are not of string type.

    Notes
    -----
    The resulting calibration matrix is saved as a '.csv' file, with the
    filename formatted as "TDC_{db}_{mb}_{fw_ver}.csv", where {db},
    {mb}, and {fw_ver} represent the daughterboard number, motherboard
    number, and firmware version, respectively.
    """

    # Parameter type check
    if not isinstance(daughterboard_number, str):
        raise TypeError("'daughterboard_number' should be a string.")
    if not isinstance(motherboard_number, str):
        raise TypeError("'motherboard_number' should be a string.")
    if not isinstance(firmware_version, str):
        raise TypeError(
            "'firmware_version' should be a string, '2208', '2212b' or"
            "'2212s'."
        )

    os.chdir(path)
    files = glob.glob("*.dat")

    # Go over all '.dat' files
    for j, file in enumerate(
        tqdm(
            files,
            desc="Calculating TDC calibration, going through files",
        )
    ):
        if firmware_version == "2208":
            # read data by 32 bit words
            raw_data = np.fromfile(file, dtype=np.uint32)
            # lowest 28 bits are the timestamp; convert to ps
            data = (raw_data & 0xFFFFFFF).astype(int) % 140
            # mask nonvalid data with '-1'; 0x80000000 - the 31st, validity bin
            data[np.where(raw_data < 0x80000000)] = -1
            # number of acquisition cycles
            cycles = int(len(data) / timestamps / 256)

            data_matrix = (
                data.reshape(cycles, 256, timestamps)
                .transpose((1, 0, 2))
                .reshape(256, timestamps * cycles)
            )

            # calibration matrix
            cal_mat = np.zeros((256, 140))
            bins = np.arange(0, 141, 1)

            for i in range(256):
                # sort the data into 140 bins
                counts, _ = np.histogram(data_matrix[i], bins=bins)
                # redefine the bin edges using the bin population from above
                cal_mat[i] = np.cumsum(counts) / np.cumsum(counts).max() * 2500

            cal_mat_df = pd.DataFrame(cal_mat)
            cal_mat_df.to_csv(
                f"TDC_{daughterboard_number}_{motherboard_number}_"
                f"{firmware_version}_{j}.csv"
            )

        elif firmware_version == "2212b" or firmware_version == "2212s":
            # read data by 32 bit words
            raw_data = np.fromfile(file, dtype=np.uint32)
            # lowest 28 bits are the timestamp; convert to ps
            data_t = (raw_data & 0xFFFFFFF).astype(int) % 140
            # pix address in the given TDC is 2 bits above timestamp
            data_p = ((raw_data >> 28) & 0x3).astype(np.longlong)
            data_t[np.where(raw_data < 0x80000000)] = -1
            # number of acquisition cycle in each data file
            cycles = int(len(data_t) / timestamps / 65)
            # transform into matrix 65 by cycles*timestamps
            data_matrix_p = (
                data_p.reshape(cycles, 65, timestamps)
                .transpose((1, 0, 2))
                .reshape(65, timestamps * cycles)
            )

            data_matrix_t = (
                data_t.reshape(cycles, 65, timestamps)
                .transpose((1, 0, 2))
                .reshape(65, timestamps * cycles)
            )

            # cut the 65th TDC that does not hold any actual data from pixels
            data_matrix_p = data_matrix_p[:-1]
            data_matrix_t = data_matrix_t[:-1]

            data_all = np.stack((data_matrix_p, data_matrix_t), axis=2).astype(
                np.longlong
            )

            # calibration matrix
            cal_mat = np.zeros((256, 140))
            bins = np.arange(0, 141, 1)

            if firmware_version == "2212b":
                pix_coor = np.arange(256).reshape(64, 4)
            else:
                pix_coor = np.arange(256).reshape(4, 64).T

            for i in range(256):
                # transform pixel number to TDC number and pixel coordinates in
                # that TDC (from 0 to 3)
                tdc, pix = np.argwhere(pix_coor == i)[0]
                # find data from that pixel
                ind = np.where(data_all[tdc].T[0] == pix)[0]
                # cut non-valid timestamps ('-1's)
                ind = ind[np.where(data_all[tdc].T[1][ind] >= 0)[0]]
                if not np.any(ind):
                    continue

                counts, _ = np.histogram(data_all[tdc].T[1][ind], bins=bins)
                cal_mat[i] = np.cumsum(counts) / np.cumsum(counts).max() * 2500

            cal_mat_df = pd.DataFrame(cal_mat)
            cal_mat_df.to_csv(
                f"TDC_{daughterboard_number}_{motherboard_number}_"
                f"{firmware_version}_{j}.csv"
            )

    # Combine all '.csv' files and average
    files_csv = glob.glob("*.csv")

    data_csv = np.zeros((256, 140))

    for i, file_csv in enumerate(files_csv):
        data_csv += pd.read_csv(file_csv, index_col=0)

    data_csv = data_csv / (i + 1)

    # Save the averaged matrix of calibration data into a '.csv' file
    data_csv.to_csv(
        f"TDC_{daughterboard_number}_{motherboard_number}_"
        f"{firmware_version}.csv"
    )

    # Remove the numbered '.csv' files
    file_pattern = f"TDC_{daughterboard_number}_{motherboard_number}"
    f"_{firmware_version}_*.csv"
    files_to_delete = glob.glob(file_pattern)
    for file_to_delete in files_to_delete:
        os.remove(file_to_delete)


def unpack_data_for_offset_calibration(
    path: str,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int = 1000,
):
    """Unpack data for offset calibration from LinoSPAD2 firmware version 2212.

    Unpacks binary-encoded data from LinoSPAD2 firmware version 2212.
    Returns a 3D array where rows are TDC numbers, columns are the data,
    and each cell contains a pixel number in the TDC (from 0 to 3) and the
    timestamp recorded by that pixel. TDC calibration is applied as it
    is necessary for the offset calibration to succeed.

    Parameters
    ----------
    path : str
        Path to the '.dat' data file.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number, including the "#".
    firmware_version : str
        LinoSPAD2 firmware version. Either '2212s' or '2212b' are accepted.
    timestamps : int, optional
        Number of timestamps per TDC per acquisition cycle. The default
        is 1000.

    Raises
    ------
    TypeError
        If 'daughterboard_number', 'motherboard_number', or
        'firmware_version' parameters are not of string type.
    FileNotFoundError
        If no calibration data is found, raise an error.

    Returns
    -------
    data_all : array-like
        3D array of pixel coordinates in the TDC and the timestamps.

    Notes
    -----
    This function unpacks binary-encoded data from LinoSPAD2 firmware
    version 2212 for offset calibration. The resulting data is returned
    as a 3D array where rows are TDC numbers, columns are the data, and
    each cell contains a pixel number in the TDC (from 0 to 3) and the
    timestamp recorded by that pixel. TDC calibration is applied as it
    is necessary for the offset calibration to succeed. The calibration
    data is loaded from the specified path based on the daughterboard,
    motherboard, and firmware version parameters.
    """
    # Parameter type check
    if not isinstance(daughterboard_number, str):
        raise TypeError("'daughterboard_number' should be a string.")
    if not isinstance(motherboard_number, str):
        raise TypeError("'motherboard_number' should be a string.")
    if not isinstance(firmware_version, str):
        raise TypeError(
            "'firmware_version' should be a string, '2212s' or '2212b'."
        )

    # Unpack binary data
    raw_file_data = np.fromfile(path, dtype=np.uint32)
    # Timestamps are lower 28 bits
    data_timestamps = (raw_file_data & 0xFFFFFFF).astype(np.longlong)
    # Pix address in the given TDC is 2 bits above timestamp
    data_pix = ((raw_file_data >> 28) & 0x3).astype(np.longlong)
    data_timestamps[np.where(raw_file_data < 0x80000000)] = -1
    # Number of acquisition cycle in each data file
    cycles = int(len(data_timestamps) / timestamps / 65)
    # Transform into matrix 65 by cycles*timestamps
    data_matrix_pix = (
        data_pix.reshape(cycles, 65, timestamps)
        .transpose((1, 0, 2))
        .reshape(65, timestamps * cycles)
    )

    data_matrix_timestamps = (
        data_timestamps.reshape(cycles, 65, timestamps)
        .transpose((1, 0, 2))
        .reshape(65, timestamps * cycles)
    )
    # Cut the 65th TDC that does not hold any actual data from pixels
    data_matrix_pix = data_matrix_pix[:-1]
    data_matrix_timestamps = data_matrix_timestamps[:-1]
    # Insert '-2' at the end of each cycle
    data_matrix_pix = np.insert(
        data_matrix_pix,
        np.linspace(timestamps, cycles * timestamps, cycles).astype(
            np.longlong
        ),
        -2,
        1,
    )

    data_matrix_timestamps = np.insert(
        data_matrix_timestamps,
        np.linspace(timestamps, cycles * timestamps, cycles).astype(
            np.longlong
        ),
        -2,
        1,
    )
    # Combine both matrices into a single one, where each cell holds pix
    # coordinates in the TDC and the timestamp
    data_all = np.stack(
        (data_matrix_pix, data_matrix_timestamps), axis=2
    ).astype(np.longlong)

    # Path to the calibration data directory
    pix_coordinate_array = np.arange(256).reshape(64, 4)
    path_calibration_data = (
        os.path.realpath(__file__) + "/../.." + "/params/calibration_data"
    )

    try:
        cal_matrix = load_calibration_data(
            path_calibration_data,
            daughterboard_number,
            motherboard_number,
            firmware_version,
            include_offset=False,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "No .csv file with the calibration data was found, "
            "check the path or run the calibration."
        )

    for i in range(256):
        # Transform pixel number to TDC number and pixel coordinates in
        # that TDC (from 0 to 3)
        tdc, pix = np.argwhere(pix_coordinate_array == i)[0]
        # Find data from that pixel
        ind = np.where(data_all[tdc].T[0] == pix)[0]
        # Cut non-valid timestamps ('-1's)
        ind = ind[np.where(data_all[tdc].T[1][ind] >= 0)[0]]
        if not np.any(ind):
            continue

        data_all[tdc].T[1][ind] = (
            data_all[tdc].T[1][ind] - data_all[tdc].T[1][ind] % 140
        ) * 2500 / 140 + cal_matrix[i, (data_all[tdc].T[1][ind] % 140)]

    return data_all


def save_offset_timestamp_differences(
    path: str,
    pixels: list,
    rewrite: bool,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int = 1000,
    delta_window: float = 50e3,
):
    """Calculate and save timestamp differences into '.csv' file.

    Unpacks data, calculates timestamp differences for the requested
    pixels, and saves them into a '.csv' table. Works with firmware
    version 2212. Calculates delta ts with TDC calibration applied
    for calculations of offset compensations.

    Parameters
    ----------
    path : str
        Path to the folder with the '.dat' data files.
    pixels : list
        List of pixel numbers for which the timestamp differences should
        be calculated and saved or list of two lists with pixel numbers
        for peak vs. peak calculations.
    rewrite : bool
        Switch for overwriting the '.csv' file if it already exists.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number, including the "#".
    firmware_version : str
        LinoSPAD2 firmware version. Versions "2212s" (skip) and "2212b"
        (block) are recognized.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default
        is 1000.
    delta_window : float, optional
        Size of a window to which timestamp differences are compared.
        Differences in that window are saved. The default is 50e3 (50 ns).

    Raises
    ------
    TypeError
        Only boolean values of 'rewrite' and string values of
        'daughterboard_number', 'motherboard_number', and
        'firmware_version' are accepted. The first error is raised so
        that the plot does not accidentally get rewritten in the case
        no clear input was given.

    Returns
    -------
    None.
    """
    # Parameter type check
    if not isinstance(pixels, list):
        raise TypeError(
            "'pixels' should be a list of integers or a list of two lists."
        )
    if not isinstance(firmware_version, str):
        raise TypeError(
            "'firmware_version' should be a string, '2212b' or '2208'."
        )
    if not isinstance(rewrite, bool):
        raise TypeError("'rewrite' should be boolean.")
    if not isinstance(daughterboard_number, str):
        raise TypeError(
            "'daughterboard_number' should be a string, either 'NL11' or 'A5'."
        )

    os.chdir(path)

    all_files = glob.glob("*.dat*")

    output_file_name = all_files[0][:-4] + "-" + all_files[-1][:-4]

    # Check if csv file exists and if it should be rewritten
    try:
        os.chdir("offset_deltas")
        if os.path.isfile(f"{output_file_name}.csv"):
            if rewrite:
                print(
                    "\n! ! ! CSV file with timestamps differences already "
                    "exists and will be overwritten ! ! !\n"
                )
                for i in range(5):
                    print(f"\n! ! ! Deleting the file in {5 - i} ! ! !\n")
                    time.sleep(1)
                os.remove(f"{output_file_name}.csv")
            else:
                sys.exit(
                    "\n CSV file already exists, 'rewrite' set to"
                    " 'False', exiting."
                )
        os.chdir("..")
    except FileNotFoundError:
        pass

    # Collect the data for the required pixels
    print(
        "\n> > > Collecting data for delta t plot for the requested "
        "pixels and saving it to .csv in a cycle < < <\n"
    )
    if firmware_version == "2212s":
        pix_coordinates = np.arange(256).reshape(4, 64).T
    elif firmware_version == "2212b":
        pix_coordinates = np.arange(256).reshape(64, 4)
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    # Mask the hot/warm pixels
    # mask = utils.apply_mask(daughterboard_number, motherboard_number)

    # Check if 'pixels' is one or two peaks, swap their positions if
    # needed
    if isinstance(pixels[0], list):
        pixels_left = pixels[0]
        pixels_right = pixels[1]
        # Check if pixels from the first list are to the left of the right
        # (peaks are not mixed up)
        if pixels_left[-1] > pixels_right[0]:
            pixels_left, pixels_right = pixels_right, pixels_left
    elif isinstance(pixels[0], int):
        pixels_left = pixels
        pixels_right = pixels

    for i in tqdm(range(ceil(len(all_files))), desc="Collecting data"):
        file = all_files[i]

        # Prepare a dictionary for output
        deltas_all = {}

        # Unpack data for the requested pixels into dictionary
        data_all = unpack_data_for_offset_calibration(
            file,
            daughterboard_number,
            motherboard_number,
            firmware_version,
            timestamps,
        )

        # Calculate and collect timestamp differences
        for q in pixels_left:
            for w in pixels_right:
                if w <= q:
                    continue
                deltas_all[f"{q},{w}"] = []
                # Find end of cycles
                cycler = np.argwhere(data_all[0].T[0] == -2)
                cycler = np.insert(cycler, 0, 0)
                # First pixel in the pair
                tdc1, pix_c1 = np.argwhere(pix_coordinates == q)[0]
                pix1 = np.where(data_all[tdc1].T[0] == pix_c1)[0]
                # Second pixel in the pair
                tdc2, pix_c2 = np.argwhere(pix_coordinates == w)[0]
                pix2 = np.where(data_all[tdc2].T[0] == pix_c2)[0]
                # Get timestamp for both pixels in the given cycle
                for cyc in range(len(cycler) - 1):
                    pix1_ = pix1[
                        np.logical_and(
                            pix1 > cycler[cyc], pix1 < cycler[cyc + 1]
                        )
                    ]
                    if not np.any(pix1_):
                        continue
                    pix2_ = pix2[
                        np.logical_and(
                            pix2 > cycler[cyc], pix2 < cycler[cyc + 1]
                        )
                    ]
                    if not np.any(pix2_):
                        continue
                    # Calculate delta t
                    tmsp1 = data_all[tdc1].T[1][
                        pix1_[np.where(data_all[tdc1].T[1][pix1_] > 0)[0]]
                    ]
                    tmsp2 = data_all[tdc2].T[1][
                        pix2_[np.where(data_all[tdc2].T[1][pix2_] > 0)[0]]
                    ]
                    for t1 in tmsp1:
                        deltas = tmsp2 - t1
                        ind = np.where(np.abs(deltas) < delta_window)[0]
                        deltas_all[f"{q},{w}"].extend(deltas[ind])

        # Save data as a .csv file in a cycle so data is not lost
        # in the case of a failure close to the end
        data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index")
        del deltas_all
        data_for_plot_df = data_for_plot_df.T
        try:
            os.chdir("offset_deltas")
        except FileNotFoundError:
            os.mkdir("offset_deltas")
            os.chdir("offset_deltas")
        csv_file = glob.glob(f"*{output_file_name}.csv*")
        if csv_file:
            data_for_plot_df.to_csv(
                f"Offset_{output_file_name}.csv",
                mode="a",
                index=False,
                header=False,
            )
        else:
            data_for_plot_df.to_csv(
                f"Offset_{output_file_name}.csv", index=False
            )
        os.chdir("..")

    if os.path.isfile(path + f"/offset_deltas/{output_file_name}.csv") is True:
        print(
            "\n> > > Timestamp differences are saved as"
            f"{output_file_name}.csv in "
            f"{os.path.join(path, 'offset_deltas')} < < <"
        )
    else:
        print("File wasn't generated. Check input parameters.")


def calculate_and_save_offset_calibration(
    path: str,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int = 1000,
):
    """Calculate offset calibration and save as .npy.

    Calculate offset calibration for all 256 pixels for the given
    motherboard-daughterboard and firmware version.

    Parameters
    ----------
    path : str
        Path to the folder with the '.dat' data files.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number, including the "#".
    firmware_version : str
        LinoSPAD2 firmware version.
    timestamps : int, optional
        Number of timestamps per cycle per TDC. The default 1000.
    """

    def gauss(x, A, x0, sigma):
        return A * np.exp(-((x - x0) ** 2) / (2 * sigma**2))

    # Calculate delta ts for pixels 0 and 4-255
    save_offset_timestamp_differences(
        path,
        pixels=[[0], [x for x in range(1, 256)]],
        rewrite=True,
        daughterboard_number=daughterboard_number,
        motherboard_number=motherboard_number,
        firmware_version=firmware_version,
        timestamps=timestamps,
    )
    os.chdir(path + r"/offset_deltas/")
    file_csv = glob.glob("*Offset_*.csv*")[0]
    dt_all = np.array(pd.read_csv(file_csv))
    os.chdir("..")

    peak_positions_3_256 = np.zeros(256)
    peak_positions_1_4 = np.zeros(256)

    # Fit to find where the peak ends up.
    for i in range(255):
        dt_nonan_arr = dt_all[:, i][~np.isnan(dt_all[:, i])]
        if dt_nonan_arr.size == 0:
            continue
        else:
            bins = np.arange(
                np.min(dt_nonan_arr), np.max(dt_nonan_arr), 2500 / 140
            )

            counts, bin_edges = np.histogram(dt_nonan_arr, bins=bins)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

            n_max = np.argmax(counts)
            arg_max = (bin_edges[n_max] + bin_edges[n_max + 1]) / 2
            sigma = 200

            parameters, covariance = curve_fit(
                gauss, bin_centers, counts, p0=[max(counts), arg_max, sigma]
            )

            peak_positions_3_256[i] = parameters[1]
            peak_positions_3_256[0:3] = 0

    # Calculate delta ts for pixels 1,2,3
    save_offset_timestamp_differences(
        path,
        pixels=[[1, 2, 3], [4]],
        rewrite=True,
        daughterboard_number=daughterboard_number,
        motherboard_number=motherboard_number,
        firmware_version=firmware_version,
        timestamps=timestamps,
    )
    os.chdir(path + r"/offset_deltas/")
    file_csv = glob.glob("*.csv*")[0]
    dt_all = np.array(pd.read_csv(file_csv))
    os.chdir("..")

    peak_positions_1_4 = np.zeros(256)

    # Fit to find where the peak ends up.
    for i in range(3):
        dt_nonan_arr = dt_all[:, i][~np.isnan(dt_all[:, i])]
        if dt_nonan_arr.size == 0:
            continue
        else:
            bins = np.arange(
                np.min(dt_nonan_arr), np.max(dt_nonan_arr), 2500 / 140
            )

            counts, bin_edges = np.histogram(dt_nonan_arr, bins=bins)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

            n_max = np.argmax(counts)
            arg_max = (bin_edges[n_max] + bin_edges[n_max + 1]) / 2
            sigma = 200

            parameters, covariance = curve_fit(
                gauss, bin_centers, counts, p0=[max(counts), arg_max, sigma]
            )

            peak_positions_1_4[i] = parameters[1]

    peak_positions = peak_positions_1_4 + peak_positions_3_256
    print(peak_positions)
    print(peak_positions)

    # Indices for a system of linear equations for offset calculation.
    # Last equation is for setting the average offset equal to zero.
    a = np.zeros((256, 256))
    for i in range(3, 255):
        a[i][0] = 1
        a[i][i + 1] = -1
    a[0][1] = 1
    a[1][2] = 1
    a[2][3] = 1
    a[0][4] = -1
    a[1][4] = -1
    a[2][4] = -1
    a[-1] = 1

    # Solving the system of equations, the result are offsets
    offsets = np.linalg.solve(a, peak_positions)

    np.save(
        f"Offset_{daughterboard_number}_{motherboard_number}"
        f"_{firmware_version}.npy",
        offsets,
    )


def load_calibration_data(
    calibration_path: str,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    include_offset: bool = False,
):
    """Load the calibration data.

    Parameters
    ----------
    calibration_path : str
        Path to the '.csv' file with the calibration matrix.
    daughterboard_number: str
        The LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number, including the "#".
    firmware_version: str
        LinoSPAD2 firmware version.
    include_offset : bool, optional
        Switch for including the offset calibration. The default is
        True.

    Returns
    -------
    data_matrix : numpy.ndarray
        256x140 matrix containing the calibrated data.
    offset_arr : numpy.ndarray, optional
        Array of 256 offset values, one for each pixel. Returned only if
        include_offset is True.
    """

    path_to_backup = os.getcwd()
    os.chdir(calibration_path)

    # Compensating for TDC nonlinearities
    try:
        file_TDC = glob.glob(f"*TDC_{daughterboard_number}_{motherboard_number}"f"_{firmware_version}*")[0]
    except IndexError as exc:
        raise FileNotFoundError(
            f"TDC calibration for {daughterboard_number}, "
            f"{motherboard_number}, and {firmware_version} is not found") from exc

    # Compensating for offset
    if include_offset:
        try:
            file_offset = glob.glob(f"*Offset_{daughterboard_number}_{motherboard_number}"f"_{firmware_version}*")[0]
        except IndexError:
            raise FileNotFoundError("No .npy file with offset calibration data was found")
        offset_arr = np.load(file_offset)

    # Skipping the first row of TDC bins' numbers
    data_matrix_TDC = np.genfromtxt(file_TDC, delimiter=",", skip_header=1)

    # Cut the first column which is pixel numbers
    data_matrix_TDC = np.delete(data_matrix_TDC, 0, axis=1)

    os.chdir(path_to_backup)

    return (data_matrix_TDC, offset_arr) if include_offset else data_matrix_TDC
