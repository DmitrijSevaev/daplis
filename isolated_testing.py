import glob
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from src.daplis.functions import calc_diff as cd, delta_t
from src.daplis.functions import utils
from src.daplis.functions.calibrate import load_calibration_data
from pyarrow import feather as ft


@dataclass
class DataParamsConfig:
    """Configuration parameters for timestamp differences calculation.

    Parameters
    ----------
    pixels : list
        List of pixel numbers for which the timestamp differences should
        be calculated and saved or list of two lists with pixel numbers
        for peak vs. peak calculations.
    path : str
        Path to data files.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number, including the "#".
    motherboard_number1 : str
        First LinoSPAD2 motherboard (FPGA) number, including the "#".
        Used for full sensor data analysis.
    motherboard_number2 : str
        Second LinoSPAD2 motherboard (FPGA) number. Used for full sensor
        data analysis.
    firmware_version : str
        LinoSPAD2 firmware version. Versions "2212s" (skip) and "2212b"
        (block) are recognized.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default
        is 512.
    delta_window : float, optional
        Size of a window to which timestamp differences are compared.
        Differences in that window are saved. The default is 50e3 (50 ns).
    app_mask : bool, optional
        Switch for applying the mask for hot pixels. The default is True.
    include_offset : bool, optional
        Switch for applying offset calibration. The default is True.
    apply_calibration : bool, optional
        Switch for applying TDC and offset calibration. If set to 'True'
        while apply_offset_calibration is set to 'False', only the TDC
        calibration is applied. The default is True.
    absolute_timestamps : bool, optional
        Indicator for data with absolute timestamps. The default is
        False.
    """

    pixels: list
    path: str = ""
    daughterboard_number: str = ""
    motherboard_number: str = ""
    motherboard_number1: str = ""
    motherboard_number2: str = ""
    firmware_version: str = ""
    timestamps: int = 512
    delta_window: float = 50e3
    app_mask: bool = True
    include_offset: bool = True
    apply_calibration: bool = True
    absolute_timestamps: bool = False


def unpack_binary_data(
        file: str,
        daughterboard_number: str,
        motherboard_number: str,
        firmware_version: str,
        calibration_matrix,
        offset_array,
        timestamps: int = 512,
        include_offset: bool = False,
        apply_calibration: bool = True,
) -> np.ndarray:
    """Unpacks binary-encoded data from LinoSPAD2 firmware version 2212.

    Parameters
    ----------
    file : str
        Path to the binary data file.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number, including the "#".
    firmware_version : str
        LinoSPAD2 firmware version. Either '2212s' (skip) or '2212b' (block).
    timestamps : int, optional
        Number of timestamps per cycle per TDC per acquisition cycle.
        The default is 512.
    include_offset : bool, optional
        Switch for applying offset calibration. The default is True.
    apply_calibration : bool, optional
        Switch for applying TDC and offset calibration. If set to 'True'
        while include_offset is set to 'False', only the TDC
        calibration is applied. The default is True.

    Returns
    -------
    data_all : array-like
        3D array of pixel coordinates in the TDC and the timestamps.

    Raises
    ------
    TypeError
        If 'daughterboard_number', 'motherboard_number', or 'firmware_version'
        parameters are not of string type.
    FileNotFoundError
        If no calibration data file is found.

    Notes
    -----
    The returned data is a 3D array where rows represent TDC numbers,
    columns represent the data, and each cell contains a pixel number in
    the TDC (from 0 to 3) and the timestamp recorded by that pixel.
    """
    # Parameter type check
    if not isinstance(daughterboard_number, str): raise TypeError("'daughterboard_number' should be a string.")
    if not isinstance(motherboard_number, str): raise TypeError("'motherboard_number' should be a string.")
    if not isinstance(firmware_version, str): raise TypeError("'firmware_version' should be a string.")

    # Unpack binary data
    raw_data = np.memmap(file, dtype=np.uint32)
    # Timestamps are stored in the lower 28 bits
    data_timestamps = (raw_data & 0xFFFFFFF).astype(np.int64)
    # Pixel address in the given TDC is 2 bits above timestamp
    data_pixels = ((raw_data >> 28) & 0x3).astype(np.int8)
    # Check the top bit, assign '-1' to invalid timestamps
    data_timestamps[raw_data < 0x80000000] = -1

    # Number of acquisition cycles in each data file
    cycles = len(data_timestamps) // (timestamps * 65)
    # Transform into a matrix of size 65 by cycles*timestamps
    data_pixels = (
        data_pixels.reshape(cycles, 65, timestamps)
        .transpose((1, 0, 2))
        .reshape(65, -1)
    )

    data_timestamps = (
        data_timestamps.reshape(cycles, 65, timestamps)
        .transpose((1, 0, 2))
        .reshape(65, -1)
    )

    # Cut the 65th TDC that does not hold any actual data from pixels
    data_pixels = data_pixels[:-1]
    data_timestamps = data_timestamps[:-1]

    # Insert '-2' at the end of each cycle
    insert_indices = np.linspace(
        timestamps, cycles * timestamps, cycles
    ).astype(np.int64)

    data_pixels = np.insert(
        data_pixels,
        insert_indices,
        -2,
        1,
    )
    data_timestamps = np.insert(
        data_timestamps,
        insert_indices,
        -2,
        1,
    )

    # Combine both matrices into a single one, where each cell holds pixel
    # coordinates in the TDC and the timestamp
    data_all = np.stack((data_pixels, data_timestamps), axis=2).astype(
        np.int64
    )

    if apply_calibration is False:
        data_all[:, :, 1] = data_all[:, :, 1] * 2500 / 140
    else:
        # Path to the calibration data
        pix_coordinates = np.arange(256).reshape(64, 4)
        for i in range(256):
            # Transform pixel number to TDC number and pixel coordinates in that TDC (from 0 to 3)
            tdc, pix = np.argwhere(pix_coordinates == i)[0]
            # Find data from that pixel
            ind = np.where(data_all[tdc].T[0] == pix)[0]
            # Cut non-valid timestamps ('-1's)
            ind = ind[data_all[tdc].T[1][ind] >= 0]
            if not np.any(ind):
                continue
            data_cut = data_all[tdc].T[1][ind]
            # Apply calibration; offset is added due to how delta ts are calculated
            if include_offset:
                data_all[tdc].T[1][ind] = (
                        (data_cut - data_cut % 140) * 2500 / 140
                        + calibration_matrix[i, (data_cut % 140)]
                        + offset_array[i])
            else:
                data_all[tdc].T[1][ind] = (data_cut - data_cut % 140) * 2500 / 140 + calibration_matrix[
                    i, (data_cut % 140)]

    return data_all


def _calculate_timestamps_differences(
        files,
        data_params,
        path,
        pix_coor,
        pixels,
        calibration_matrix,
        offset_array,
        daughterboard_number,
        motherboard_number,
        firmware_version
):
    for file in files:
        # unpack data from binary files
        data_all = unpack_binary_data(
            file,
            daughterboard_number,
            motherboard_number,
            firmware_version,
            calibration_matrix,
            offset_array,
            data_params.timestamps,
            data_params.include_offset,
            data_params.apply_calibration
        )

        # calculate the differences and convert them to a pandas dataframe
        deltas_all = cd.calculate_differences_2212_fast(data_all, pixels, pix_coor)
        data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index").T

        # save the data to a feather file
        file_name = os.path.basename(file)

        output_file = os.path.join(path, str(file_name.replace(".dat", ".feather")))
        data_for_plot_df.reset_index(drop=True, inplace=True)
        ft.write_feather(data_for_plot_df, output_file)


def calculate_and_save_timestamp_differences_mp(
        path: str,
        pixels: list,
        rewrite: bool,
        daughterboard_number: str,
        motherboard_number: str,
        firmware_version: str,
        timestamps: int,
        delta_window: float = 50e3,
        app_mask: bool = True,
        include_offset: bool = False,
        apply_calibration: bool = True,
        number_of_cores: int = 4,
) -> None:
    """Unpack data and collect timestamps differences using all CPU cores.

    Unpack data files and collect timestamps differences using all
    available CPU cores to speed up the process, while saving the results
    to a single Feather file on the go.

    Parameters
    ----------
    path : str
        Path to data files.
    pixels : list
        List of pixel numbers for which the timestamp differences should
        be calculated and saved or list of two lists with pixel numbers
        for peak vs. peak calculations.
    rewrite : bool
        Switch for rewriting the '.feather' file if it already exists.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number, including the "#".
    firmware_version: str
        LinoSPAD2 firmware version. Versions "2212s" (skip) and "2212b"
        (block) are recognized.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel.
    delta_window : float, optional
        Size of a window to which timestamp differences are compared.
        Differences in that window are saved. The default is 50e3 (50 ns).
    app_mask : bool, optional
        Switch for applying the mask for hot pixels. The default is True.
    include_offset : bool, optional
        Switch for applying offset calibration. The default is True.
    apply_calibration : bool, optional
        Switch for applying TDC and offset calibration. If set to 'True'
        while apply_offset_calibration is set to 'False', only the TDC
        calibration is applied. The default is True.
    chunksize : int, optional
        Number of files processed in each iteration. The default is 20.
    number_of_cores : int, optional
        Number of cores to use for multiprocessing. The default is 4.
    maxtasksperchild : int, optional
        Number of tasks per core. The default is 1000.
    Raises
    ------
    TypeError
        Only boolean values of 'rewrite' and string values of
        'daughterboard_number', 'motherboard_number', and 'firmware_version'
        are accepted. The first error is raised so that the plot does not
        accidentally get rewritten in the case no clear input was given.

    Returns
    -------
    None.
    """
    # parameter type check
    if isinstance(pixels, list) is False:
        raise TypeError("'pixels' should be a list of integers or a list of two lists")
    if isinstance(firmware_version, str) is False:
        raise TypeError("'firmware_version' should be string, '2212s', '2212b' or '2208'")
    if isinstance(rewrite, bool) is False:
        raise TypeError("'rewrite' should be boolean")
    if isinstance(daughterboard_number, str) is False:
        raise TypeError("'daughterboard_number' should be string")

    # check the firmware version and set the pixel coordinates accordingly
    if firmware_version == "2212s":
        pix_coor = np.arange(256).reshape(4, 64).T
    elif firmware_version == "2212b":
        pix_coor = np.arange(256).reshape(64, 4)
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    # Generate a dataclass object
    data_params = DataParamsConfig(
        pixels=pixels,
        daughterboard_number=daughterboard_number,
        motherboard_number=motherboard_number,
        firmware_version=firmware_version,
        timestamps=timestamps,
        delta_window=delta_window,
        app_mask=app_mask,
        include_offset=include_offset,
        apply_calibration=apply_calibration,
        absolute_timestamps=True,
    )

    calibration_matrix, offset_array = None, None  # initialize calibration matrix and offset array in case they are used

    # Load calibration data if necessary
    if data_params.apply_calibration:
        path_calibration_data = os.path.join(path, '..', 'src/daplis/params/calibration_data')
        calibration_data = load_calibration_data(
            path_calibration_data,
            daughterboard_number,
            motherboard_number,
            firmware_version,
            include_offset,
        )
        if include_offset:
            calibration_matrix, offset_array = calibration_data
        else:
            calibration_matrix = calibration_data

    # Apply mask if necessary
    if data_params.app_mask:
        mask = utils.apply_mask(data_params.daughterboard_number, data_params.motherboard_number)
        if isinstance(data_params.pixels[0], int) and isinstance(data_params.pixels[1], int):
            pixels = [pix for pix in data_params.pixels if pix not in mask]
        else:
            pixels = [pix for pix in data_params.pixels[0] if pix not in mask]
            pixels.extend(pix for pix in data_params.pixels[1] if pix not in mask)

    os.chdir(path)

    # Find all LinoSPAD2 data files
    files = glob.glob(os.path.join(path, "*.dat*"))
    num_of_files = len(files)

    # split files into chunks of equal length (files / number_of_cores)
    files = np.array_split(files, number_of_cores)
    output_directory = os.path.join(path, 'delta_ts_data_mp')

    print("Starting analysis of the files")
    start_time = time.time()

    processes = []
    # Create processes (number of cores) and assign to each process its specified files chunk (files/number_of_cores)
    # each process will run the _calculate_timestamps_differences function with its own parameters target: the function to be run
    for i in range(number_of_cores):
        p = multiprocessing.Process(
            target=_calculate_timestamps_differences,
            args=(
                files[i],
                data_params,
                output_directory,
                pix_coor,
                pixels,
                calibration_matrix,
                offset_array,
                daughterboard_number,
                motherboard_number,
                firmware_version
            ),
        )
        p.start()
        processes.append(p)  # add the process to the list so we can wait for all of them to finish

    # wait for all the processes to finish, and only then continue to the next step
    for process in processes: process.join()

    end_time = time.time()

    output_string = f"Parallel processing of {num_of_files} files (with each writing to its file) finished in: {round(end_time - start_time, 2)} s"
    print(output_string)


def parallel(path: str, num_of_cores):
    calculate_and_save_timestamp_differences_mp(
        path,
        pixels=[144, 171],
        rewrite=True,
        daughterboard_number="NL11",
        motherboard_number="#33",
        firmware_version="2212b",
        timestamps=300,
        include_offset=False,
        number_of_cores=num_of_cores,
    )


def sequential(path: str):
    start = time.time()
    delta_t.calculate_and_save_timestamp_differences_fast(
        path,
        pixels=[144, 171],
        rewrite=True,
        daughterboard_number="NL11",
        motherboard_number="#33",
        firmware_version="2212b",
        timestamps=300,
        include_offset=False,
    )
    finish = time.time()
    print(f"{finish - start} s")


def _merge_files(path: str):
    # Find all .feather files in the directory
    feather_files = [path + "/" + f for f in os.listdir(path) if f.endswith(".feather")]

    # Use pandas to concatenate all feather files into a single dataframe
    dfs = [pd.read_feather(file) for file in feather_files]
    merged_df = pd.concat(dfs, ignore_index=True)

    # Save the merged dataframe back to a single feather file with
    output_file_path = path + "/mp.feather"
    merged_df.to_feather(output_file_path)


def rename_seq_result(path: str):
    # find the only existing file
    feather_files = [path + "/" + f for f in os.listdir(path) if f.endswith(".feather")]
    # rename the file to seq.feather
    os.rename(feather_files[0], path + "/seq.feather")


def _delete_results(path: str):
    # Find all .feather files in the directory
    feather_files = [path + "/" + f for f in os.listdir(path) if f.endswith(".feather")]

    # Delete all the feather files
    for file in feather_files:
        os.remove(file)

    print("Deleted all the feather files")


def compare_results(file1, file2):
    data1 = ft.read_feather(file1).dropna()
    data2 = ft.read_feather(file2).dropna()

    plt.figure()
    plt.hist(data1, bins=100)
    plt.figure()
    plt.hist(data2, bins=100)
    plt.show()


if __name__ == "__main__":
    current_directory = Path(__file__).parent
    path = str(current_directory / 'isolated_data')

    seq_path = os.path.join(path, 'delta_ts_data')
    mp_path = os.path.join(path, 'delta_ts_data_mp')

    _delete_results(seq_path)
    _delete_results(mp_path)

    sequential(path)
    parallel(path, 7)

    rename_seq_result(seq_path)
    _merge_files(mp_path)

    seq_file = seq_path + "/seq.feather"
    mp_file = mp_path + "/mp.feather"

    compare_results(seq_file, mp_file)
