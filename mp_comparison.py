import glob
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from src.daplis.functions import calc_diff as cd
from src.daplis.functions import utils
from src.daplis.functions.calibrate import load_calibration_data
from numpy import ndarray
from pyarrow import feather as ft


class MpWizard:

    # Initialize by passing the input parameters which later will be
    # passed into all internal functions
    def __init__(
            self,
            path: str = "",
            pixels: list = [],
            daughterboard_number: str = "",
            motherboard_number: str = "",
            firmware_version: str = "",
            timestamps: int = 512,
            delta_window: float = 50e3,
            include_offset: bool = False,
            apply_calibration: bool = True,
            apply_mask: bool = True,
            absolute_timestamps: bool = False,
            number_of_cores: int = 1,
    ):
        """Initialization of the class.

        Set all the input parameters for later use with internal class
        functions. Additionally, preload the calibration matrix.

        Parameters
        ----------
        path : str, optional
            _description_, by default ""
        pixels : list, optional
            _description_, by default []
        daughterboard_number : str, optional
            _description_, by default ""
        motherboard_number : str, optional
            _description_, by default ""
        firmware_version : str, optional
            _description_, by default ""
        timestamps : int, optional
            _description_, by default 512
        delta_window : float, optional
            _description_, by default 50e3
        include_offset : bool, optional
            _description_, by default False
        apply_calibration : bool, optional
            _description_, by default True
        apply_mask : bool, optional
            _description_, by default True
        absolute_timestamps : bool, optional
            _description_, by default False
        number_of_cores : int, optional
            _description_, by default 1
        """

        self.path = path
        self.pixels = pixels
        self.daughterboard_number = daughterboard_number
        self.motherboard_number = motherboard_number
        self.firmware_version = firmware_version
        self.timestamps = timestamps
        self.delta_window = delta_window
        self.include_offset = include_offset
        self.apply_calibration = apply_calibration
        self.apply_mask = apply_mask
        self.absolute_timestamps = absolute_timestamps
        self.number_of_cores = number_of_cores

        os.chdir(self.path)

        # Load calibration if requested
        if self.apply_calibration:

            # work_dir = Path(__file__).resolve().parent.parent

            # TODO
            # path_calibration_data = os.path.join(
            #     work_dir, r"params\calibration_data"
            # )
            path_calibration_data = (
                r"C:\Users\fintv\Desktop\CAPADS\Daplis\daplis\src\daplis\params\calibration_data"
            )
            # path_calibration_data = r"C:\Users\bruce\Documents\GitHub\daplis\src\daplis\params\calibration_data"

            calibration_data = load_calibration_data(
                path_calibration_data,
                daughterboard_number,
                motherboard_number,
                firmware_version,
                include_offset,
            )

            if self.include_offset:
                self.calibration_matrix, self.offset_array = calibration_data
            else:
                self.calibration_matrix = calibration_data

        # Apply mask if requested
        if self.apply_mask:
            mask = utils.apply_mask(
                self.daughterboard_number,
                self.motherboard_number,
            )
            if isinstance(self.pixels[0], int) and isinstance(
                    self.pixels[1], int
            ):
                self.pixels = [pix for pix in self.pixels if pix not in mask]
            else:
                self.pixels = [
                    [value for value in sublist if value not in mask]
                    for sublist in pixels
                ]

        # Check the firmware version and set the pixel coordinates accordingly
        if self.firmware_version == "2212s":
            self.pix_coor = np.arange(256).reshape(4, 64).T
        elif firmware_version == "2212b":
            self.pix_coor = np.arange(256).reshape(64, 4)
        else:
            print("\nFirmware version is not recognized.")
            sys.exit()

    def _unpack_binary_data(
            self,
            file: str,
    ) -> np.ndarray:
        """Unpack binary data from LinoSPAD2.

        Same unpacking function as the standard 'daplis' one, except
        the calibration matrix is preloaded during initialization and
        called here. Return a 3D matrix of pixel numbers and timestamps.

        Parameters
        ----------
        file : str
            A '.dat' data file from LinoSPAD2 with the binary-encoded
            data.

        Returns
        -------
        np.ndarray
            A 3D matrix of the pixel numbers where timestamp was
            recorded and timestamps themselves.
        """
        # Unpack binary data
        raw_data = np.memmap(file, dtype=np.uint32)
        # Timestamps are stored in the lower 28 bits
        data_timestamps = (raw_data & 0xFFFFFFF).astype(np.int64)
        # Pixel address in the given TDC is 2 bits above timestamp
        data_pixels = ((raw_data >> 28) & 0x3).astype(np.int8)
        # Check the top bit, assign '-1' to invalid timestamps
        data_timestamps[raw_data < 0x80000000] = -1

        # Number of acquisition cycles in each data file
        cycles = len(data_timestamps) // (self.timestamps * 65)
        # Transform into a matrix of size 65 by cycles*timestamps
        data_pixels = (
            data_pixels.reshape(cycles, 65, self.timestamps)
            .transpose((1, 0, 2))
            .reshape(65, -1)
        )

        data_timestamps = (
            data_timestamps.reshape(cycles, 65, self.timestamps)
            .transpose((1, 0, 2))
            .reshape(65, -1)
        )

        # Cut the 65th TDC that does not hold any actual data from pixels
        data_pixels = data_pixels[:-1]
        data_timestamps = data_timestamps[:-1]

        # Insert '-2' at the end of each cycle
        insert_indices = np.linspace(
            self.timestamps, cycles * self.timestamps, cycles
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

        if self.apply_calibration is False:
            data_all[:, :, 1] = data_all[:, :, 1] * 2500 / 140
        else:
            # Path to the calibration data
            pix_coordinates = np.arange(256).reshape(64, 4)
            for i in range(256):
                # Transform pixel number to TDC number and pixel
                # coordinates in that TDC (from 0 to 3)
                tdc, pix = np.argwhere(pix_coordinates == i)[0]
                # Find data from that pixel
                ind = np.where(data_all[tdc].T[0] == pix)[0]
                # Cut non-valid timestamps ('-1's)
                ind = ind[data_all[tdc].T[1][ind] >= 0]
                if not np.any(ind):
                    continue
                data_cut = data_all[tdc].T[1][ind]
                # Apply calibration; offset is added due to how delta
                # ts are calculated
                if self.include_offset:
                    data_all[tdc].T[1][ind] = (
                            (data_cut - data_cut % 140) * 2500 / 140
                            + self.calibration_matrix[i, (data_cut % 140)]
                            + self.offset_array[i]
                    )
                else:
                    data_all[tdc].T[1][ind] = (
                                                      data_cut - data_cut % 140
                                              ) * 2500 / 140 + self.calibration_matrix[
                                                  i, (data_cut % 140)
                                              ]

        return data_all

    def _calculate_differences_2212_fast(
            self,
            data: ndarray,
            delta_window: float = 50e3,
            cycle_length: float = 4e9,
    ):
        """Calculate timestamp differences for firmware version 2212.

        Calculate timestamp differences for the given pixels and LinoSPAD2
        firmware version 2212. Modified compared to the standard 'daplis'
        one. Modifications are for working with prechunked data, i.e.,
        sliced down to two arrays from the whole 64xN matrix.

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

        deltas_all = []

        data_pix_1 = data[0]
        data_pix_2 = data[1]

        indices1 = data_pix_1[0]
        indices2 = data_pix_2[0]
        timestamps1 = data_pix_1[1]
        timestamps2 = data_pix_2[1]

        timestamps_1 = []
        timestamps_2 = []

        # Go over cycles, shifting the timestamps from each next
        # cycle by lengths of cycles before (e.g., for the 4th cycle
        # add 12 ms)
        for i, _ in enumerate(self.cycle_ends[:-1]):
            slice_from = self.cycle_ends[i]
            slice_to = self.cycle_ends[i + 1]
            pix1_slice = indices1[
                (indices1 >= slice_from) & (indices1 < slice_to)
                ]
            if not np.any(pix1_slice):
                continue
            pix2_slice = indices2[
                (indices2 >= slice_from) & (indices2 < slice_to)
                ]
            if not np.any(pix2_slice):
                continue

            # Shift timestamps by cycle length
            tmsp1 = timestamps1[np.isin(indices1, pix1_slice)]
            tmsp1 = tmsp1[tmsp1 > 0]
            tmsp1 = tmsp1 + cycle_length * i

            tmsp2 = timestamps2[np.isin(indices2, pix2_slice)]
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
        df1 = pd.DataFrame(pix1_data.T, columns=["Pixel_index", "Timestamp"])
        df2 = pd.DataFrame(pix2_data.T, columns=["Pixel_index", "Timestamp"])

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
        filtered_df = df_combined[abs(df_combined["Pixel_index_diff"]) == 1]

        # Save only timestamps differences in the requested window
        delta_ts = filtered_df[
            abs(filtered_df["Timestamp_diff"]) < delta_window
            ]["Timestamp_diff"].values

        deltas_all.extend(delta_ts)

        return deltas_all

    def _calculate_timestamps_differences(self, args):
        """Calculate photon coincidences and save to '.feather'."""
        file, data, pixel_pair = args
        try:
            # Check if the 'delta_ts_data' folder exists
            output_dir = Path(self.path) / "delta_ts_data_mp"
            output_dir.mkdir(exist_ok=True)

            # Calculate the differences and convert them to a pandas
            # dataframe - pre-chunking the data down to 2 arrays
            deltas_all = self._calculate_differences_2212_fast(data)

            data_for_plot_df = pd.DataFrame(deltas_all, columns=[f"{pixel_pair[0]},{pixel_pair[1]}"]).T

            # Save the data to a '.feather' file
            file_name = Path(file).stem
            output_file = (output_dir / f"{file_name}_{pixel_pair[0]}_{pixel_pair[1]}.feather")
            ft.write_feather(data_for_plot_df.reset_index(drop=True), output_file)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    def _combine_feather_files(self, path_to_feather_files: str):

        os.chdir(path_to_feather_files)

        for pixel_pair in self.pixel_pairs:
            ft_files = glob.glob(f"*{pixel_pair[0]}_{pixel_pair[1]}*.feather")
            data_all = pd.DataFrame()
            for ft_file in ft_files:
                data = ft.read_feather(ft_file)
                data_all = pd.concat((data_all, data), ignore_index=True)
            data_all.to_feather(
                f"combined_{pixel_pair[0]}_{pixel_pair[1]}.feather"
            )

    def _chunk_that_data(self, data, pixel_pair, cycle_length: float = 4e9):
        """Chunk data down to 2 rows - prepare for child processes.

        Parameters
        ----------
        data : np.ndarray
            The whole matrix.
        pixel_pair : list
            Pair of pixels to slice out of the whole matrix.
        cycle_length : float, optional
            Cycle length in ps, by default 4e9

        Returns
        -------
        tuple
            Tuple of two arrays, each contains timestamps from the two
            pixels requested and the indices of the timestamps in the
            original matrix of data. The indices are used for correct
            assigning to the corresponding acquisition cycles.
        """

        tdc1, pix_c1 = np.argwhere(self.pix_coor == pixel_pair[0])[0]
        indices1 = np.where(data[tdc1].T[0] == pix_c1)[0]

        # Second pixel in the pair
        tdc2, pix_c2 = np.argwhere(self.pix_coor == pixel_pair[1])[0]
        indices2 = np.where(data[tdc2].T[0] == pix_c2)[0]

        data_cut_1 = np.array((indices1, data[tdc1].T[1][indices1]))
        data_cut_2 = np.array((indices2, data[tdc2].T[1][indices2]))

        return (data_cut_1, data_cut_2)

    def calculate_and_save_timestamp_differences_mp(self):
        """Optimized parallelized function that processes each file once and parallelizes pixel pair computations."""

        # Find all LinoSPAD2 data files
        files = glob.glob("*.dat")
        if not files:
            raise ValueError("No .dat files found in the specified path.")

        # Generate all pixel pairs once
        self.pixel_pairs = [[i, j] for i in self.pixels[0] for j in self.pixels[1]]

        start_time = time.time()

        # Use a single process pool for efficiency
        with multiprocessing.Pool(processes=self.number_of_cores) as pool:

            # Go file by file
            for file in files:
                # Unpack the data from the file (DO THIS ONCE PER FILE)
                data = self._unpack_binary_data(file)

                # Pre-collect the indices of the acquisition cycles' ends
                self.cycle_ends = np.argwhere(data[0].T[0] == -2)
                self.cycle_ends = np.insert(self.cycle_ends, 0, 0)

                # Prepare args for multiprocessing: only pixel pairs change
                args = [(file, self._chunk_that_data(data, pixel_pair), pixel_pair) for pixel_pair in self.pixel_pairs]

                # Process pixel pairs in parallel for this file
                pool.map(self._calculate_timestamps_differences, args)

                print(f"Processed {file} in parallel")

        end_time = time.time()
        print(f"Parallel processing of {len(files)} files finished in: {round(end_time - start_time, 2)} s")

        # Combine the resulting '.feather' files
        path_to_feathers = os.path.join(self.path, "delta_ts_data_mp")
        self._combine_feather_files(path_to_feathers)

        print(
            f"The feather files with the timestamp differences were combined into 'combined.feather' in {path_to_feathers}")


def mp():
    path = r"C:\Users\fintv\Desktop\CAPADS\Daplis\daplis\isolated_data"

    mp = MpWizard(
        path,
        pixels=[[144], [171, 172]],
        #pixels=[[x for x in range(55, 60)], [x for x in range(175, 180)]],
        # pixels=[[x for x in range(20, 80)], [x for x in range(130, 190)]],
        daughterboard_number="B7d",
        motherboard_number="#28",
        firmware_version="2212b",
        timestamps=500,
        number_of_cores=5,
    )

    mp.calculate_and_save_timestamp_differences_mp()


def seq():
    ### Standard approach - for control and comparison

    import time

    from src.daplis.functions import delta_t

    time_start = time.time()

    path = r"C:\Users\fintv\Desktop\CAPADS\Daplis\daplis\isolated_data"

    delta_t.calculate_and_save_timestamp_differences_fast(
        path,
        rewrite=True,
        #pixels=[144, 171],
        pixels=[[x for x in range(55, 60)], [x for x in range(175, 180)]],
        # pixels=[[x for x in range(20, 80)], [x for x in range(130, 190)]],
        daughterboard_number="NL11",
        motherboard_number="#33",
        firmware_version="2212b",
        timestamps=300,
        include_offset=False,
        # daughterboard_number="B7d",
        # motherboard_number="#28",
        # firmware_version="2212b",
        # timestamps=500,
    )

    print(f"Finished in {time.time() - time_start}")


if __name__ == "__main__":
    #mp()
    seq()

