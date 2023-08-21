"""The main hub for LinoSPAD2 data analysis."""


from functions import plot_tmsp

# ======================================================================
# Paths to where data or the 'csv' files with the resuts are located.
# ======================================================================

path_expl = ""

# ======================================================================
# Function execution.
# ======================================================================

if __name__ == "__main__":
    plot_tmsp.plot_sen_pop(
        path_expl,
        board_number="A5",
        fw_ver="2212b",
        timestamps=200,
        show_fig=True,
        app_mask=True,
    )
