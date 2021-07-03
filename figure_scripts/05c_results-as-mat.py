import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd


def update_dict_from_df(save_dict, df):
    n, m = save_dict["rho"].shape

    df.sort_values("rowid", inplace=True)

    for c in df.columns:
        if c == "rowid":
            continue

        save_dict.update({c: df[c].values.reshape((n, m), order="C")})


if __name__ == "__main__":

    args_file, convert_file = sys.argv[1:]
    args_file = Path(args_file)
    convert_file = Path(convert_file)

    assert os.path.splitext(args_file)[1] == ".npz"

    convert_base, ext = os.path.splitext(convert_file)
    assert ext == ".csv"
    results_df = pd.read_csv(convert_file)

    save_dict = dict(np.load(args_file))  # dict is required to modify
    update_dict_from_df(save_dict, results_df)

    out_filename = convert_base + ".npz"
    print("Writing to: ", out_filename)
    np.savez(out_filename, **save_dict)
