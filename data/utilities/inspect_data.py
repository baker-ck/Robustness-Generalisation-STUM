"""
inspect_data.py

Basic metadata inspection utility for traffic forecasting datasets.

Supported file types:
    - .npz
    - .npy
    - .h5 
    - .pkl
    - .csv
    
Example:
    # 1. Dataset split and NPZ structure
            python inspect_data.py his.npz idx_test.npy idx_train.npy idx_val.npy 
    # 2. Traffic time-series data
            python inspect_data.py metrla_his_2012.h5 
            python inspect_data.py pemsbay_his_2017.h5 
    # 3. Road network adjacency definitions
            python inspect_data.py metrla_rn_adj.npy 
            python inspect_data.py pemsbay_rn_adj.npy 
    # 4. Serialised adjacency matrices
            python inspect_data.py adj_mx.pkl adj_mx_bay.pkl 
    # 5. Sensor metadata and road distances
            python inspect_data.py \
            graph_sensor_locations.csv \
            graph_sensor_locations_bay.csv \
            distances_la_2012.csv \
            distances_bay_2017.csv

"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd

try:
    import h5py
except ImportError:
    h5py = None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def file_size(path):

    size = os.path.getsize(path)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} PB"
    
def separator():
    print("=" * 70)

# ---------------------------------------------------------------------
# NPY
# ---------------------------------------------------------------------

def inspect_npy(path):

    arr = np.load(path)

    separator()

    print(f"File        : {os.path.basename(path)}")
    print("Type        : NumPy array (.npy)")
    print(f"Size        : {file_size(path)}")

    print(f"\nShape       : {arr.shape}")
    print(f"Dtype       : {arr.dtype}")
    print(f"Dimensions  : {arr.ndim}")
    print(f"Elements    : {arr.size}")

    # ---------------------------------------------------------
    # Array preview
    # ---------------------------------------------------------
    if arr.size > 0:
        preview = arr.flatten()[:10]
        print(f"Preview     : {preview}")

    # ---------------------------------------------------------
    # GRAPH DETECTION (adjacency matrix case)
    # ---------------------------------------------------------
    is_square = (
        arr.ndim == 2 and
        arr.shape[0] == arr.shape[1]
    )

    if is_square:

        A = arr
        n = A.shape[0]

        print("\n" + "-" * 60)
        print("Detected: SQUARE MATRIX → possible adjacency matrix")
        print("-" * 60)

        # Treat as graph (binary or weighted)
        A_bin = (A > 0)

        num_edges = np.sum(A_bin)
        density = num_edges / (n * n)
        sparsity = 1.0 - density
        
        print(f"Nodes              : {n}")
        print(f"Num edges (nonzero): {int(num_edges)}")
        print(f"Density            : {density:.6f}")
        print(f"Sparsity           : {sparsity:.6f}")

        # -----------------------------------------------------
        # Symmetry check
        # -----------------------------------------------------
        symmetric = np.allclose(A, A.T)

        print(f"\nSymmetric          : {symmetric}")
# ---------------------------------------------------------------------
# NPZ
# ---------------------------------------------------------------------

def inspect_npz(path):

    data = np.load(path)

    separator()

    print(f"File        : {os.path.basename(path)}")
    print("Type        : NumPy archive (.npz)")
    print(f"Size        : {file_size(path)}")

    print(f"\nArrays ({len(data.files)}):")

    for key in data.files:

        arr = data[key]

        print(f"\n[{key}]")
        print(f"  Shape      : {arr.shape}")
        print(f"  Dtype      : {arr.dtype}")
        print(f"  Dimensions : {arr.ndim}")
        print(f"  Elements   : {arr.size}")
        
        if arr.size > 0:
            preview = arr.reshape(-1)[:10]
            print(f"  Preview    : {preview}")
        
        if arr.ndim == 3:
            frames, sensors, features = arr.shape
            print(f"  Frames      : {frames}")
            print(f"  Sensors     : {sensors}")
            print(f"  Features    : {features}")
            print("  Interpretation: traffic tensor (time × sensors × features)")

        elif arr.ndim == 2:
            rows, cols = arr.shape
            print(f"  Rows        : {rows}")
            print(f"  Columns     : {cols}")

        elif arr.ndim == 1:
            print(f"  Length      : {arr.shape[0]}")

        elif arr.ndim == 0:
            print(f"  Scalar      : {arr.item()}")


# ---------------------------------------------------------------------
# HDF5
# ---------------------------------------------------------------------

def inspect_h5(path):
    separator()

    print(f"File        : {os.path.basename(path)}")
    print("Type        : HDF5")
    print(f"Size        : {file_size(path)}")


    # ----------------------------------------
    # Load pandas HDF5
    # ----------------------------------------

    try:

        df = pd.read_hdf(path)

    except Exception as e:

        print("Could not read pandas HDF5:")
        print(e)
        return


    print("\nDataFrame")
    print("-"*50)

    print("Shape       :", df.shape)
    print(f"Index type  : {type(df.index).__name__}")

    print("\nFirst timestamp:")
    print(df.index[0])

    print("\nLast timestamp:")
    print(df.index[-1])


    # ----------------------------------------
    # Time-series metadata
    # ----------------------------------------

    frames = len(df)

    interval = (
        df.index[1] - df.index[0]
    ).seconds / 60


    duration = (
        (df.index[-1]-df.index[0])
        .total_seconds()
        /
        (24*3600)
    )


    print("\nTime-series statistics")
    print("-"*50)

    print(f"Duration (days)          : {duration:.2f}")

    print(f"Sampling interval (min)  : {interval}")

    print(f"Number of frames         : {frames}")

    print(
        f"Time range               : "
        f"{df.index[0]} --> {df.index[-1]}"
    )


    # ----------------------------------------
    # Sensors
    # ----------------------------------------

    shape = df.shape

    if len(shape) == 2:
        sensors = shape[1]
        features = 1

    elif len(shape) == 3:
        sensors = shape[1]
        features = shape[2]

    else:
        sensors = shape[-1]
        features = "Unknown"

    print(f"Sensors                  : {sensors}")
    print(f"Features per sensor      : {features}")

# ---------------------------------------------------------------------
# Pickle
# ---------------------------------------------------------------------

def inspect_pkl(path):

    separator()

    print(f"File        : {os.path.basename(path)}")
    print("Type        : Pickle")
    print(f"Size        : {file_size(path)}")

    with open(path, "rb") as f:
        obj = pickle.load(f, encoding="bytes")


    print(f"\nPython object : {type(obj)}")


    # ---------------------------------------------------------
    # Detect METR-LA / PEMS-BAY adjacency pickle format
    # Typical structure:
    # (
    #   sensor_ids,
    #   sensor_id_to_ind,
    #   adjacency_matrix
    # )
    # ---------------------------------------------------------

    if (
        isinstance(obj, tuple)
        and len(obj) == 3
        and isinstance(obj[2], np.ndarray)
        and obj[2].ndim == 2
    ):

        print("\nDetected: Traffic graph adjacency pickle")

        print("-" * 60)

        print("\nComponent 0: Sensor IDs")
        print(f"  Type   : {type(obj[0])}")

        if isinstance(obj[0], (list, tuple)):
            print(f"  Length : {len(obj[0])}")


        print("\nComponent 1: Sensor ID mapping")
        print(f"  Type   : {type(obj[1])}")

        if isinstance(obj[1], dict):
            print(f"  Keys   : {len(obj[1])}")


        print("\nComponent 2: Adjacency matrix")

        adj = obj[2]

        print(f"  Shape  : {adj.shape}")
        print(f"  Dtype  : {adj.dtype}")

        weighted = not np.all(
            np.isin(adj, [0, 1])
        )

        if weighted:
            print("  Type   : Weighted adjacency matrix")
        else:
            print("  Type   : Binary adjacency matrix")


        return


    # ---------------------------------------------------------
    # Generic dictionary pickle
    # ---------------------------------------------------------

    if isinstance(obj, dict):

        print(f"Keys ({len(obj)}):")

        for key, value in obj.items():

            print(f"\n{key}")

            print(f"  Type : {type(value)}")


            if isinstance(value, np.ndarray):

                print(f"  Shape : {value.shape}")
                print(f"  Dtype : {value.dtype}")


            elif isinstance(value, (list, tuple)):

                print(f"  Length : {len(value)}")


    # ---------------------------------------------------------
    # Generic list / tuple pickle
    # ---------------------------------------------------------

    elif isinstance(obj, (list, tuple)):

        print(f"\nLength : {len(obj)}")


        for i, item in enumerate(obj):

            print(f"\nItem {i}")

            print(f"  Type : {type(item)}")


            if isinstance(item, np.ndarray):

                print(f"  Shape : {item.shape}")
                print(f"  Dtype : {item.dtype}")


            elif isinstance(item, dict):

                print(f"  Keys : {len(item)}")


            elif isinstance(item, (list, tuple)):

                print(f"  Length : {len(item)}")



# ---------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------

def inspect_csv(path):

    import pandas as pd

    separator()

    print(f"File        : {os.path.basename(path)}")
    print("Type        : CSV")
    print(f"Size        : {file_size(path)}")


    try:

        df = pd.read_csv(path)

    except Exception as e:

        print("Could not read CSV:")
        print(e)
        return


    print("\nDataFrame")
    print("-" * 60)

    print(f"Rows        : {df.shape[0]}")
    print(f"Columns     : {df.shape[1]}")


    print("\nColumns:")

    for col in df.columns:
        print(f"  - {col}")


    print("\nDtype:")
    
    for col, dtype in df.dtypes.items():
        print(f"  {col:<20} : {dtype}")


    print("\nPreview:")
    print(df.head())


    # ---------------------------------------------------------
    # Traffic dataset format detection
    # ---------------------------------------------------------

    columns = set(df.columns)


    if (
        {"sensor_id", "latitude", "longitude"}
        .issubset(columns)
        or
        {"sensor_id", "lat", "lon"}
        .issubset(columns)
    ):

        print("\nDetected: Sensor location file")


        print(f"Sensors     : {len(df)}")


    elif (
        {"from", "to", "distance"}
        .issubset(columns)
        or
        {"from_id", "to_id", "distance"}
        .issubset(columns)
    ):

        print("\nDetected: Distance / edge list file")


        print(f"Edges       : {len(df)}")

# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------

def inspect(path):

    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        inspect_npy(path)

    elif ext == ".npz":
        inspect_npz(path)

    elif ext in (".h5", ".hdf5"):
        inspect_h5(path)

    elif ext == ".pkl":
        inspect_pkl(path)

    elif ext == ".csv":
        inspect_csv(path)
        
    else:
        print(f"Unsupported file type: {path}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Inspect dataset files."
    )

    parser.add_argument(
        "files",
        nargs="+",
        help="Files to inspect"
    )

    args = parser.parse_args()

    for file in args.files:

        if not os.path.exists(file):

            print(f"{file} does not exist.")

            continue

        inspect(file)


if __name__ == "__main__":
    main()
