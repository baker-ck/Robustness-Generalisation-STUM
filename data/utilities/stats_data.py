"""
stats_data.py

Basic statistical analysis utility for traffic forecasting datasets.

Supported file types:
    - .npz
    - .npy
    - .h5 
    - .pkl
    - .csv
    
Example:
    # 1. Dataset split and NPZ structure
            python stats_data.py his.npz idx_test.npy idx_train.npy idx_val.npy
    # 2. Traffic time-series data
            python stats_data.py metrla_his_2012.h5 pemsbay_his_2017.h5 
    # 3. Road network adjacency definitions
            python stats_data.py metrla_rn_adj.py 
            python stats_data.py pemsbay_rn_adj.py
    # 4. Serialised adjacency matrices
            python stats_data.py adj_mx.pkl adj_mx_bay.pkl 
    # 5. Sensor metadata and road distances
            python stats_data.py \
            graph_sensor_locations.csv \
            graph_sensor_locations_bay.csv \
            distances_la_2012.csv \
            distances_bay_2012.csv
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
# Numerical statistics helper
# ---------------------------------------------------------------------

def analyse_values(arr):

    arr = np.asarray(arr)

    print("\nStatistics")
    print("-" * 40)

    print(f"Min        : {np.nanmin(arr):.4f}")
    print(f"Max        : {np.nanmax(arr):.4f}")
    print(f"Mean       : {np.nanmean(arr):.4f}")
    print(f"Std        : {np.nanstd(arr):.4f}")
    print(f"Median     : {np.nanmedian(arr):.4f}")

    print(f"25%        : {np.nanpercentile(arr,25):.4f}")
    print(f"75%        : {np.nanpercentile(arr,75):.4f}")

    missing = np.isnan(arr).sum()

    print(f"Missing    : {missing}")

# ---------------------------------------------------------------------
# NPY
# ---------------------------------------------------------------------

def analyse_npy(path):

    arr = np.load(path)

    print(f"\n{'='*60}")
    print(f"File: {path}")
    print(f"Type: NPY")
    print(f"Size        : {file_size(path)}")

    print("\nArray structure")
    print("-" * 60)

    print(f"Shape      : {arr.shape}")
    print(f"Dtype      : {arr.dtype}")
    print(f"Dimensions : {arr.ndim}")


    # -----------------------------------------------------
    # General value statistics
    # -----------------------------------------------------

    if np.issubdtype(arr.dtype, np.number):

        print("\nValue statistics")
        print("-" * 60)

        print(f"Min        : {np.min(arr)}")
        print(f"Max        : {np.max(arr)}")
        print(f"Mean       : {np.mean(arr):.4f}")
        print(f"Std        : {np.std(arr):.4f}")

        print(f"Median     : {np.median(arr):.4f}")
        print(f"25%        : {np.percentile(arr, 25):.4f}")
        print(f"75%        : {np.percentile(arr, 75):.4f}")

    # -----------------------------------------------------
    # Feature-wise statistics
    # Assumes: (frames, sensors, features)
    # -----------------------------------------------------

    if arr.ndim == 3:

        frames, sensors, features = arr.shape

        print("\nTraffic tensor structure")
        print("-" * 60)

        print(f"Frames     : {frames}")
        print(f"Sensors    : {sensors}")
        print(f"Features   : {features}")


        print("\nFeature-wise statistics")
        print("-" * 60)


        for f in range(features):

            feature_data = arr[:, :, f]

            print(f"\nFeature {f}")

            print(f"  Min      : {np.min(feature_data):.4f}")
            print(f"  Max      : {np.max(feature_data):.4f}")
            print(f"  Mean     : {np.mean(feature_data):.4f}")
            print(f"  Std      : {np.std(feature_data):.4f}")
            print(f"  Median   : {np.median(feature_data):.4f}")
            print(f"  25%      : {np.percentile(feature_data,25):.4f}")
            print(f"  75%      : {np.percentile(feature_data,75):.4f}")
            
    # -----------------------------------------------------
    # Graph statistics (adjacency matrix)
    # -----------------------------------------------------

    is_square = (
        arr.ndim == 2 and
        arr.shape[0] == arr.shape[1]
    )

    if is_square:

        print("\nGraph statistics")
        print("-" * 60)

        A_bin = arr > 0

        n = arr.shape[0]

        degrees = np.sum(A_bin, axis=1)


        print(f"Nodes      : {n}")
        print(f"Edges      : {int(np.sum(A_bin))}")
        print(f"Density    : {np.sum(A_bin)/(n*n):.6f}")

        print("\nDegree distribution")

        print(f"Mean       : {degrees.mean():.2f}")
        print(f"Median     : {np.median(degrees):.2f}")
        print(f"25%        : {np.percentile(degrees,25):.2f}")
        print(f"75%        : {np.percentile(degrees,75):.2f}")
        print(f"Std        : {degrees.std():.2f}")

# ---------------------------------------------------------------------
# NPZ
# ---------------------------------------------------------------------

def analyse_npz(path):

    data = np.load(path)

    print(f"\n{'='*60}")
    print(f"File: {path}")
    print(f"Type: NPZ")
    print(f"Size: {file_size(path)}")
    print(f"Arrays: {list(data.keys())}")


    for key in data.files:

        arr = data[key]

        print(f"\n[{key}]")

        print("\nStructure")
        print("-" * 40)

        print(f"Shape      : {arr.shape}")
        print(f"Dtype      : {arr.dtype}")
        print(f"Dimensions : {arr.ndim}")
        
        # ---------------------------------------------------------
        # Skip statistics for index arrays
        # ---------------------------------------------------------

        if "idx" in key.lower():

            print("\nIndex array detected")
            print("Skipping value statistics")
            continue

        # ---------------------------------------------------------
        # Numerical statistics
        # ---------------------------------------------------------

        if np.issubdtype(arr.dtype, np.number):

            print("\nStatistics")
            print("-" * 40)

            print(f"Min        : {np.min(arr):.4f}")
            print(f"Max        : {np.max(arr):.4f}")
            print(f"Mean       : {np.mean(arr):.4f}")
            print(f"Std        : {np.std(arr):.4f}")
            print(f"Median     : {np.median(arr):.4f}")
            print(f"25%        : {np.percentile(arr,25):.4f}")
            print(f"75%        : {np.percentile(arr,75):.4f}")

            missing = np.isnan(arr).sum()

            print(f"Missing    : {missing}")

        # ---------------------------------------------------------
        # Traffic tensor feature statistics
        # Assumes (frames, sensors, features)
        # ---------------------------------------------------------

        if arr.ndim == 3:

            frames, sensors, features = arr.shape

            print("\nTraffic tensor")
            print("-" * 40)

            print(f"Frames     : {frames}")
            print(f"Sensors    : {sensors}")
            print(f"Features   : {features}")
           
            if features == 3:
                print("Feature 0: normalized traffic value")
                print("Feature 1: time-of-day encoding")
                print("Feature 2: day-of-week encoding")

            print("\nFeature statistics")
            print("-" * 40)


            for f in range(features):

                feature = arr[:, :, f]

                print(f"\nFeature {f}")

                print(f"  Mean     : {np.mean(feature):.4f}")
                print(f"  Std      : {np.std(feature):.4f}")
                print(f"  Min      : {np.min(feature):.4f}")
                print(f"  Max      : {np.max(feature):.4f}")
# ---------------------------------------------------------------------
# HDF5
# ---------------------------------------------------------------------

def analyse_h5(path):

    import pandas as pd

    print(f"\n{'='*60}")
    print(f"File: {path}")
    print(f"Type: HDF5")
    print(f"Size: {file_size(path)}")


    # ---------------------------------------------------------
    # Pandas HDF5 (METR-LA / PEMS-BAY format)
    # ---------------------------------------------------------

    try:

        df = pd.read_hdf(path)

        values = df.values.astype(float)

        print("\nPandas HDF5")
        print("-" * 40)

        print(f"Shape      : {df.shape}")
        print(f"Sensors    : {df.shape[1]}")
        print(f"Frames     : {df.shape[0]}")

        # ---------------------------------------------------------
        # Time index information
        # ---------------------------------------------------------
        print("\nTime index")
        print("-" * 40)

        print(f"Index type : {type(df.index).__name__}")
        if isinstance(df.index, pd.DatetimeIndex):

            print("\nTime index")
            print("-" * 40)

            print(f"Start      : {df.index[0]}")
            print(f"End        : {df.index[-1]}")

            frequency = df.index.to_series().diff().mode()

            if len(frequency) > 0:
                print(f"Frequency  : {frequency.iloc[0]}")
            else:
                print("Frequency  : Unknown")
        else:
            print("No datetime index detected")

        analyse_values(values)

        return


    except Exception:

        pass


    # ---------------------------------------------------------
    # Raw HDF5 datasets
    # ---------------------------------------------------------

    if h5py is None:

        print("h5py not installed.")
        return


    with h5py.File(path, "r") as f:

        print("\nDatasets")
        print("-" * 40)


        def visitor(name, obj):

            if isinstance(obj, h5py.Dataset):

                print(f"\nDataset: {name}")

                arr = obj[:]

                print(f"Shape      : {arr.shape}")
                print(f"Dtype      : {arr.dtype}")


                analyse_values(arr)


        f.visititems(visitor)
# ---------------------------------------------------------------------
# Pickle
# ---------------------------------------------------------------------

def analyse_pkl(path):

    print(f"\n{'='*60}")
    print(f"File: {path}")
    print(f"Type: Pickle (.pkl)")
    print(f"Size: {file_size(path)}")


    with open(path, "rb") as f:
        obj = pickle.load(f, encoding="bytes")


    print(f"Python type: {type(obj)}")


    # ---------------------------------------------------------
    # Traffic adjacency pickle
    # METR-LA / PEMS-BAY format:
    # (sensor_ids, sensor_id_to_ind, adjacency_matrix)
    # ---------------------------------------------------------

    if (
        isinstance(obj, tuple)
        and len(obj) == 3
        and isinstance(obj[2], np.ndarray)
    ):

        adj = obj[2]


        print("\nDetected: Traffic adjacency matrix")

        print("-" * 40)

        print(f"Nodes      : {adj.shape[0]}")
        print(f"Shape      : {adj.shape}")
        print(f"Dtype      : {adj.dtype}")


        # -------------------------------------------------
        # Graph statistics
        # -------------------------------------------------

        A_bin = adj > 0

        degrees = np.sum(A_bin, axis=1)

        edges = np.sum(A_bin)

        density = edges / (adj.shape[0] ** 2)


        print("\nGraph statistics")
        print("-" * 40)

        print(f"Edges      : {int(edges)}")
        print(f"Density    : {density:.6f}")

        print("\nDegree distribution")

        print(f"Mean       : {degrees.mean():.2f}")
        print(f"Median     : {np.median(degrees):.2f}")
        print(f"Std        : {degrees.std():.2f}")
        print(f"Min        : {degrees.min():.2f}")
        print(f"Max        : {degrees.max():.2f}")
        print(f"25%        : {np.percentile(degrees,25):.2f}")
        print(f"75%        : {np.percentile(degrees,75):.2f}")


        weighted = not np.all(np.isin(adj, [0,1]))

        print(f"\nWeighted   : {weighted}")


        return



    # ---------------------------------------------------------
    # Generic numpy pickle
    # ---------------------------------------------------------

    if isinstance(obj, np.ndarray):

        print("\nArray statistics")
        print("-" * 40)

        print(f"Shape      : {obj.shape}")
        print(f"Dtype      : {obj.dtype}")

        print(f"Min        : {np.min(obj)}")
        print(f"Max        : {np.max(obj)}")
        print(f"Mean       : {np.mean(obj):.4f}")
        print(f"Std        : {np.std(obj):.4f}")


    else:

        print("\nNo statistical analysis available.")
        print("Object structure handled by inspect_data.py")

# ---------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------

def analyse_csv(path):

    import pandas as pd

    print(f"\n{'='*60}")
    print(f"File: {path}")
    print(f"Type: CSV")
    print(f"Size: {file_size(path)}")


    try:

        df = pd.read_csv(path)

    except Exception as e:

        print("Could not read CSV:")
        print(e)
        return


    print("\nShape")
    print("-" * 40)

    print(f"Rows       : {df.shape[0]}")
    print(f"Columns    : {df.shape[1]}")


    columns = set(df.columns)


    # ---------------------------------------------------------
    # Sensor locations
    # ---------------------------------------------------------

    if (
        {"latitude", "longitude"}.issubset(columns)
        or
        {"lat", "lon"}.issubset(columns)
    ):

        print("\nDetected: Sensor locations")

        print("-" * 40)

        print(f"Sensors    : {len(df)}")


        if "latitude" in df.columns:

            lat = df["latitude"]
            lon = df["longitude"]

        else:

            lat = df["lat"]
            lon = df["lon"]


        print("\nGeographic statistics")

        print(f"Latitude")
        print(f"  Min     : {lat.min():.6f}")
        print(f"  Max     : {lat.max():.6f}")
        print(f"  Mean    : {lat.mean():.6f}")


        print(f"\nLongitude")
        print(f"  Min     : {lon.min():.6f}")
        print(f"  Max     : {lon.max():.6f}")
        print(f"  Mean    : {lon.mean():.6f}")


    # ---------------------------------------------------------
    # Distance / edge list
    # ---------------------------------------------------------

    elif (
        "distance" in columns
        or
        {"from", "to"}.issubset(columns)
    ):

        print("\nDetected: Distance edge list")

        print("-" * 40)


        print(f"Edges      : {len(df)}")


        if "distance" in df.columns:

            dist = df["distance"]


            print("\nDistance statistics")

            print(f"Min        : {dist.min():.4f}")
            print(f"Max        : {dist.max():.4f}")
            print(f"Mean       : {dist.mean():.4f}")
            print(f"Median     : {dist.median():.4f}")
            print(f"Std        : {dist.std():.4f}")

            print(f"25%        : {dist.quantile(0.25):.4f}")
            print(f"75%        : {dist.quantile(0.75):.4f}")


    # ---------------------------------------------------------
    # Generic CSV numerical statistics
    # ---------------------------------------------------------

    else:

        print("\nGeneric CSV numerical statistics")

        print("-" * 40)

        numeric = df.select_dtypes(include=np.number)

        if numeric.shape[1] > 0:

            print(numeric.describe())

        else:

            print("No numerical columns detected.")

# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------

def inspect(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        analyse_npy(path)

    elif ext == ".npz":
        analyse_npz(path)

    elif ext in [".h5", ".hdf5"]:
        analyse_h5(path)

    elif ext == ".pkl":
        analyse_pkl(path)
        
    elif ext == ".csv":
        analyse_csv(path)

    else:
        print(f"Unsupported file: {path}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Statistical analysis of dataset files."
    )

    parser.add_argument(
        "files",
        nargs="+",
        help="Files to analyse"
    )

    args = parser.parse_args()

    for file in args.files:

        if not os.path.exists(file):

            print(f"{file} does not exist.")

            continue

        inspect(file)

if __name__ == "__main__":
    main()
