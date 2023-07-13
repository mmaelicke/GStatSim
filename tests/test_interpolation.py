import pytest

import os

import pandas as pd
import numpy as np
import random

import gstatsim as gs


# def test_ordinary_kriging():
#     pass


# def test_simple_kriging():
#     pass


def test_sequential_gaussian_simulation_ordinary_kriging():
    """
    This tests the sequential gaussian simulation with ordinary kriging.
    The test is roughly based on demos/4_Sequential_Gaussian_Simulation.ipynb

    """
    # read demo data
    data_file_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '../demos/data/greenland_test_data.csv')
    df_bed = pd.read_csv(data_file_path)

    # Grid and transform data, compute variogram parameters
    # grid data to 100 m resolution and remove coordinates with NaNs
    res = 1000
    df_grid, _, _, _ = gs.Gridding.grid_data(df_bed, 'X', 'Y', 'Bed', res)

    # remove NaNs
    df_grid = df_grid[df_grid["Z"].isnull() == False]

    # Initialize grid
    # define coordinate grid
    xmin = np.min(df_grid['X'])
    xmax = np.max(df_grid['X'])     # min and max x values

    ymin = np.min(df_grid['Y'])
    ymax = np.max(df_grid['Y'])     # min and max y values

    Pred_grid_xy = gs.Gridding.prediction_grid(xmin, xmax, ymin, ymax, res)

    # set random seeds
    # this line can be removed, if we decide to use numpy.random.shuffle instead of random.shuffle
    random.seed(42)
    np.random.seed(42)

    # pick ("random") points from grid
    index_points = np.random.choice(
        range(len(Pred_grid_xy)), size=25, replace=False)
    Pred_grid_xy = Pred_grid_xy[index_points, :]

    # Sequential Gaussian simulation
    # set variogram parameters
    azimuth = 0
    nugget = 0
    k = 48         # number of neighboring data points used to estimate a given point
    rad = 50000    # 50 km search radius

    # the major and minor ranges are the same in this example because it is isotropic
    major_range = minor_range = 31852.
    sill = 0.7
    vtype = 'Exponential'

    # save variogram parameters as a list
    vario = [azimuth, nugget, major_range, minor_range, sill, vtype]

    # ordinary kriging
    sim = gs.Interpolation.okrige_sgs(
        Pred_grid_xy, df_grid, 'X', 'Y', 'Z', k, vario, rad)

    # as we set the numpy random seed, the simulation is deterministic and we can compare to the following (rounded) results
    expected_sim = np.array([445.5, 299.5, 358.9, 395.9, 153.5,  78.3, 344.5, 226.1, 432.2,
                             377.1, 202.9, 319.3, 276.9, 296., 368.8, 405.2, 328.2, 134.6,
                             307.8, 252.3, 270.5, 117.9, 425., 411.6,  31.2])

    # assert
    np.testing.assert_array_almost_equal(sim, expected_sim, decimal=1)


def test_sequential_gaussian_simulation_simple_kriging():
    """
    This tests the sequential gaussian simulation with simple kriging.
    The test is roughly based on demos/4_Sequential_Gaussian_Simulation.ipynb

    """
    data_file_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '../demos/data/greenland_test_data.csv')
    df_bed = pd.read_csv(data_file_path)

    # Grid and transform data, compute variogram parameters
    # grid data to 100 m resolution and remove coordinates with NaNs
    res = 1000
    df_grid, grid_matrix, rows, cols = gs.Gridding.grid_data(
        df_bed, 'X', 'Y', 'Bed', res)

    # remove NaNs
    df_grid = df_grid[df_grid["Z"].isnull() == False]

    # maximum range distance
    maxlag = 50000
    # num of bins
    n_lags = 70

    # Initialize grid
    # define coordinate grid
    xmin = np.min(df_grid['X'])
    xmax = np.max(df_grid['X'])     # min and max x values

    ymin = np.min(df_grid['Y'])
    ymax = np.max(df_grid['Y'])     # min and max y values

    Pred_grid_xy = gs.Gridding.prediction_grid(xmin, xmax, ymin, ymax, res)

    # set random seed
    # this line can be removed, if we decide to use numpy.random.shuffle instead of random.shuffle
    random.seed(42)
    np.random.seed(42)

    # pick ("random") points from grid
    index_points = np.random.choice(
        range(len(Pred_grid_xy)), size=25, replace=False)
    Pred_grid_xy = Pred_grid_xy[index_points, :]

    # Sequential Gaussian simulation
    # set variogram parameters
    azimuth = 0
    nugget = 0
    k = 48         # number of neighboring data points used to estimate a given point
    rad = 50000    # 50 km search radius

    # the major and minor ranges are the same in this example because it is isotropic
    major_range = minor_range = 31852.
    sill = 0.7
    vtype = 'Exponential'

    # save variogram parameters as a list
    vario = [azimuth, nugget, major_range, minor_range, sill, vtype]

    # simple kriging
    sim = gs.Interpolation.skrige_sgs(
        Pred_grid_xy, df_grid, 'X', 'Y', 'Z', k, vario, rad)

    # as we set the numpy random seed, the simulation is deterministic and we can compare to the following (rounded) results
    expected_sim = np.array([440.3, 299.5, 360., 396.7, 152.1,  78.3, 344.5, 225.6, 368.2,
                             378.1, 205.2, 304.5, 294.2, 296.3, 368.8, 400.4, 329.9, 133.9,
                             307.4, 252.2, 270.5, 116.5, 403.3, 411.6,  29.1])

    # assert
    np.testing.assert_array_almost_equal(sim, expected_sim, decimal=1)


if __name__ == '__main__':
    import pytest
    pytest.main()
