import pytest

import os

import pandas as pd
import numpy as np

import gstatsim as gs


def check_sequential_gaussian_simulation_ordinary_kriging() -> bool:
    """
    This tests the sequential gaussian simulation with ordinary kriging.
    The test is roughly based on demos/4_Sequential_Gaussian_Simulation.ipynb

    """
    # read demo data
    data_file_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), './data/greenland_test_data.csv')
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

    # set random seed
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
    expected_sim = np.array([439.9, 299.5, 360.7, 397.5, 152.,  77.6, 344.6, 225., 369.,
                             379.8, 205.5, 302.8, 294.6, 296.3, 368.8, 400.3, 329.7, 134.9,
                             307.8, 254., 270.5, 115.4, 403.1, 411.7,  29.4])

    # assert
    np.testing.assert_allclose(sim, expected_sim, atol=0.1)

    return True


def check_sequential_gaussian_simulation_simple_kriging() -> bool:
    """
    This tests the sequential gaussian simulation with simple kriging.
    The test is roughly based on demos/4_Sequential_Gaussian_Simulation.ipynb

    """
    data_file_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), './data/greenland_test_data.csv')
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
    expected_sim = np.array([439.9, 299.5, 360.7, 397.5, 152.,  77.6, 344.6, 225., 369.,
                             379.8, 205.5, 302.8, 294.6, 296.3, 368.8, 400.3, 329.7, 134.9,
                             307.8, 254., 270.5, 115.4, 403.1, 411.7,  29.4])

    # assert
    np.testing.assert_allclose(sim, expected_sim, atol=0.1)

    return True


def test_sequential_gaussian_simulation():
    """
    Execute tests of sequential gaussian simulation.

    """
    assert check_sequential_gaussian_simulation_ordinary_kriging()
    assert check_sequential_gaussian_simulation_simple_kriging()
