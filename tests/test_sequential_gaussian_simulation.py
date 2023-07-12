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

    # as we set the numpy random seed, the simulation is deterministic and we can compare to the following results
    expected_sim = np.array([445.58862124, 299.53765396, 359.56605613, 396.60995493,
                             153.28838087,  77.63036298, 344.5631518, 225.55553587,
                             432.87925129, 378.95730331, 203.21548867, 317.62793126,
                             277.2812756, 295.98086447, 368.84047619, 405.14835289,
                             327.96207952, 135.65011045, 308.16606146, 254.00633239,
                             270.50381579, 116.882082, 424.80502096, 411.70562776,
                             31.52758769])

    # assert
    assert sim == pytest.approx(expected_sim, rel=1e-3, abs=1e-3)

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

    # as we set the numpy random seed, the simulation is deterministic and we can compare to the following results
    expected_sim = np.array([439.90931728, 299.53765396, 360.67854707, 397.45580118,
                             151.97394082,  77.62306358, 344.62284859, 225.00439516,
                             368.96339368, 379.78384266, 205.48563175, 302.83678308,
                             294.59412781, 296.26297957, 368.84047619, 400.34741858,
                             329.67033291, 134.90363369, 307.79843452, 253.96837702,
                             270.50381579, 115.42910899, 403.11529701, 411.69057941,
                             29.43989149])

    # assert
    #assert sim == pytest.approx(expected_sim, rel=1e-3, abs=1e-3)
    np.testing.assert_allclose(sim, expected_sim, rtol=1e-5, atol=1e-5)

    return True


def test_sequential_gaussian_simulation():
    """
    Execute tests of sequential gaussian simulation.

    """
    assert check_sequential_gaussian_simulation_ordinary_kriging()
    assert check_sequential_gaussian_simulation_simple_kriging()
