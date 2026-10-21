"""Shared pytest fixtures for the other particle filter unit tests.

ParticleFilter.__init__ waits on the ROS map_server service and initializes TF
listeners which stalls pytest. These fixtures are pseudo-classes and instantiate
the filter via __new__ (to bypass __init__) and populate only the attributes required
by the math and update methods. Using the __new__ method and certain ROS objects
were ideas from an LLM.
"""

import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

import pf as particle_filter		# cant use pf since it is name of variable in pytest
from helper_functions import TFHelper
from rclpy.logging import get_logger

package_dir = Path(__file__).parent.parent		# package root for loading map files

# manually copied from declare_parameter() in pf.py, so ensure they match
DEFAULTS = {
    "n_particles": 1200,
    "d_thresh": 0.2,
    "a_thresh": math.pi / 6,
    "random_resample": 0.15,
    "resample_xy_sigma": 0.1,
    "resample_theta_sigma": 0.1,
    "sigma_hit": 0.2,
    "min_likelihood": 0.01,
    "beam_step": 2,
    "init_xy_sigma": 0.3,
    "init_theta_sigma": 0.3,
    "odom_trans_noise": 0.1,
    "odom_rot_noise": 0.1,
    "cluster_radius": 0.5,
    "log_scale": 10.0,
}


def room_grid():
    """Creates a 3 m x 2 m occupancy grid with an asymmetric obstacle.

    Returns:
        np.ndarray: A 60 x 40 integer array indexed as [x, y] at 0.05 m/cell,
            where 100 = occupied wall cell and 0 = free space.
    """
    grid = np.zeros((60, 40), dtype=int)		# 3.0 m x 2.0 m at 0.05 m resolution (from yaml)
    grid[0, :] = grid[-1, :] = 100		# left and right boundary walls
    grid[:, 0] = grid[:, -1] = 100		# bottom and top boundary walls

    # add off-center box so the room is not symmetric under 180 deg turns
    grid[44:51, 8:15] = 100		# x from [0.70, 1.05] m, y from [-0.60, -0.25] m
    return grid


def fake_occupancy_field(grid, resolution=0.05, origin=(-1.5, -1.0)):
    """Wraps a 2D grid array into a mock OccupancyField object.

    Uses SimpleNamespace ("dummy" object supporting dot-access like
    obj.attr) so pf.py can access .map.info just like the real
    ROS message without actually running ROS.

    Args:
        grid (np.ndarray): Occupancy values indexed as [x, y].
        resolution (float): Grid cell size in m (default = 0.05).
        origin (tuple[float, float]): World (x, y) coords of the bottom-left
            grid cell. Default = (-1.5, -1.0) to center the room map on (0, 0).

    Returns:
        SimpleNamespace: A mock OccupancyField containing `map` (mimicking a
            nav_msgs/OccupancyGrid message) and `map_array` indexed by [y, x].
    """
    width, height = grid.shape

    # transpose from [x, y] to [y, x] and flatten to 1D list to match OccupancyGrid.data
    data = grid.T.flatten().tolist()

    # This formatting was LLM assisted.
    # nest SimpleNamespaces to fake the ROS nav_msgs/MapMetaData dot-attribute hierarchy
    info = SimpleNamespace(resolution=resolution, width=width, height=height,
                           origin=SimpleNamespace(position=SimpleNamespace(x=origin[0], y=origin[1])))

    # occupancy_field.py stores map_array as [y, x] floats, whereas input grid is [x, y]
    return SimpleNamespace(map=SimpleNamespace(data=data, info=info),
                           map_array=grid.T.astype(float))


def load_map_file(yaml_name):
    """Loads a ROS map YAML and image file into a mock OccupancyField.

    Args:
        yaml_name (str): Filename of the map config inside `maps/` (e.g., "gauntlet.yaml").

    Returns:
        SimpleNamespace: A mock OccupancyField populated with the map data.
    """
    yaml_file = package_dir / "maps" / yaml_name
    cfg = yaml.safe_load(open(yaml_file))
    img = np.array(Image.open(yaml_file.parent / cfg["image"]))

    # convert pixel brightness (0 = black/wall, 255 = white/free) to occupancy prob [0, 1]
    p_occupied = (255 - img) / 255.0

    # threshold probabilities into ROS occupancy values: 100 = wall, 0 = free, -1 = unknown
    grid = np.where(p_occupied > cfg["occupied_thresh"], 100,
                    np.where(p_occupied < cfg["free_thresh"], 0, -1))

    # image row 0 is top-left, but map y=0 is bottom-left: flip vertically, then transpose [y, x] to [x, y]
    grid = np.flipud(grid).T
    return fake_occupancy_field(grid, cfg["resolution"], cfg["origin"][:2])


@pytest.fixture(autouse=True)
def seed():
    """Seeds random number generators before each test for deterministic execution.

    `autouse=True` tells pytest to run this automatically before every test
    without needing to pass `seed` as a test function argument.
    """
    np.random.seed(0)		# seed numpy rng used for gaussian noise in pf.py
    random.seed(0)		# seed python rng used for random cell picks in random_particle


@pytest.fixture
def make_pf():
    """Provides a factory function to instantiate ParticleFilter without ROS nodes.

    Because pytest fixtures cannot take custom arguments directly from a test,
    this fixture returns an inner `make()` builder function that tests can call
    with custom settings (e.g., `pf = make_pf(n_particles=5)`).

    Returns:
        Callable: A builder function accepting an optional `occupancy_field` and
            keyword overrides for any parameter in `DEFAULTS`.
    """
    def make(occupancy_field=None, **settings):
        # __new__ allocates the ParticleFilter instance in memory without calling __init__
        pf = particle_filter.ParticleFilter.__new__(particle_filter.ParticleFilter)

        # merge DEFAULTS dict with any **settings overrides and attach each as an attribute on pf
        for name, value in {**DEFAULTS, **settings}.items():
            setattr(pf, name, value)

        pf.map_frame = "map"
        pf.get_logger = lambda: get_logger("pf")		# no ROS node here, so give pf.py's debug logs a plain rclpy logger
        pf.occupancy_field = occupancy_field or fake_occupancy_field(room_grid())

        # allocate TFHelper via __new__ as well so we can call its angle/pose math methods without ROS tf
        pf.transform_helper = TFHelper.__new__(TFHelper)

        pf.laser_offset = (0.0, 0.0)		# laser at robot center by default (overridden in offset test)
        pf.particle_cloud = []
        pf.particle_weights = []

        # initialize odom tracking state at the origin (0, 0, 0)
        pf.current_odom_xy_theta = (0.0, 0.0, 0.0)
        pf.odom_pose = particle_filter.Particle(0.0, 0.0, 0.0).as_pose()

        # fake ROS publisher: calling pf.particle_pub.publish(msg) appends msg to pf.published list
        pf.published = []
        pf.particle_pub = SimpleNamespace(publish=pf.published.append)

        # precompute likelihood distance field last since it reads pf.occupancy_field and pf.sigma_hit
        pf.build_likelihood_field()
        return pf
    return make


@pytest.fixture
def pf(make_pf):
    """Provides a default ParticleFilter instance configured with the room map.

    Args:
        make_pf (Callable): The particle filter factory fixture injected by pytest.

    Returns:
        particle_filter.ParticleFilter: A configured filter instance using `DEFAULTS`.
    """
    return make_pf()


@pytest.fixture
def make_map():
    """Exposes the `fake_occupancy_field` helper as a pytest fixture.

    Returns:
        Callable: The `fake_occupancy_field` function for building custom test maps.
    """
    return fake_occupancy_field


@pytest.fixture(scope="session")
def gauntlet_map():
    """Loads the gauntlet map once per test session.

    `scope="session"` caches the loaded map across all tests in the pytest run
    so the PNG image only has to be parsed once.

    Returns:
        SimpleNamespace: The mock OccupancyField for `gauntlet.yaml`.
    """
    return load_map_file("gauntlet.yaml")


@pytest.fixture
def simulate_scan():
    """Provides a raycasting function to simulate noise-free laser range scans.

    Returns:
        Callable: A `simulate(pf, pose, theta, max_range)` function that returns
            an array of simulated beam distances in meters.
    """
    def simulate(pf, pose, theta, max_range=5.0):		# 5.0 m = approx physical neato lidar max range
        """Casts rays from a robot pose to compute distances to the nearest walls.

        Args:
            pf (particle_filter.ParticleFilter): Filter instance containing the map.
            pose (tuple[float, float, float]): Robot (x (m), y (m), heading (rad)) pose in map frame.
            theta (Sequence[float]): Beam angles in rad relative to robot heading.
            max_range (float): Max raycast distance in m (default = 5.0).

        Returns:
            np.ndarray: Array of distances to the first wall hit in m (`np.inf` = no hit).
        """
        x, y, heading = pose

        # 2D rotation matrix to transform laser_offset from robot base_link frame to global map frame
        laser_x = x + pf.laser_offset[0] * math.cos(heading) - pf.laser_offset[1] * math.sin(heading)
        laser_y = y + pf.laser_offset[0] * math.sin(heading) + pf.laser_offset[1] * math.cos(heading)

        # march along ray in 1/4 cell increments so diagonal rays cant skip diagonally across thin walls
        step = pf.map_resolution / 4
        distances = np.arange(step, max_range, step)

        ranges = np.full(len(theta), np.inf)		# default to inf (no hit) to match real lidar behavior
        for k in range(len(theta)):
            # compute (x, y) coords for all sample steps along beam k at global angle (heading + theta[k])
            xs = laser_x + distances * math.cos(heading + theta[k])
            ys = laser_y + distances * math.sin(heading + theta[k])
            ix, iy, on_map = pf.world_to_cell(xs, ys)

            # only index map_array where on_map is True so negative out-of-bounds indices dont wrap around in numpy
            hit = on_map.copy()
            hit[on_map] = pf.occupancy_field.map_array[iy[on_map], ix[on_map]] > 0		# > 0 hits walls (100), ignores free (0) and unknown (-1)
            if hit.any():
                ranges[k] = distances[np.argmax(hit)]		# np.argmax on a boolean array returns the index of the first True
        return ranges
    return simulate