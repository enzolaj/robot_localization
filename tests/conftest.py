""" setup shared by all the test files. pytest loads this file automatically, and anything marked
    @pytest.fixture in here can be used by any test just by putting its name in the test's arguments.
    The reason for this architecture is that we cannot make a ParticleFilter instance, as it hangs
    on a map_server call which never happens in our testing design. Thus, we are creating an object
    and bypassing the __init__ portion via __new__. This architecture was influenced by an LLM.
"""

import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

import pf as particle_filter     # 'pf' on its own is already the name of our test filter
from helper_functions import TFHelper

package_dir = Path(__file__).parent.parent

# the default settings from the declare_parameter calls in pf.py. __init__ never runs in the tests,
# so the test filters get their settings from here, which is just a mirror (dont forget to update!!!)
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
}


def room_grid():
    """ our main test map: a 3 m by 2 m room, 60 by 40 cells at 5 cm each, with walls all the way
        around and a box near one corner. the grid is indexed grid[x cell, y cell], with 100 for a wall
        and 0 for open floor.

        we added the box because an empty rectangle looks exactly the same after turning it 180
        degrees. the laser would have no way to tell those two poses apart, and the laser tests could
        end up passing or failing because of that instead of because of our code. """
    grid = np.zeros((60, 40), dtype=int)
    grid[0, :] = grid[-1, :] = 100     # left and right walls
    grid[:, 0] = grid[:, -1] = 100     # bottom and top walls
    grid[44:51, 8:15] = 100            # the box, covering about x = 0.7 to 1.05 m and y = -0.6 to -0.25 m
    return grid


def fake_occupancy_field(grid, resolution=0.05, origin=(-1.5, -1.0)):
    """ dresses up a grid so the filter can use it in place of the real OccupancyField. the filter only
        ever reads two things from it: .map, which looks like the OccupancyGrid message map_server
        sends, and .map_array, the 2d copy my partner added.

        SimpleNamespace is an empty object that we can stick attributes on, which is enough to fake a
        message as long as the filter only reads from it. """
    width, height = grid.shape
    # the real message stores the map as one long list. it starts at the bottom left corner and goes
    # along x first, finishing one row of y before starting the next. flipping our grid to [y, x] and
    # flattening it gives that same order.
    data = grid.T.flatten().tolist()
    info = SimpleNamespace(resolution=resolution, width=width, height=height,
                           origin=SimpleNamespace(position=SimpleNamespace(x=origin[0], y=origin[1])))
    # occupancy_field.py fills map_array as [y, x], so it gets the flipped grid too
    return SimpleNamespace(map=SimpleNamespace(data=data, info=info),
                           map_array=grid.T.astype(float))


def load_map_file(yaml_name):
    """ loads one of the real maps from the maps folder, using the same rules map_server uses to turn
        the image into walls and open space """
    yaml_file = package_dir / "maps" / yaml_name
    cfg = yaml.safe_load(open(yaml_file))
    img = np.array(Image.open(yaml_file.parent / cfg["image"]))

    # black pixels are walls and white pixels are open floor. map_server turns each pixel into a chance
    # of being a wall (0 for white, 1 for black), then uses the two thresholds from the yaml to decide.
    # anything in between the thresholds is marked unknown (-1).
    p_occupied = (255 - img) / 255.0
    grid = np.where(p_occupied > cfg["occupied_thresh"], 100,
                    np.where(p_occupied < cfg["free_thresh"], 0, -1))

    # the first row of an image is its top edge, but the map starts from the bottom left corner, so we
    # flip the image upside down. then .T swaps it from [row, column] to [x, y].
    grid = np.flipud(grid).T
    return fake_occupancy_field(grid, cfg["resolution"], cfg["origin"][:2])


@pytest.fixture(autouse=True)
def seed():
    # autouse means this runs before every test without being asked for. the filter uses random numbers
    # everywhere: np.random for noise and python's random module for picking map cells. seeding both
    # means a test gets the same random numbers every run, so it can't pass one time and fail the next.
    np.random.seed(0)
    random.seed(0)


@pytest.fixture
def make_pf():
    """ gives tests a function for building a filter. make_pf() makes one on the room map with the default
        settings. make_pf(n_particles=5) changes a setting, and make_pf(occupancy_field=...) uses a
        different map. """
    def make(occupancy_field=None, **settings):
        # make the object without running __init__ (see the top of this file for why)
        pf = particle_filter.ParticleFilter.__new__(particle_filter.ParticleFilter)

        # start from the defaults, then apply whatever the test asked to change
        for name, value in {**DEFAULTS, **settings}.items():
            setattr(pf, name, value)

        pf.map_frame = "map"
        pf.occupancy_field = occupancy_field or fake_occupancy_field(room_grid())

        # TFHelper has the same problem as the filter: its __init__ sets up tf. but the parts we use,
        # like angle_diff and angle_normalize, are just math, so skipping __init__ is fine here too.
        pf.transform_helper = TFHelper.__new__(TFHelper)

        # put the laser right at the center of the robot, which keeps most tests simple. the one test
        # that checks the real 8.4 cm offset sets it itself.
        pf.laser_offset = (0.0, 0.0)

        pf.particle_cloud = []
        pf.particle_weights = []
        pf.current_odom_xy_theta = (0.0, 0.0, 0.0)
        pf.odom_pose = particle_filter.Particle(0.0, 0.0, 0.0).as_pose()

        # instead of sending messages anywhere, the fake publisher adds them to a list so a test can
        # look at what would have gone out
        pf.published = []
        pf.particle_pub = SimpleNamespace(publish=pf.published.append)

        # __init__ normally does this step, so we do it here
        pf.build_likelihood_field()
        return pf
    return make


@pytest.fixture
def pf(make_pf):
    """ a filter on the room map with default settings, for tests that don't need anything special """
    return make_pf()


@pytest.fixture
def make_map():
    """ for tests that want to build their own small map: make_map(grid, resolution, origin) """
    return fake_occupancy_field


@pytest.fixture(scope="session")
def gauntlet_map():
    # the real gauntlet map. it's big, so scope="session" loads it once for the whole test run
    # instead of once for every test that uses it.
    return load_map_file("gauntlet.yaml")


@pytest.fixture
def simulate_scan():
    """ gives tests a fake laser. simulate(pf, pose, theta) returns the ranges a perfect laser would
        measure from that pose, one for each beam angle in theta. this lets us test the scoring with
        scans where we know exactly where the robot was. """
    def simulate(pf, pose, theta, max_range=5.0):
        x, y, heading = pose

        # the scan comes from the laser, which sits at laser_offset on the robot, so first work out
        # where the laser is on the map. this is the same rotate-then-shift math the filter uses.
        laser_x = x + pf.laser_offset[0] * math.cos(heading) - pf.laser_offset[1] * math.sin(heading)
        laser_y = y + pf.laser_offset[0] * math.sin(heading) + pf.laser_offset[1] * math.cos(heading)

        # walk outward along each beam in small steps (a quarter of a cell) until we land in a wall cell
        step = pf.map_resolution / 4
        distances = np.arange(step, max_range, step)

        # a beam that never hits anything comes back as inf, like the real laser does
        ranges = np.full(len(theta), np.inf)
        for k in range(len(theta)):
            xs = laser_x + distances * math.cos(heading + theta[k])
            ys = laser_y + distances * math.sin(heading + theta[k])
            ix, iy, on_map = pf.world_to_cell(xs, ys)

            # mark which steps landed on a wall. map_array is [y, x], and we only check steps that are
            # on the map, since the ones off the edge don't have a cell to look up.
            hit = on_map.copy()
            hit[on_map] = pf.occupancy_field.map_array[iy[on_map], ix[on_map]] > 0
            if hit.any():
                # argmax on a list of True/False gives the index of the first True, the first wall we reach
                ranges[k] = distances[np.argmax(hit)]
        return ranges
    return simulate
