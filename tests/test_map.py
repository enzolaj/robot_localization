"""Unit tests for the map and the starting particle cloud.

Covers building the likelihood field, converting map coordinates into grid cells, finding free
space, and spreading out the first particles with and without a pose guess.
"""

import math

import numpy as np

import pf as particle_filter		# cant use pf since it is name of variable in pytest


def test_likelihood_field_is_indexed_x_then_y(make_pf, make_map):
    """Checks that likelihood_field is indexed [x, y] and is_free agrees on where walls are.

    Uses a map wider than it is tall (10 x 6 cells) with a single wall cell at x = 7, y = 2. If x and
    y were swapped anywhere, the field would come out as shape (6, 10) or its peak would move.
    """
    grid = np.zeros((10, 6), dtype=int)
    grid[7, 2] = 100
    pf = make_pf(occupancy_field=make_map(grid, resolution=0.1, origin=(0.0, 0.0)))

    assert pf.likelihood_field.shape == (10, 6)
    # argmax gives a flat index, unravel_index turns it back into (x, y)
    assert np.unravel_index(np.argmax(pf.likelihood_field), (10, 6)) == (7, 2)

    # is_free reads map_array, which is indexed [y, x] instead. with 0.1 m cells, the wall
    # covers x from 0.7 to 0.8 m and y from 0.2 to 0.3 m
    assert not pf.is_free(0.75, 0.25)
    assert pf.is_free(0.25, 0.45)


def test_likelihood_field_is_high_at_walls_and_low_in_the_open(pf):
    """Checks that the field peaks at 1.0 on walls and falls off with distance from them."""
    # walls are scaled so a typical wall is 1.0 and thicker ones are capped there
    assert np.max(pf.likelihood_field) == 1.0

    # the left wall is only one cell thick, so it comes out a bit under 1.0 (about 0.94)
    ix, iy, _ = pf.world_to_cell(-1.48, 0.0)
    assert pf.likelihood_field[ix, iy] > 0.1

    # middle of the room, at least 0.5 m from every wall
    ix, iy, _ = pf.world_to_cell(-0.5, 0.0)
    assert pf.likelihood_field[ix, iy] < 0.01

    # walking away from the left wall, values should only go down
    values = []
    for x in [-1.48, -1.4, -1.3, -1.2]:
        ix, iy, _ = pf.world_to_cell(x, 0.0)
        values.append(pf.likelihood_field[ix, iy])
    assert values == sorted(values, reverse=True)


def test_world_to_cell(pf):
    """Checks world_to_cell on the room map's corners and edges, for single points and arrays.

    The room map's bottom-left corner is at (-1.5, -1.0) [m], with 0.05 m cells, 60 x 40 cells.
    """
    assert pf.world_to_cell(-1.5, -1.0) == (0, 0, True)		# the corner itself
    assert pf.world_to_cell(-1.46, -0.96) == (0, 0, True)		# anywhere inside a cell gives that cell
    assert pf.world_to_cell(1.49, 0.99) == (59, 39, True)		# far corner is the last cell
    assert not pf.world_to_cell(-1.51, 0.0)[2]		# just left of the map
    assert not pf.world_to_cell(1.5, 0.0)[2]		# the right edge belongs to no cell

    # update_particles_with_laser passes whole arrays at once
    ix, iy, on_map = pf.world_to_cell(np.array([-1.5, 0.0, 9.0]), np.array([-1.0, 0.0, 9.0]))
    assert list(ix[:2]) == [0, 30] and list(iy[:2]) == [0, 20]
    assert list(on_map) == [True, True, False]


def test_random_particles_land_in_free_space_all_over_the_map(pf):
    """Checks that random particles are never in walls, have wrapped headings, and cover the room.

    If the cell picking only used part of the map (e.g. mixing up width and height), the particles
    would bunch up in one area instead of reaching every side of the room.
    """
    particles = np.array([pf.random_particle() for _ in range(500)])

    assert all(pf.is_free(x, y) for x, y, _ in particles)
    assert np.all(np.abs(particles[:, 2]) <= math.pi)

    assert particles[:, 0].min() < -1.0 and particles[:, 0].max() > 1.0
    assert particles[:, 1].min() < -0.5 and particles[:, 1].max() > 0.5


def circular_mean(angles):
    """Averages headings as directions so 179 and -179 degrees average to 180 instead of 0.

    Args:
        angles (np.ndarray): Headings [rad].

    Returns:
        float: Average heading [rad] in [-pi, pi].
    """
    return math.atan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))


def test_no_guess_spreads_particles_over_free_space(pf):
    """Checks global initialization: n_particles rows of [x, y, theta], none in walls, equal weights."""
    pf.initialize_particle_cloud(None)

    assert pf.particle_cloud.shape == (pf.n_particles, 3)
    assert all(pf.is_free(x, y) for x, y, _ in pf.particle_cloud)
    np.testing.assert_allclose(pf.particle_weights, 1 / pf.n_particles)


def test_guess_puts_particles_around_it(pf):
    """Checks that a pose guess gives a cloud centered on it with about init_xy_sigma of spread.

    The guessed heading of 3.0 rad (about 172 degrees) makes some particles cross 180 degrees and
    wrap to negative angles, which should not shift the average heading.
    """
    guess = (-0.5, 0.2, 3.0)
    pf.initialize_particle_cloud(None, guess)
    x, y, theta = pf.particle_cloud.T

    assert pf.particle_cloud.shape == (pf.n_particles, 3)
    assert all(pf.is_free(a, b) for a, b in zip(x, y))

    assert abs(np.mean(x) - guess[0]) < 0.05 and abs(np.mean(y) - guess[1]) < 0.05
    assert 0.2 < np.std(x) < 0.4		# init_xy_sigma is 0.3 m

    assert np.all(np.abs(theta) <= math.pi)
    assert abs(pf.transform_helper.angle_diff(circular_mean(theta), guess[2])) < 0.05


def test_guess_off_the_map_still_finishes(pf):
    """Checks that a guess far off the map still returns instead of hanging.

    Every draw lands off the map and gets redrawn, so this only finishes because normal_particles
    stops after 100 tries.
    """
    pf.initialize_particle_cloud(None, (50.0, 50.0, 0.0))
    assert pf.particle_cloud.shape == (pf.n_particles, 3)


def test_new_cloud_measures_motion_from_now(pf):
    """Checks that resetting the cloud also resets the odom baseline to the current pose.

    The last update was at (0, 0, 0) but the robot is at (2, 3, 1) when the cloud is reset. Without
    resetting the baseline, the next odometry update would push the new cloud by all the driving that
    happened before the reset.
    """
    pf.current_odom_xy_theta = (0.0, 0.0, 0.0)
    pf.odom_pose = particle_filter.Particle(2.0, 3.0, 1.0).as_pose()
    pf.initialize_particle_cloud(None, (0.0, 0.0, 0.0))
    np.testing.assert_allclose(pf.current_odom_xy_theta, (2.0, 3.0, 1.0))
