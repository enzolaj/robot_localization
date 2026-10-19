""" tests for the map and for the starting particle cloud: the likelihood field, turning meters into
    grid cells, finding open space, and spreading out the first particles. """

import math

import numpy as np

import pf as particle_filter     # 'pf' on its own is already the name of our test filter


def test_likelihood_field_is_indexed_x_then_y(make_pf, make_map):
    # we use a map that's wider than it is tall (10 x 6 cells) with just one wall cell, at x = 7, y = 2.
    # if the field were built with x and y swapped, its brightest spot would land somewhere else, or
    # the shape would come out as (6, 10).
    grid = np.zeros((10, 6), dtype=int)
    grid[7, 2] = 100
    pf = make_pf(occupancy_field=make_map(grid, resolution=0.1, origin=(0.0, 0.0)))

    assert pf.likelihood_field.shape == (10, 6)
    # np.argmax gives the position of the biggest value as one number, and unravel_index turns that
    # back into (x, y)
    assert np.unravel_index(np.argmax(pf.likelihood_field), (10, 6)) == (7, 2)

    # is_free reads map_array, which is indexed the other way around ([y, x]). make sure it also thinks
    # the wall is at x = 7, y = 2. with 10 cm cells, that cell covers x 0.7 to 0.8 m and y 0.2 to 0.3 m.
    assert not pf.is_free(0.75, 0.25)
    assert pf.is_free(0.25, 0.45)


def test_likelihood_field_is_high_at_walls_and_low_in_the_open(pf):
    # after dividing by the max, the brightest spot should be exactly 1
    assert np.max(pf.likelihood_field) == 1.0

    # a cell in the left wall should be clearly bright. the thick box is what actually reaches 1. a wall
    # that's only one cell thick ends up around 0.2 after blurring, so we only check it's above 0.1.
    ix, iy, _ = pf.world_to_cell(-1.48, 0.0)
    assert pf.likelihood_field[ix, iy] > 0.1

    # a spot in the middle of the room, at least 0.5 m from every wall, should be close to 0
    ix, iy, _ = pf.world_to_cell(-0.5, 0.0)
    assert pf.likelihood_field[ix, iy] < 0.01

    # walking away from the left wall, the values should keep getting smaller
    values = []
    for x in [-1.48, -1.4, -1.3, -1.2]:
        ix, iy, _ = pf.world_to_cell(x, 0.0)
        values.append(pf.likelihood_field[ix, iy])
    assert values == sorted(values, reverse=True)


def test_world_to_cell(pf):
    # the room map's bottom left corner is at (-1.5, -1.0), cells are 5 cm, and it's 60 x 40 cells.
    # world_to_cell gives back (x cell, y cell, whether the point is on the map).
    assert pf.world_to_cell(-1.5, -1.0) == (0, 0, True)       # the corner itself is cell (0, 0)
    assert pf.world_to_cell(-1.46, -0.96) == (0, 0, True)     # anywhere inside a cell gives that same cell
    assert pf.world_to_cell(1.49, 0.99) == (59, 39, True)     # the far corner is the last cell
    assert not pf.world_to_cell(-1.51, 0.0)[2]                # a hair left of the map is off it
    assert not pf.world_to_cell(1.5, 0.0)[2]                  # and so is the right edge itself

    # the laser code hands it whole arrays of points at once, so check that works too
    ix, iy, on_map = pf.world_to_cell(np.array([-1.5, 0.0, 9.0]), np.array([-1.0, 0.0, 9.0]))
    assert list(ix[:2]) == [0, 30] and list(iy[:2]) == [0, 20]
    assert list(on_map) == [True, True, False]


def test_random_particles_land_in_free_space_all_over_the_map(pf):
    particles = np.array([pf.random_particle() for _ in range(500)])

    # none of them should be in a wall, and every heading should be between -pi and pi
    assert all(pf.is_free(x, y) for x, y, _ in particles)
    assert np.all(np.abs(particles[:, 2]) <= math.pi)

    # they should cover the whole room. if the random cell picking had a bug, like only using part of
    # the map's width, they'd all end up bunched in one area.
    assert particles[:, 0].min() < -1.0 and particles[:, 0].max() > 1.0
    assert particles[:, 1].min() < -0.5 and particles[:, 1].max() > 0.5


def circular_mean(angles):
    # the average of some headings. a plain average breaks across 180 degrees (179 and -179 would average
    # to 0, which points the opposite way), so we average them as directions instead and convert back
    return math.atan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))


def test_no_guess_spreads_particles_over_free_space(pf):
    pf.initialize_particle_cloud(None)

    # one row per particle, each row is [x, y, theta]
    assert pf.particle_cloud.shape == (pf.n_particles, 3)
    # nothing starts inside a wall
    assert all(pf.is_free(x, y) for x, y, _ in pf.particle_cloud)
    # with no information yet, every particle is equally likely
    np.testing.assert_allclose(pf.particle_weights, 1 / pf.n_particles)


def test_guess_puts_particles_around_it(pf):
    # the heading here is 3.0 rad, about 172 degrees, so some particles will go past 180 degrees and
    # wrap around to negative angles. this checks that the wrapping happens and doesn't mess anything up.
    guess = (-0.5, 0.2, 3.0)
    pf.initialize_particle_cloud(None, guess)
    x, y, theta = pf.particle_cloud.T

    assert pf.particle_cloud.shape == (pf.n_particles, 3)
    assert all(pf.is_free(a, b) for a, b in zip(x, y))

    # the cloud should be centered on the guess
    assert abs(np.mean(x) - guess[0]) < 0.05 and abs(np.mean(y) - guess[1]) < 0.05
    # and spread out by about init_xy_sigma (0.3 m)
    assert 0.2 < np.std(x) < 0.4

    # headings should stay between -pi and pi, and still average out to the guessed heading
    assert np.all(np.abs(theta) <= math.pi)
    assert abs(pf.transform_helper.angle_diff(circular_mean(theta), guess[2])) < 0.05


def test_guess_off_the_map_still_finishes(pf):
    # if someone clicks way off the map, every draw lands off the map and gets redrawn. the code gives
    # up after 100 tries, so this should finish. without that limit, this test would hang forever.
    pf.initialize_particle_cloud(None, (50.0, 50.0, 0.0))
    assert pf.particle_cloud.shape == (pf.n_particles, 3)


def test_new_cloud_measures_motion_from_now(pf):
    # the robot was last updated at (0, 0, 0), but it's at (2, 3, 1) now when we reset the cloud.
    # the next odometry update should measure from (2, 3, 1). otherwise the new cloud would get shoved
    # by all the driving that happened before the reset.
    pf.current_odom_xy_theta = (0.0, 0.0, 0.0)
    pf.odom_pose = particle_filter.Particle(2.0, 3.0, 1.0).as_pose()
    pf.initialize_particle_cloud(None, (0.0, 0.0, 0.0))
    np.testing.assert_allclose(pf.current_odom_xy_theta, (2.0, 3.0, 1.0))
