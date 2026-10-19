""" tests for moving the particles with odometry, and for deciding when the robot has moved far
    enough to run an update """

import math

import numpy as np
import pytest

import pf as particle_filter     # 'pf' on its own is already the name of our test filter


def move_robot(pf, start, end):
    """ pretends the odometry went from start to end, both (x, y, theta) in the odom frame, and runs
        the odometry update """
    pf.current_odom_xy_theta = start
    pf.odom_pose = particle_filter.Particle(*end).as_pose()
    pf.update_particles_with_odom()


# parametrize runs this test four times, once per (heading, expected) pair
@pytest.mark.parametrize("heading, expected", [
    (0.0, (1.0, 0.0)),               # facing +x, so it should end up 1 m along x
    (math.pi / 2, (0.0, 1.0)),       # facing +y
    (math.pi, (-1.0, 0.0)),          # facing -x
    (-math.pi / 2, (0.0, -1.0)),     # facing -y
])
def test_forward_means_forward_for_each_particle(make_pf, heading, expected):
    # this is the main thing the odometry update has to get right. in the odom frame the robot faces +y
    # and drives 1 m. that's 1 m straight ahead for the robot, so each particle should move 1 m straight
    # ahead along its own heading. it should not move 1 m in +y, because the odom frame and the map
    # frame point in different directions.
    # the noise is turned off so we can check exact numbers.
    pf = make_pf(odom_trans_noise=0.0, odom_rot_noise=0.0)
    pf.particle_cloud = np.array([[0.0, 0.0, heading]])
    move_robot(pf, (5.0, 5.0, math.pi / 2), (5.0, 6.0, math.pi / 2))

    np.testing.assert_allclose(pf.particle_cloud[0, :2], expected, atol=1e-9)
    # it drove straight, so the heading shouldn't change
    assert pf.particle_cloud[0, 2] == pytest.approx(heading)


def test_sideways_drift(make_pf):
    # the robot faces +x and slides 0.5 m to its left. a particle facing +y has its left side pointing
    # toward -x, so it should slide 0.5 m in -x.
    pf = make_pf(odom_trans_noise=0.0, odom_rot_noise=0.0)
    pf.particle_cloud = np.array([[0.0, 0.0, math.pi / 2]])
    move_robot(pf, (0.0, 0.0, 0.0), (0.0, 0.5, 0.0))
    np.testing.assert_allclose(pf.particle_cloud[0, :2], (-0.5, 0.0), atol=1e-9)


def test_turning_across_180_degrees(make_pf):
    # going from 3.0 rad to -3.0 rad looks like a change of -6 rad if you just subtract, but it's really
    # a small left turn of about 0.28 rad across the 180 degree line. the particle should turn by 0.28
    # and stay where it is.
    pf = make_pf(odom_trans_noise=0.0, odom_rot_noise=0.0)
    pf.particle_cloud = np.array([[0.2, 0.3, 0.0]])
    move_robot(pf, (0.0, 0.0, 3.0), (0.0, 0.0, -3.0))

    np.testing.assert_allclose(pf.particle_cloud[0], (0.2, 0.3, 2 * math.pi - 6.0), atol=1e-9)


def test_noise_spreads_particles_but_keeps_them_centered(pf):
    # 300 particles start in the same spot and the robot drives 1 m. with the default noise, each one
    # takes a slightly different step. on average they should still have gone 1 m, but they shouldn't
    # all be in the same place anymore.
    pf.particle_cloud = np.zeros((300, 3))
    move_robot(pf, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    assert np.mean(pf.particle_cloud[:, 0]) == pytest.approx(1.0, abs=0.03)
    # odom_trans_noise is 0.1, so the spread should be about 10% of the 1 m drive
    assert np.std(pf.particle_cloud[:, 0]) > 0.05


def test_no_movement_means_no_noise(pf):
    # the noise is based on how far the robot moved, so a robot that didn't move shouldn't scatter
    # the particles at all
    pf.particle_cloud = np.zeros((300, 3))
    move_robot(pf, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    assert np.all(pf.particle_cloud == 0.0)


@pytest.mark.parametrize("new_pose, should_update", [
    ((0.1, 0.0, 0.0), False),    # only 0.1 m, under the 0.2 m limit
    ((0.25, 0.0, 0.0), True),    # past the limit in x
    ((0.0, -0.25, 0.0), True),   # past it in y, going the negative way
    ((0.0, 0.0, 0.6), True),     # turned about 34 degrees, past the 30 degree limit
])
def test_moved_far_enough(pf, new_pose, should_update):
    pf.current_odom_xy_theta = (0.0, 0.0, 0.0)
    assert pf.moved_far_enough_to_update(new_pose) == should_update


def test_179_to_minus_179_degrees_is_a_small_turn(pf):
    # this is a 2 degree turn. if the code just subtracted the two angles, it would see a 358 degree turn
    # and run an update for no reason.
    pf.current_odom_xy_theta = (0.0, 0.0, math.radians(179))
    assert not pf.moved_far_enough_to_update((0.0, 0.0, math.radians(-179)))
