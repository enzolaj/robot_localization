"""Unit tests for the odometry motion model.

Covers moving particles by an odometry change along each particle's own heading, the noise added
to that motion, and deciding when the robot has moved far enough to run an update.
"""

import math

import numpy as np
import pytest

import pf as particle_filter		# cant use pf since it is name of variable in pytest


def move_robot(pf, start, end):
    """Fakes an odometry change from start to end and runs the odometry update.

    Args:
        pf (particle_filter.ParticleFilter): Filter to update.
        start (tuple[float, float, float]): Previous (x (m), y (m), theta (rad)) in odom frame.
        end (tuple[float, float, float]): New (x (m), y (m), theta (rad)) in odom frame.
    """
    pf.current_odom_xy_theta = start
    pf.odom_pose = particle_filter.Particle(*end).as_pose()
    pf.update_particles_with_odom()


@pytest.mark.parametrize("heading, expected", [
    (0.0, (1.0, 0.0)),		# facing +x
    (math.pi / 2, (0.0, 1.0)),		# facing +y
    (math.pi, (-1.0, 0.0)),		# facing -x
    (-math.pi / 2, (0.0, -1.0)),		# facing -y
])
def test_forward_means_forward_for_each_particle(make_pf, heading, expected):
    """Checks that driving forward moves each particle forward along its own heading.

    In the odom frame the robot faces +y and drives 1 m. That is 1 m straight ahead for the robot,
    so each particle should move 1 m along its own heading, not 1 m in +y, since the odom and map
    frames can point in different directions. Noise is off so the result is exact.
    """
    pf = make_pf(odom_trans_noise=0.0, odom_rot_noise=0.0)
    pf.particle_cloud = np.array([[0.0, 0.0, heading]])
    move_robot(pf, (5.0, 5.0, math.pi / 2), (5.0, 6.0, math.pi / 2))

    np.testing.assert_allclose(pf.particle_cloud[0, :2], expected, atol=1e-9)
    assert pf.particle_cloud[0, 2] == pytest.approx(heading)


def test_sideways_drift(make_pf):
    """Checks sideways motion: the robot faces +x and slides 0.5 m to its left.

    A particle facing +y has its left side pointing toward -x, so it should slide 0.5 m in -x.
    """
    pf = make_pf(odom_trans_noise=0.0, odom_rot_noise=0.0)
    pf.particle_cloud = np.array([[0.0, 0.0, math.pi / 2]])
    move_robot(pf, (0.0, 0.0, 0.0), (0.0, 0.5, 0.0))
    np.testing.assert_allclose(pf.particle_cloud[0, :2], (-0.5, 0.0), atol=1e-9)


def test_turning_across_180_degrees(make_pf):
    """Checks that a turn from 3.0 to -3.0 rad is a small 0.28 rad left turn, not a -6 rad one.

    The particle should turn by 2 pi - 6 (about 0.28 rad) and stay where it is.
    """
    pf = make_pf(odom_trans_noise=0.0, odom_rot_noise=0.0)
    pf.particle_cloud = np.array([[0.2, 0.3, 0.0]])
    move_robot(pf, (0.0, 0.0, 3.0), (0.0, 0.0, -3.0))

    np.testing.assert_allclose(pf.particle_cloud[0], (0.2, 0.3, 2 * math.pi - 6.0), atol=1e-9)


def test_noise_spreads_particles_but_keeps_them_centered(pf):
    """Checks that motion noise scatters particles that started together without biasing them.

    300 particles start in one spot and the robot drives 1 m. On average they should still move
    1 m, but each takes a slightly different step.
    """
    pf.particle_cloud = np.zeros((300, 3))
    move_robot(pf, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    assert np.mean(pf.particle_cloud[:, 0]) == pytest.approx(1.0, abs=0.03)
    assert np.std(pf.particle_cloud[:, 0]) > 0.05		# odom_trans_noise of 0.1 m/m gives about 0.1 m here


def test_no_movement_means_no_noise(pf):
    """Checks that a robot that didnt move doesnt scatter the particles, since noise scales with motion."""
    pf.particle_cloud = np.zeros((300, 3))
    move_robot(pf, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    assert np.all(pf.particle_cloud == 0.0)


# thresholds come from conftest: d_thresh 0.2 m and a_thresh pi/6 rad (30 degrees)
@pytest.mark.parametrize("new_pose, should_update", [
    ((0.1, 0.0, 0.0), False),		# under d_thresh
    ((0.25, 0.0, 0.0), True),		# past it in x
    ((0.0, -0.25, 0.0), True),		# past it in negative y
    ((0.0, 0.0, 0.6), True),		# about 34 degrees, past a_thresh
])
def test_moved_far_enough(pf, new_pose, should_update):
    """Checks moved_far_enough_to_update against d_thresh and a_thresh from a start at (0, 0, 0)."""
    pf.current_odom_xy_theta = (0.0, 0.0, 0.0)
    assert pf.moved_far_enough_to_update(new_pose) == should_update


def test_179_to_minus_179_degrees_is_a_small_turn(pf):
    """Checks that 179 to -179 degrees counts as a 2 degree turn, not 358, so it doesnt trigger an update."""
    pf.current_odom_xy_theta = (0.0, 0.0, math.radians(179))
    assert not pf.moved_far_enough_to_update((0.0, 0.0, math.radians(-179)))
