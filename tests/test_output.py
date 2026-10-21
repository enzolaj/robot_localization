"""Unit tests for what the filter puts out.

Covers the pose estimate, the map -> odom transform made from it, and the particle cloud message
sent to RViz.
"""

import math

import numpy as np
import pytest
from builtin_interfaces.msg import Time

import pf as particle_filter		# cant use pf since it is name of variable in pytest


def test_single_particle_is_the_estimate(pf):
    """Checks that with only one particle, the estimate is exactly that particle."""
    pf.particle_cloud = np.array([[0.3, -0.2, 1.0]])
    pf.particle_weights = np.array([1.0])
    assert pf.estimate_pose() == pytest.approx((0.3, -0.2, 1.0))


def test_headings_average_across_180_degrees(pf):
    """Checks that particles at 179 and -179 degrees average to 180 degrees, not 0.

    Both point roughly backwards, so a plain average of the two numbers (0) would point the
    opposite way.
    """
    pf.particle_cloud = np.array([[0.0, 0.0, math.radians(179)], [0.0, 0.0, math.radians(-179)]])
    pf.particle_weights = np.array([0.5, 0.5])
    _, _, theta = pf.estimate_pose()
    assert abs(theta) == pytest.approx(math.pi)		# 180 degrees can come out as pi or -pi


def test_estimate_picks_the_heavier_group_not_the_middle(pf):
    """Checks that the estimate lands in the heavier of two groups instead of between them.

    Group a (around x = -1) has 70% of the weight and group b (around x = 1) has 30%. Averaging
    every particle would give x of about -0.4, where no particle is. estimate_pose only averages
    particles within cluster_radius of the best one, so it should land in group a.
    """
    group_a = np.column_stack([np.random.normal(-1.0, 0.05, 100), np.random.normal(0.0, 0.05, 100), np.zeros(100)])
    group_b = np.column_stack([np.random.normal(1.0, 0.05, 100), np.random.normal(0.0, 0.05, 100), np.zeros(100)])
    pf.particle_cloud = np.vstack([group_a, group_b])
    pf.particle_weights = np.concatenate([np.full(100, 0.7 / 100), np.full(100, 0.3 / 100)])

    x, y, _ = pf.estimate_pose()
    assert x == pytest.approx(-1.0, abs=0.05)


def test_robot_pose_becomes_the_map_to_odom_transform(pf):
    """Checks that update_robot_pose turns the estimate into the right map -> odom transform.

    The filter never publishes the robot's pose directly. It publishes where the odom frame sits on
    the map, and TF places the robot from that. Here odometry says the robot is at (1, 0) facing +x,
    but the particles say it is at (3, 2) facing +y. So odom has to be rotated 90 degrees on the
    map, and since the robot is 1 m along odom's x axis (which now points +y on the map), the odom
    origin sits 1 m below the robot at (3, 1).
    """
    pf.odom_pose = particle_filter.Particle(1.0, 0.0, 0.0).as_pose()
    pf.particle_cloud = np.array([[3.0, 2.0, math.pi / 2]])
    pf.particle_weights = np.array([1.0])
    pf.update_robot_pose()

    # fix_map_to_odom_transform stores the result on transform_helper
    helper = pf.transform_helper
    assert (helper.translation[0], helper.translation[1]) == pytest.approx((3.0, 1.0))
    # rotation is an (x, y, z, w) quaternion, and a pure yaw turn has angle 2 * atan2(z, w)
    x, y, z, w = helper.rotation
    assert 2 * math.atan2(z, w) == pytest.approx(math.pi / 2)


def test_every_particle_is_published(pf):
    """Checks that publish_particles sends one message with every particle's pose and weight.

    Weights are random so a mix-up between particles in the message would show up.
    """
    pf.initialize_particle_cloud(None)
    pf.particle_weights = np.random.random(pf.n_particles)
    pf.normalize_particles()
    pf.publish_particles(Time(sec=12, nanosec=500))

    # the fake publisher in conftest saves every message it gets
    assert len(pf.published) == 1
    msg = pf.published[0]

    assert msg.header.frame_id == "map"
    assert (msg.header.stamp.sec, msg.header.stamp.nanosec) == (12, 500)
    assert len(msg.particles) == pf.n_particles

    # spot check the first two and the last one. headings are sent as quaternions, so they get
    # converted back with the same helper the filter uses
    for i in [0, 1, pf.n_particles - 1]:
        p = msg.particles[i]
        x, y, theta = pf.transform_helper.convert_pose_to_xy_and_theta(p.pose)
        assert (x, y, theta) == pytest.approx(tuple(pf.particle_cloud[i]))
        assert p.weight == pytest.approx(pf.particle_weights[i])
