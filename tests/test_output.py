""" tests for what the filter puts out: its pose estimate, the map -> odom transform, the particle
    cloud message for rviz, and a full simulated drive that runs every step together """

import math

import numpy as np
import pytest
from builtin_interfaces.msg import Time

import pf as particle_filter     # 'pf' on its own is already the name of our test filter


def test_single_particle_is_the_estimate(pf):
    # with only one particle, the estimate should be exactly that particle
    pf.particle_cloud = np.array([[0.3, -0.2, 1.0]])
    pf.particle_weights = np.array([1.0])
    assert pf.estimate_pose() == pytest.approx((0.3, -0.2, 1.0))


def test_headings_average_across_180_degrees(pf):
    # two particles facing almost straight back, one at 179 degrees and one at -179. they both point
    # roughly at 180, so the average heading should be 180 (pi). a plain average of the two numbers
    # would give 0, which points the opposite way.
    pf.particle_cloud = np.array([[0.0, 0.0, math.radians(179)], [0.0, 0.0, math.radians(-179)]])
    pf.particle_weights = np.array([0.5, 0.5])
    _, _, theta = pf.estimate_pose()
    # 180 degrees can come out as pi or -pi, and both are right, so we compare the size
    assert abs(theta) == pytest.approx(math.pi)


def test_estimate_picks_the_heavier_group_not_the_middle(pf):
    # two groups of 100 particles, one around x = -1 and one around x = 1. group a has 70% of the weight.
    # averaging every particle would give x of about -0.4, a spot between the groups where no particle
    # is. estimate_pose only averages particles near the best one, so it should land in group a.
    group_a = np.column_stack([np.random.normal(-1.0, 0.05, 100), np.random.normal(0.0, 0.05, 100), np.zeros(100)])
    group_b = np.column_stack([np.random.normal(1.0, 0.05, 100), np.random.normal(0.0, 0.05, 100), np.zeros(100)])
    pf.particle_cloud = np.vstack([group_a, group_b])
    pf.particle_weights = np.concatenate([np.full(100, 0.7 / 100), np.full(100, 0.3 / 100)])

    x, y, _ = pf.estimate_pose()
    assert x == pytest.approx(-1.0, abs=0.05)


def test_robot_pose_becomes_the_map_to_odom_transform(pf):
    # the filter never says "the robot is here" directly. it publishes where the odom frame sits on the
    # map, and tf puts the robot on the map from that.
    #
    # odometry says the robot is at (1, 0) facing +x. the particles say it's really at (3, 2) facing +y.
    # so the odom frame has to be turned 90 degrees on the map. and since the robot is 1 m in front of the
    # odom origin (along odom's x, which now points +y on the map), the odom origin is 1 m below the robot,
    # at (3, 1).
    pf.odom_pose = particle_filter.Particle(1.0, 0.0, 0.0).as_pose()
    pf.particle_cloud = np.array([[3.0, 2.0, math.pi / 2]])
    pf.particle_weights = np.array([1.0])
    pf.update_robot_pose()

    # fix_map_to_odom_transform saves the result in transform_helper as a translation and a rotation
    helper = pf.transform_helper
    assert (helper.translation[0], helper.translation[1]) == pytest.approx((3.0, 1.0))
    # the rotation is a quaternion (x, y, z, w). for a turn around the vertical axis, the angle is 2 * atan2(z, w)
    x, y, z, w = helper.rotation
    assert 2 * math.atan2(z, w) == pytest.approx(math.pi / 2)


def test_every_particle_is_published(pf):
    pf.initialize_particle_cloud(None)
    # give the particles different weights so we can tell if they get mixed up in the message
    pf.particle_weights = np.random.random(pf.n_particles)
    pf.normalize_particles()
    pf.publish_particles(Time(sec=12, nanosec=500))

    # the fake publisher in conftest.py saves every message it's given, so there should be exactly one
    assert len(pf.published) == 1
    msg = pf.published[0]

    # rviz needs to know the particles are in the map frame, and the message should carry the scan's time
    assert msg.header.frame_id == "map"
    assert (msg.header.stamp.sec, msg.header.stamp.nanosec) == (12, 500)
    # every particle should be in there, not just some of them
    assert len(msg.particles) == pf.n_particles

    # spot check the first two and the last one: each should come back as the same x, y, heading and
    # weight we have in particle_cloud. the heading is sent as a quaternion, so we turn it back into an
    # angle with the same helper the filter uses.
    for i in [0, 1, pf.n_particles - 1]:
        p = msg.particles[i]
        x, y, theta = pf.transform_helper.convert_pose_to_xy_and_theta(p.pose)
        assert (x, y, theta) == pytest.approx(tuple(pf.particle_cloud[i]))
        assert p.weight == pytest.approx(pf.particle_weights[i])