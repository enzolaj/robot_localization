""" tests for using the laser: scoring particles against a scan, normalizing the weights, and
    resampling based on those weights """

import math
from pathlib import Path

import numpy as np
import pytest

# 180 beam angles spread evenly all the way around the robot. these are already measured from the
# front of the robot, which is what update_particles_with_laser expects
theta = np.linspace(-math.pi, math.pi, 180, endpoint=False)

# where the robot really is for most of these tests, picked to be somewhere in the open
true_pose = np.array([-0.4, 0.2, 0.5])

# the true pose plus some wrong guesses near it, added to true_pose below. the first row is zero, so
# the first particle is always the right answer.
offsets = np.array([
    [0.0, 0.0, 0.0],      # the true pose
    [0.2, 0.0, 0.0],      # 20 cm off in x
    [0.0, 0.2, 0.0],      # 20 cm off in y
    [0.0, 0.0, 0.3],      # right spot, but turned about 17 degrees
    [0.8, -0.4, 1.5],     # way off
])


def score(pf, particles, r, theta):
    """ puts the given particles in the filter, runs the laser update, and returns their weights.
        beams that came back as inf (didn't hit anything) are dropped first, like run_loop does. """
    keep = np.isfinite(r)
    pf.particle_cloud = np.array(particles, dtype=float)
    pf.particle_weights = np.ones(len(particles)) / len(particles)
    pf.update_particles_with_laser(r[keep], theta[keep])
    return pf.particle_weights


def test_true_pose_scores_best(pf, simulate_scan):
    # this is the main laser test. we make the scan the robot would see from the true pose, and the
    # particle sitting at the true pose should get the most weight
    r = simulate_scan(pf, true_pose, theta)
    weights = score(pf, true_pose + offsets, r, theta)

    assert np.argmax(weights) == 0
    # the weights get normalized at the end of the update, so they should add up to 1.
    # pytest.approx allows for tiny rounding errors.
    assert np.sum(weights) == pytest.approx(1.0)


def test_a_few_bad_beams_dont_change_the_winner(pf, simulate_scan):
    # in real life some beams hit things that aren't on the map, like people or chairs. here 27 of the
    # 180 beams (15%) are cut short at 0.3 m as if something was standing next to the robot. thanks to
    # min_likelihood, a few bad beams only cost a particle a little, so the true pose should still win.
    r = simulate_scan(pf, true_pose, theta)
    blocked = np.random.choice(len(r), size=27, replace=False)
    r[blocked] = 0.3
    weights = score(pf, true_pose + offsets, r, theta)
    assert np.argmax(weights) == 0


def test_beams_off_the_map_dont_break_anything(pf, simulate_scan):
    # the second particle is 10 m outside the map, so all of its beams land off the map. those should just
    # count as min_likelihood. we're making sure nothing crashes and no weights come out as nan.
    r = simulate_scan(pf, true_pose, theta)
    weights = score(pf, [true_pose, [10.0, 10.0, 0.0]], r, theta)

    assert np.all(np.isfinite(weights))
    assert np.argmax(weights) == 0


def test_laser_offset_is_used(make_pf, simulate_scan):
    # the neato's laser is 8.4 cm behind the center of the robot. the fake scan is taken from the laser's
    # spot, so if update_particles_with_laser forgot about the offset, the scan would line up best with a
    # robot sitting where the laser is. with the offset handled, the real center should win.
    pf = make_pf()
    pf.laser_offset = (-0.084, 0.0)
    r = simulate_scan(pf, true_pose, theta)

    x, y, heading = true_pose
    laser_spot = [x - 0.084 * math.cos(heading), y - 0.084 * math.sin(heading), heading]
    weights = score(pf, [true_pose, laser_spot], r, theta)
    assert weights[0] > weights[1]


def read_first_scan():
    """ reads the ranges of the first scan saved in data.txt. that file is a copy of `ros2 topic echo /scan`
        output: each scan has a "ranges:" line followed by one "- value" line per beam. """
    ranges = []
    reading = False
    for line in open(Path(__file__).parent.parent / "robot_localization" / "data.txt"):
        if line.startswith("ranges:"):
            reading = True
        elif reading and line.startswith("- "):
            # the echo writes infinity as ".inf", but python's float() only understands "inf"
            ranges.append(float(line[2:].replace(".inf", "inf")))
        elif reading:
            break   # the first line that isn't a range means the first scan is over
    return np.array(ranges)


def test_real_scan_from_the_gauntlet(make_pf, gauntlet_map):
    # everything above uses fake scans. this one uses a real scan we recorded in the gauntlet sim with the
    # neato sitting at (0, 0) facing +x, scored on the real gauntlet map.
    r = read_first_scan()

    # beam angles from the scan's header: angle_min is -pi and they're 1 degree apart. the neato's laser
    # is mounted facing backwards, so we add pi to every angle, the same thing
    # convert_scan_to_polar_in_robot_frame does when the filter runs.
    # the math here is given in the scan documenation w/ index * step
    angles = -3.141590118408203 + np.arange(len(r)) * 0.017453277483582497 + math.pi
    # keep only the readings inside the laser's valid range (0.1 to 5 m)
    good = (r >= 0.1) & (r < 5.0)

    # the right pose against the same spot turned sideways or backwards, and spots a meter away. we don't
    # compare against poses only a few cm away, since we found the gauntlet map is a little smaller than
    # the sim world, so the best fit is about 0.2 m off from (0, 0). if the backwards-facing laser were
    # handled wrong, the pose turned by 180 degrees would win this.
    guesses = [
        [0.0, 0.0, 0.0],            # where the robot really was
        [0.0, 0.0, math.pi / 2],    # same spot, turned left
        [0.0, 0.0, math.pi],        # same spot, facing backwards
        [0.0, 0.0, -math.pi / 2],   # same spot, turned right
        [1.0, 0.0, 0.0],            # a meter ahead
        [-1.0, 0.0, 0.0],           # a meter behind
        [0.0, -1.0, 0.0],           # a meter to the right
    ]
    # this test uses the sharper sigma_hit of 0.1 instead of our usual 0.2. the gauntlet is close to
    # symmetric, and with the map being a bit too small, the blurrier 0.2 field can't tell facing forward
    # from facing backwards at exactly (0, 0). 0.1 can, and the laser direction is what we're checking.
    pf = make_pf(occupancy_field=gauntlet_map, sigma_hit=0.1)
    pf.laser_offset = (-0.084, 0.0)
    weights = score(pf, guesses, r[good], angles[good])
    assert np.argmax(weights) == 0


def test_normalize_keeps_the_ratios(pf):
    # the first particle has twice the weight of the others. after normalizing, the weights should add up
    # to 1 and the first should still be twice the others.
    pf.particle_cloud = np.zeros((3, 3))
    pf.particle_weights = np.array([2.0, 1.0, 1.0])
    pf.normalize_particles()
    np.testing.assert_allclose(pf.particle_weights, [0.5, 0.25, 0.25])


# three kinds of broken weights: all zero (every particle scored 0), a nan, and an infinity
@pytest.mark.parametrize("bad_weights", [
    [0.0, 0.0, 0.0, 0.0],
    [np.nan, 1.0, 1.0, 1.0],
    [np.inf, 1.0, 1.0, 1.0],
])
def test_normalize_falls_back_to_equal_weights(pf, bad_weights):
    # dividing by the total doesn't work for these: dividing by 0 gives nan, and nan or inf just spreads
    # to every weight. we can't trust any of them, so normalize should make them all equal instead.
    pf.particle_cloud = np.zeros((4, 3))
    pf.particle_weights = np.array(bad_weights)
    pf.normalize_particles()
    np.testing.assert_allclose(pf.particle_weights, 0.25)


def test_resample_keeps_the_shape_and_resets_the_weights(pf):
    pf.particle_cloud = np.array([pf.random_particle() for _ in range(pf.n_particles)])
    pf.particle_weights = np.random.random(pf.n_particles)
    pf.resample_particles()

    # we should get the same number of particles back
    assert pf.particle_cloud.shape == (pf.n_particles, 3)
    # and the new particles should all start out equal. the next laser update gives them new weights.
    np.testing.assert_allclose(pf.particle_weights, 1 / pf.n_particles)


def test_one_heavy_particle_takes_over(make_pf):
    # particle 7 has all the weight and the rest have none, so every new particle should be a copy of
    # particle 7. each copy gets moved a little at random (resample_xy_sigma, 0.1 m), so they won't sit
    # exactly on it. we check they're all within 0.5 m, five times that spread. that's loose enough that
    # random noise won't fail it, but way tighter than the original particles, which were spread over
    # the whole room.
    pf = make_pf(random_resample=0.0)
    pf.particle_cloud = np.array([pf.random_particle() for _ in range(pf.n_particles)])
    pf.particle_weights = np.zeros(pf.n_particles)
    pf.particle_weights[7] = 1.0
    heavy = pf.particle_cloud[7].copy()
    pf.resample_particles()

    distances = np.hypot(pf.particle_cloud[:, 0] - heavy[0], pf.particle_cloud[:, 1] - heavy[1])
    assert np.all(distances < 0.5)


def test_particles_are_picked_in_proportion_to_their_weight(make_pf):
    # three spots 1 m apart. spot a (one particle) gets 50% of the weight, spot b (one particle) gets 30%,
    # and spot c has all the other 2998 particles sharing the last 20%. after resampling, about half the
    # particles should be at a, about 30% at b, and about 20% at c. we use 3000 particles so random luck
    # doesn't move those numbers much, and allow 3% either way.
    n = 3000
    pf = make_pf(n_particles=n, random_resample=0.0)
    pf.particle_cloud = np.zeros((n, 3))
    pf.particle_cloud[0] = [-1.0, 0.0, 0.0]     # spot a
    pf.particle_cloud[1] = [0.0, 0.0, 0.0]      # spot b
    pf.particle_cloud[2:] = [1.0, 0.5, 0.0]     # spot c
    pf.particle_weights = np.full(n, 0.2 / (n - 2))
    pf.particle_weights[0] = 0.5
    pf.particle_weights[1] = 0.3
    pf.resample_particles()

    # count how many new particles ended up near each spot. np.mean of a list of True/False gives the
    # fraction that are True.
    x = pf.particle_cloud[:, 0]
    assert np.mean(np.abs(x + 1.0) < 0.3) == pytest.approx(0.5, abs=0.03)
    assert np.mean(np.abs(x) < 0.3) == pytest.approx(0.3, abs=0.03)
    assert np.mean(np.abs(x - 1.0) < 0.3) == pytest.approx(0.2, abs=0.03)
