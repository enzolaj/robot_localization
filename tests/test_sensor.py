"""Unit tests for the laser sensor model and resampling.

Covers scoring particles against simulated and real scans, normalizing the weights, and resampling
the cloud based on those weights.
"""

import math
from pathlib import Path

import numpy as np
import pytest

# 180 evenly spaced beam angles [rad] around the robot, measured from its front like
# update_particles_with_laser expects
theta = np.linspace(-math.pi, math.pi, 180, endpoint=False)

# where the robot really is for most of these tests [x (m), y (m), theta (rad)], somewhere in the open
true_pose = np.array([-0.4, 0.2, 0.5])

# wrong guesses near true_pose, added to it below. the first row is zero so particle 0 is always right
offsets = np.array([
    [0.0, 0.0, 0.0],		# the true pose
    [0.2, 0.0, 0.0],		# 20 cm off in x
    [0.0, 0.2, 0.0],		# 20 cm off in y
    [0.0, 0.0, 0.3],		# right spot, turned about 17 degrees
    [0.8, -0.4, 1.5],		# way off
])


def score(pf, particles, r, theta):
    """Puts particles in the filter, runs the laser update, and returns their weights.

    Beams that came back as inf (no hit) are dropped first, like the range check in run_loop.

    Args:
        pf (particle_filter.ParticleFilter): Filter to score with.
        particles (array-like): M x 3 [x (m), y (m), theta (rad)] particles.
        r (np.ndarray): Beam ranges [m].
        theta (np.ndarray): Beam angles relative to the robot [rad].

    Returns:
        np.ndarray: Length M normalized weights.
    """
    keep = np.isfinite(r)
    pf.particle_cloud = np.array(particles, dtype=float)
    pf.particle_weights = np.ones(len(particles)) / len(particles)
    pf.update_particles_with_laser(r[keep], theta[keep])
    return pf.particle_weights


def test_true_pose_scores_best(pf, simulate_scan):
    """Checks that the particle at the true pose gets the most weight from a scan taken there.

    This is the main laser test. Before walls were scaled by their median, a thin wall scored much
    lower than a thick one, and a pose 20 cm off could beat the true pose here.
    """
    r = simulate_scan(pf, true_pose, theta)
    weights = score(pf, true_pose + offsets, r, theta)

    assert np.argmax(weights) == 0
    assert np.sum(weights) == pytest.approx(1.0)


def test_a_few_bad_beams_dont_change_the_winner(pf, simulate_scan):
    """Checks that the true pose still wins when 15% of beams hit something not on the map.

    27 of the 180 beams are cut short at 0.3 m, like a person standing next to the robot. Since
    every beam gets at least min_likelihood, a few bad beams only cost a particle a little.
    """
    r = simulate_scan(pf, true_pose, theta)
    blocked = np.random.choice(len(r), size=27, replace=False)
    r[blocked] = 0.3
    weights = score(pf, true_pose + offsets, r, theta)
    assert np.argmax(weights) == 0


def test_beams_off_the_map_dont_break_anything(pf, simulate_scan):
    """Checks that a particle 10 m off the map scores without crashing or giving nan weights.

    All its beams land off the map, so they should only count as min_likelihood.
    """
    r = simulate_scan(pf, true_pose, theta)
    weights = score(pf, [true_pose, [10.0, 10.0, 0.0]], r, theta)

    assert np.all(np.isfinite(weights))
    assert np.argmax(weights) == 0


def test_laser_offset_is_used(make_pf, simulate_scan):
    """Checks that laser_offset is applied, using the neato's laser 8.4 cm behind the robot's center.

    The fake scan is taken from the laser's position. If the offset were ignored, the scan would
    line up best with a robot sitting where the laser is instead of at the real center.
    """
    pf = make_pf()
    pf.laser_offset = (-0.084, 0.0)
    r = simulate_scan(pf, true_pose, theta)

    x, y, heading = true_pose
    laser_spot = [x - 0.084 * math.cos(heading), y - 0.084 * math.sin(heading), heading]
    weights = score(pf, [true_pose, laser_spot], r, theta)
    assert weights[0] > weights[1]


def read_first_scan():
    """Reads the ranges of the first scan saved in data.txt.

    data.txt is saved output of ros2 topic echo /scan, where each scan has a "ranges:" line
    followed by one "- value" line per beam.

    Returns:
        np.ndarray: Ranges of the first scan [m], with inf where a beam hit nothing.
    """
    ranges = []
    reading = False
    for line in open(Path(__file__).parent.parent / "robot_localization" / "data.txt"):
        if line.startswith("ranges:"):
            reading = True
        elif reading and line.startswith("- "):
            # echo writes infinity as ".inf" but float() needs "inf"
            ranges.append(float(line[2:].replace(".inf", "inf")))
        elif reading:
            break		# first non-range line ends the first scan
    return np.array(ranges)


def test_real_scan_from_the_gauntlet(make_pf, gauntlet_map):
    """Checks a real scan from the gauntlet sim, taken with the neato at (0, 0) facing +x.

    The real pose has to beat the same spot turned sideways or backwards and spots 1 m away. It
    isnt compared against poses only a few cm off, since the gauntlet map is a little smaller than
    the sim world and the best fit ends up about 0.2 m from (0, 0). If the backwards-facing laser
    were handled wrong, the pose turned 180 degrees would win.
    """
    r = read_first_scan()

    # angle_min and angle_increment from the scan header. the neato's laser faces backwards, so
    # pi is added, same as convert_scan_to_polar_in_robot_frame does when the filter runs
    angles = -3.141590118408203 + np.arange(len(r)) * 0.017453277483582497 + math.pi
    good = (r >= 0.1) & (r < 5.0)		# laser's valid range [m]

    guesses = [
        [0.0, 0.0, 0.0],		# where the robot really was
        [0.0, 0.0, math.pi / 2],		# same spot, turned left
        [0.0, 0.0, math.pi],		# same spot, facing backwards
        [0.0, 0.0, -math.pi / 2],		# same spot, turned right
        [1.0, 0.0, 0.0],		# 1 m ahead
        [-1.0, 0.0, 0.0],		# 1 m behind
        [0.0, -1.0, 0.0],		# 1 m to the right
    ]
    # sharper sigma_hit than the 0.2 m in conftest. the gauntlet is nearly symmetric and the map is a
    # bit too small, so at 0.2 facing backwards wins at (0, 0). the laser direction is what this checks
    pf = make_pf(occupancy_field=gauntlet_map, sigma_hit=0.1)
    pf.laser_offset = (-0.084, 0.0)
    weights = score(pf, guesses, r[good], angles[good])
    assert np.argmax(weights) == 0


def test_normalize_keeps_the_ratios(pf):
    """Checks that normalizing makes weights sum to 1 without changing their ratios."""
    pf.particle_cloud = np.zeros((3, 3))
    pf.particle_weights = np.array([2.0, 1.0, 1.0])
    pf.normalize_particles()
    np.testing.assert_allclose(pf.particle_weights, [0.5, 0.25, 0.25])


@pytest.mark.parametrize("bad_weights", [
    [0.0, 0.0, 0.0, 0.0],		# every particle scored 0
    [np.nan, 1.0, 1.0, 1.0],
    [np.inf, 1.0, 1.0, 1.0],
])
def test_normalize_falls_back_to_equal_weights(pf, bad_weights):
    """Checks that weights that cant be normalized are reset to equal.

    Dividing by a total of 0 gives nan, and a nan or inf spreads to every weight, so none of them
    can be trusted.
    """
    pf.particle_cloud = np.zeros((4, 3))
    pf.particle_weights = np.array(bad_weights)
    pf.normalize_particles()
    np.testing.assert_allclose(pf.particle_weights, 0.25)


def test_resample_keeps_the_shape_and_resets_the_weights(pf):
    """Checks that resampling returns n_particles particles that all start with equal weight."""
    pf.particle_cloud = np.array([pf.random_particle() for _ in range(pf.n_particles)])
    pf.particle_weights = np.random.random(pf.n_particles)
    pf.resample_particles()

    assert pf.particle_cloud.shape == (pf.n_particles, 3)
    np.testing.assert_allclose(pf.particle_weights, 1 / pf.n_particles)


def test_one_heavy_particle_takes_over(make_pf):
    """Checks that when one particle has all the weight, every new particle is a copy of it.

    Each copy is moved by resample_xy_sigma (0.1 m), so they should all be within 0.5 m of it.
    That's loose enough that noise wont fail the test, but far tighter than the original cloud,
    which was spread over the whole room.
    """
    pf = make_pf(random_resample=0.0)
    pf.particle_cloud = np.array([pf.random_particle() for _ in range(pf.n_particles)])
    pf.particle_weights = np.zeros(pf.n_particles)
    pf.particle_weights[7] = 1.0
    heavy = pf.particle_cloud[7].copy()
    pf.resample_particles()

    distances = np.hypot(pf.particle_cloud[:, 0] - heavy[0], pf.particle_cloud[:, 1] - heavy[1])
    assert np.all(distances < 0.5)


def test_particles_are_picked_in_proportion_to_their_weight(make_pf):
    """Checks that each spot gets a share of the new particles matching its share of the weight.

    Spot a (one particle) has 50% of the weight, spot b (one particle) has 30%, and spot c (the
    other 2998 particles) shares the last 20%. 3000 particles keeps random luck small enough to
    allow only 3% either way.
    """
    n = 3000
    pf = make_pf(n_particles=n, random_resample=0.0)
    pf.particle_cloud = np.zeros((n, 3))
    pf.particle_cloud[0] = [-1.0, 0.0, 0.0]		# spot a
    pf.particle_cloud[1] = [0.0, 0.0, 0.0]		# spot b
    pf.particle_cloud[2:] = [1.0, 0.5, 0.0]		# spot c
    pf.particle_weights = np.full(n, 0.2 / (n - 2))
    pf.particle_weights[0] = 0.5
    pf.particle_weights[1] = 0.3
    pf.resample_particles()

    # mean of a boolean array is the fraction that are True
    x = pf.particle_cloud[:, 0]
    assert np.mean(np.abs(x + 1.0) < 0.3) == pytest.approx(0.5, abs=0.03)
    assert np.mean(np.abs(x) < 0.3) == pytest.approx(0.3, abs=0.03)
    assert np.mean(np.abs(x - 1.0) < 0.3) == pytest.approx(0.2, abs=0.03)
