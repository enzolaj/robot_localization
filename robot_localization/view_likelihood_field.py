""" shows what a map looks like after build_likelihood_field blurs it, so you can see what the laser
    update is actually scoring particles against.

    run it from anywhere (with ros sourced, since it imports pf.py):
        python3 view_likelihood_field.py                              gauntlet map, sigma_hit 0.2
        python3 view_likelihood_field.py mac_1st_floor_final.yaml     a different map
        python3 view_likelihood_field.py gauntlet.yaml 0.1 0.2 0.3    compare a few sigma_hit values

    the first picture is the plain map, then one blurred picture per sigma_hit. bright means a beam ending
    there scores well, dark means it scores close to nothing.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

from pf import ParticleFilter

maps_dir = Path(__file__).parent.parent / "maps"


def load_map(yaml_name):
    """ reads a map from the maps folder and dresses it up like the OccupancyGrid message map_server
        would send, since that's what build_likelihood_field reads from """
    cfg = yaml.safe_load(open(maps_dir / yaml_name))
    img = np.array(Image.open(maps_dir / cfg["image"])).astype(float)

    # same rules map_server uses: black pixels are walls (100), white is open floor (0), and anything
    # between the two thresholds is unknown (-1)
    p_occupied = (255 - img) / 255.0
    grid = np.where(p_occupied > cfg["occupied_thresh"], 100,
                    np.where(p_occupied < cfg["free_thresh"], 0, -1))

    # the image's first row is its top edge, but the message starts at the bottom left corner, so flip it
    # before flattening it into one long list
    grid = np.flipud(grid)
    height, width = grid.shape
    origin = SimpleNamespace(position=SimpleNamespace(x=cfg["origin"][0], y=cfg["origin"][1]))
    info = SimpleNamespace(resolution=cfg["resolution"], width=width, height=height, origin=origin)
    return SimpleNamespace(map=SimpleNamespace(data=grid.flatten().tolist(), info=info))


def blur(occupancy_field, sigma_hit):
    """ runs the real build_likelihood_field from pf.py on the map, so the picture is exactly what the
        filter uses. __new__ makes the filter without running __init__, which would wait for map_server. """
    pf = ParticleFilter.__new__(ParticleFilter)
    pf.occupancy_field = occupancy_field
    pf.sigma_hit = sigma_hit
    pf.build_likelihood_field()
    return pf.likelihood_field


def main():
    yaml_name = sys.argv[1] if len(sys.argv) > 1 else "gauntlet.yaml"
    sigmas = [float(s) for s in sys.argv[2:]] or [0.2]

    field = load_map(yaml_name)
    info = field.map.info
    # extent tells imshow where the picture sits in meters, so the axes line up with rviz's map frame
    left = info.origin.position.x
    bottom = info.origin.position.y
    extent = [left, left + info.width * info.resolution, bottom, bottom + info.height * info.resolution]

    fig, axes = plt.subplots(1, len(sigmas) + 1, figsize=(5 * (len(sigmas) + 1), 5), sharex=True, sharey=True)

    # the plain map: walls black, open floor white, unknown gray. gray_r draws 0 as white and 1 as black,
    # so walls become 1, unknown cells 0.3, and open floor stays 0
    raw = np.array(field.map.data).reshape(info.height, info.width)
    shades = np.where(raw > 0, 1.0, np.where(raw < 0, 0.3, 0.0))
    axes[0].imshow(shades, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1)
    axes[0].set_title(f"{yaml_name} (plain map)")

    for ax, sigma in zip(axes[1:], sigmas):
        # likelihood_field is indexed [x, y], but imshow wants [row, column] which is [y, x], hence the .T.
        # origin="lower" puts row 0 at the bottom, matching the map.
        image = ax.imshow(blur(field, sigma).T, origin="lower", extent=extent, cmap="cividis", vmin=0, vmax=1)
        ax.set_title(f"sigma_hit = {sigma} m")

    fig.colorbar(image, ax=axes[1:], label="likelihood (1 = wall more likely, bright)")
    for ax in axes:
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    plt.show()


if __name__ == "__main__":
    main()
