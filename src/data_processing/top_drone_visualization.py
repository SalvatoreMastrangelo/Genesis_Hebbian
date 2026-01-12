"""
Utility to render a single snapshot of a drone URDF with Genesis.

The goal is to quickly drop a URDF into a clean, white scene, position the
camera at a slight angle, and save a picture that shows overall proportions.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Tuple

import imageio
import numpy as np
import genesis as gs
from genesis.utils.misc import tensor_to_array


class TopDroneVisualizer:
    """
    Minimal helper to render a URDF once and export a PNG.

    Typical usage:
        viz = TopDroneVisualizer()
        png_path = viz.render_urdf("path/to/drone.urdf", "output.png")
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (800, 800),
        camera_distance: float = 3.0,
        camera_height: float = 1.2,
        camera_azimuth_deg: float = 35.0,
    ) -> None:
        """
        Configure the visualizer.

        Parameters
        ----------
        resolution : (int, int)
            Output image resolution (width, height).
        camera_distance : float
            Radial distance of the camera from the origin in meters.
        camera_height : float
            Camera height above the origin in meters.
        camera_azimuth_deg : float
            Yaw angle in degrees around the +Z axis. Positive values rotate
            counter-clockwise when looking down from +Z.
        """
        self.resolution = resolution
        self.camera_distance = camera_distance
        self.camera_height = camera_height
        self.camera_azimuth_deg = camera_azimuth_deg

    def render_urdf(self, urdf_path: str | Path, output_path: str | Path) -> Path:
        """
        Load a URDF, drop it in a clean scene, and save a screenshot.

        Parameters
        ----------
        urdf_path : str | Path
            Path to the URDF file to render.
        output_path : str | Path
            Where to save the resulting PNG.

        Returns
        -------
        Path
            The absolute path to the saved image.
        """
        urdf_path = Path(urdf_path).expanduser().resolve()
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        camera_pos, camera_lookat = self._compute_camera_pose()

        gs.init()

        scene = self._build_scene(camera_pos, camera_lookat)
        try:
            scene.add_entity(
                gs.morphs.URDF(
                    file=str(urdf_path),
                    pos=(0.0, 0.0, 0.0),
                    quat=(1.0, 0.0, 0.0, 0.0),
                    collision=False,  # visualization-only rendering
                    merge_fixed_links=True,
                )
            )

            # Camera must be added before building the scene
            camera = scene.add_camera(
                res=self.resolution,
                pos=camera_pos,
                lookat=camera_lookat,
                up=(0.0, 0.0, 1.0),
                fov=35,
                near=0.05,
                far=20.0,
                debug=True,  # ensures rendering even without viewer
            )

            scene.build(n_envs=0)

            # One render is enough; force_render to update fresh geometry.
            rgb, *_ = camera.render(
                rgb=True,
                depth=False,
                segmentation=False,
                normal=False,
                antialiasing=True,
                force_render=True,
            )
            if rgb is None:
                gs.raise_exception("No RGB output was produced by camera.render().")

            rgb_np = tensor_to_array(rgb)
            # If batched, drop the batch dimension.
            if rgb_np.ndim == 4:
                rgb_np = rgb_np[0]

            # Renderer returns BGR; flip to RGB and drop alpha if present.
            rgb_np = np.flip(rgb_np, axis=-1)
            if rgb_np.shape[-1] == 4:
                rgb_np = rgb_np[..., :3]

            imageio.imwrite(output_path, rgb_np)
            return output_path
        finally:
            # Always clean up to release GPU/GL resources.
            scene.destroy()
            gs.destroy()

    def _compute_camera_pose(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Compute a slight off-axis camera pose to show overall proportions.
        """
        az_rad = math.radians(self.camera_azimuth_deg)
        x = self.camera_distance * math.cos(az_rad)
        y = self.camera_distance * math.sin(az_rad)
        z = self.camera_height
        lookat = (0.0, 0.0, 0.3)
        return (x, y, z), lookat

    def _build_scene(
        self,
        camera_pos: Iterable[float],
        camera_lookat: Iterable[float],
    ) -> gs.Scene:
        """
        Create a stripped-down Genesis scene tuned for still renders.
        """
        return gs.Scene(
            sim_options=gs.options.SimOptions(dt=1 / 60.0, substeps=1),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=30,
                camera_pos=tuple(camera_pos),
                camera_lookat=tuple(camera_lookat),
                res=self.resolution,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[0],
                show_world_frame=False,
                show_link_frame=False,
                background_color=(1.0, 1.0, 1.0),
                ambient_light=(1.0, 1.0, 1.0),
                shadow=False,
                plane_reflection=False,
            ),
            rigid_options=gs.options.RigidOptions(enable_collision=False, enable_joint_limit=False),
            show_viewer=False,
            renderer=gs.renderers.Rasterizer(),
        )


__all__ = ["TopDroneVisualizer"]

