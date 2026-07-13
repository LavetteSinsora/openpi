from typing import List, Optional  # noqa: UP035

import einops
import numpy as np
from openpi_client.runtime import environment as _environment
from PIL import Image
from typing_extensions import override
import unitree_env as _real_env
import unitree_env_dataset as _real_env_dataset


class UnitreeRealEnvironment(_environment.Environment):
    """An environment for an Aloha robot on real hardware."""

    def __init__(
        self,
        reset_position: Optional[List[float]] = None,  # noqa: UP006,UP007
        render_height: int = 224,
        render_width: int = 224,
        robot_type: str = "",
        dt: float = 1 / 30,
        init_pose_arm: np.ndarray | list[float] | None = None,
        *,
        use_dataset: bool = False,
        repo_id: str | None = None,
        visualization: bool = False,
        episode_index: int | None = None,
        image_path: str | None = None,
    ) -> None:
        self._env = (
            _real_env_dataset.make_dataset_env(repo_id, episode_index, image_path, visualization=visualization)
            if use_dataset
            else _real_env.make_unitree_real_env(robot_type, dt, init_pose_arm)
        )
        self._render_height = render_height
        self._render_width = render_width

        self._ts = None

    @override
    def reset(self) -> None:
        self._ts = self._env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = self._ts.observation
        for k in list(obs["images"].keys()):
            if "_depth" in k:
                del obs["images"][k]

        for cam_name in obs["images"]:
            img = np.array(Image.fromarray(obs["images"][cam_name]).resize((224, 224), Image.BILINEAR))

            # img = (
            #     image_tools.resize_with_pad((np.expand_dims(obs["images"][cam_name], axis=0)), self._render_height, self._render_width)
            # )
            obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")

        return {
            "state": obs["qpos"],
            "images": obs["images"],
        }

    @override
    def apply_action(self, action: dict) -> None:
        self._ts = self._env.step(action["actions"])
