import collections

import dm_env
import numpy as np
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import TextColumn
from rich.progress import TimeRemainingColumn
from unitree_deploy.eval_dataset_env import DatasetEvalEnv


class UnitreeDatasetEnv:
    def __init__(self, repo_id: str, episode_index: int = 0, image_path: str = "", *, visualization: bool = True):
        self.env = DatasetEvalEnv(repo_id, episode_index, image_path, visualization)
        self.step_idx = 0

        self.progress = Progress(
            TextColumn("[cyan]Episode Progress: {task.completed}/{task.total}"),
            BarColumn(bar_width=None),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )
        self.task = self.progress.add_task("", total=self.env.espoide_total_steps)
        self.progress.start()

    def get_observation(self):
        observation = self.env.get_observation()

        if "observation.images.cam_left_high" in observation["images"]:
            observation["images"]["observation.images.cam_high"] = observation["images"].pop(
                "observation.images.cam_left_high"
            )

        if "observation.images.cam_right_high" in observation["images"]:
            observation["images"]["observation.images.cam_low"] = observation["images"].pop(
                "observation.images.cam_right_high"
            )

        observation["images"] = {
            name.split(".")[-1]: np.transpose((img * 255).clip(0, 255).astype(np.uint8), (1, 2, 0))
            for name, img in observation["images"].items()
        }
        obs = collections.OrderedDict()
        obs["qpos"] = observation["qpos"]
        obs["qvel"] = []
        obs["effort"] = []
        obs["images"] = observation["images"]

        return obs

    def reset(self, *, fake=False):
        if not fake:
            pass
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST, reward=0, discount=None, observation=self.get_observation()
        )

    def step(self, action):
        self.step_idx += 1
        self.progress.update(self.task, advance=1)
        if self.step_idx >= self.env.espoide_total_steps:
            self.progress.stop()

        self.env.step(action)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID, reward=0, discount=None, observation=self.get_observation()
        )


def make_dataset_env(
    repo_id: str, episode_index: int = 0, image_path: str = "", *, visualization: bool = True
) -> UnitreeDatasetEnv:
    return UnitreeDatasetEnv(repo_id, episode_index, image_path, visualization=visualization)
