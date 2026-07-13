import collections

import cv2
import dm_env
import numpy as np
from unitree_deploy.real_unitree_env import make_real_env


class UnitreeEnv:
    def __init__(self, robot_type: str, dt: float | None, init_pose_arm: np.ndarray | list[float] | None = None):

        self.env = make_real_env(robot_type, dt, init_pose_arm)
        self.env.connect()

    def get_observation(self):
        result = self.env.get_observation()
        # result.observation["images"] = {
        #     name.split(".")[-1]: np.transpose((img * 255).clip(0, 255).astype(np.uint8), (1, 2, 0))
        #     for name, img in result.observation["images"].items()
        # }
        # print(result.observation["images"])
        # exit(0)
        for name, img in result.observation["images"].items():
            cv2.imwrite(f"{name}.png",img)
        obs = collections.OrderedDict()
        # print("result.observation", result.observation["qpos"])

        left_arm = result.observation["qpos"][:6]
        right_arm = result.observation["qpos"][6:12]
        left_gripper = result.observation["qpos"][12] / 5.4
        right_gripper = result.observation["qpos"][13] / 5.4

        state = np.concatenate((
            left_arm,
            np.array([left_gripper]),
            right_arm,
            np.array([right_gripper])
        ))
        # state = np.concatenate((
        #     left_arm,
        #     right_arm,
        #     np.array([left_gripper]),
        #     np.array([right_gripper])
        # ))
        # print("state", state)
        obs["qpos"] = state
        obs["qvel"] = []
        obs["effort"] = []
        obs["images"] = result.observation["images"]
        return obs

    def reset(self, *, fake=False):
        if not fake:
            pass
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST, reward=0, discount=None, observation=self.get_observation()
        )

    def step(self, action):
        try:
            # print("action", action)
            left_arm = action[:6]
            right_arm = action[7:13]
            left_gripper = action[6]*5.4
            right_gripper = action[13]*5.4

            # left_arm = action[:6]
            # right_arm = action[6:12]
            # left_gripper = action[12]*5.4
            # right_gripper = action[13]*5.4

            action = np.concatenate((
                left_arm,
                right_arm,
                np.array([left_gripper]),
                np.array([right_gripper])
            ))
            # print("action", action)

            self.env.step(action)

            # Return the updated timestep
            return dm_env.TimeStep(
                step_type=dm_env.StepType.MID, reward=0, discount=None, observation=self.get_observation()
            )
        except KeyboardInterrupt:
            self.env.close()


def make_unitree_real_env(
    robot_type: str, dt: float | None, init_pose_arm: np.ndarray | list[float] | None = None
) -> UnitreeEnv:
    return UnitreeEnv(robot_type, dt, init_pose_arm)
