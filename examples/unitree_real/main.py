import dataclasses
import logging

import env as _env
import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    action_horizon: int = 25

    num_episodes: int = 1
    max_episode_steps: int = 50000

    # use robot
    robot_type: str = ""
    dt: float = 1 / 30
    init_pose_arm: np.ndarray | list[float] | None = None

    # use_dataset
    use_dataset: bool = True
    repo_id: str = ""
    visualization: bool = False
    episode_index: int = 0
    image_path: str = ""


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    # metadata = ws_client_policy.get_server_metadata()
    runtime = _runtime.Runtime(
        environment=_env.UnitreeRealEnvironment(
            robot_type=args.robot_type,
            dt=args.dt,
            init_pose_arm=args.init_pose_arm,
            use_dataset=args.use_dataset,
            repo_id=args.repo_id,
            visualization=args.visualization,
            episode_index=args.episode_index,
            image_path=args.image_path,
        ),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[],
        max_hz=30,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
