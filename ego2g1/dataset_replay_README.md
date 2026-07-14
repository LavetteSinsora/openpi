Directly replay joint angle in dataset:
python -m ego2g1.deploy.check replay --dataset ../../lerobot_datasets/ego2g1/put_bottle_in_box --episode 0

Return relative eef from dataset as action chunk (processed exactly the same as policy's output during `deploy`, i.e., uses IK):
python -m ego2g1.deploy.check replay-actions --dataset ../../lerobot_datasets/ego2g1/put_bottle_in_box --episode 0
