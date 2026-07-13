# Run unitree (Real Robot)

```bash
# Create virtual environment
uv venv --python 3.10 examples/unitree_real/.venv
source examples/unitree_real/.venv/bin/activate
git clone http://10.0.8.99:4000/gh/unitree-deploy.git && cd unitree-deploy && uv pip install -e . && pip install -e ".[lerobot]"
uv pip sync examples/unitree_real/requirements.txt
uv pip install -e packages/openpi-client

# shell
nohup bash train.sh > train.log 2>&1 &
nohup bash train_1.sh > train_1.log 2>&1 &

ps -ef | grep train.sh

tail -f train.log
# compute_norm
python scripts/compute_norm_stats.py --config-name pi05_unitree_z1_satckbox
python scripts/compute_norm_stats.py --config-name pi05_unitree_z1_fold_clothes


# train
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_unitree_z1_fold_clothes --exp-name=fold_clothes_experiment --overwrite



XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_unitree_g1_dex1_pick_coffeebottle --exp-name=pi05_unitree_g1_dex1_pick_coffeebottle_experiment --overwrite

# Run the robot
python -m examples.unitree_real.main




# Run the dataset
python examples/unitree_real/main.py \
  --args.host 10.3.0.241 \
  --args.port 14090 \
  --args.use_dataset \
  --args.repo_id unitreerobotics/Z1_Dual_Dex1_FoldClothes_Dataset \
  --args.visualization \
  --args.episode_index 10 \
  --args.image_path /home/unitree/code/test/pi05/openpi/
```


pi05_unitree_z1_fold_clothes:
```bash
python scripts/serve_policy.py policy:checkpoint --policy.config=pi05_unitree_z1_fold_clothes --policy.dir=checkpoints/pi05_unitree_z1_fold_clothes/fold_clothes_experiment/19999

python examples/unitree_real/main.py \
  --args.host 10.3.0.241 \
  --args.port 14090 \
  --args.use_dataset \
  --args.repo_id unitreerobotics/Z1_Dual_Dex1_FoldClothes_Dataset \
  --args.visualization \
  --args.episode_index 20 \
  --args.image_path /home/unitree/code/test/pi05/openpi/
```



pi05_unitree_z1_pour_coffee:
```bash
python scripts/serve_policy.py policy:checkpoint --policy.config=pi05_unitree_z1_pour_coffee --policy.dir=checkpoints/pi05_unitree_z1_pour_coffee/pour_coffee_experiment/19999

python examples/unitree_real/main.py \
  --args.host 10.3.0.241 \
  --args.port 14090 \
  --args.use_dataset \
  --args.repo_id unitreerobotics/Z1_Dual_Dex1_PourCoffee_Dataset \
  --args.visualization \
  --args.episode_index 10

python examples/unitree_real/main.py \
  --args.host 10.3.0.241 \
  --args.port 14090 \
  --args.no-use-dataset \
  --args.robot_type z1_dual_dex1_opencv 
```


pi05_unitree_z1_new_pour_coffee:
```bash
python scripts/serve_policy.py policy:checkpoint --policy.config=pi05_unitree_z1_new_pour_coffee --policy.dir=checkpoints/pi05_unitree_z1_new_pour_coffee/new_pour_coffee_experiment/19999

python examples/unitree_real/main.py \
  --args.host 10.3.0.241 \
  --args.port 14090 \
  --args.use_dataset \
  --args.repo_id /home/unitree/llm/manipulation/lerobot_v2.0/new_z1_pour_coffee \
  --args.visualization \
  --args.episode_index 10
```

---------------------------------------------------------
pi05_unitree_z1_new_fold_clothes:
```bash
python scripts/serve_policy.py policy:checkpoint --policy.config=pi05_unitree_z1_new_fold_clothes --policy.dir=checkpoints/pi05_unitree_z1_new_fold_clothes/new_fold_clothes_experiment/19999

python examples/unitree_real/main.py \
  --args.host 10.3.0.241 \
  --args.port 14090 \
  --args.use_dataset \
  --args.repo_id /home/unitree/llm/manipulation/lerobot_v2.0/new_z1_fold_cloth \
  --args.visualization \
  --args.episode_index 10
```


pi05_unitree_g1_brainco_grasp_oreo:
```bash
python scripts/serve_policy.py policy:checkpoint --policy.config=pi05_unitree_g1_brainco_grasp_oreo --policy.dir=checkpoints/pi05_unitree_g1_brainco_grasp_oreo/pi05_unitree_g1_brainco_grasp_oreo_experiment/19999

python examples/unitree_real/main.py \
  --args.host 10.3.0.241 \
  --args.port 14090 \
  --args.use_dataset \
  --args.repo_id unitreerobotics/G1_Brainco_GraspOreo_Dataset \
  --args.visualization \
  --args.episode_index 10
```



pi05_unitree_g1_brainco_pick_place:
```bash
python scripts/serve_policy.py policy:checkpoint --policy.config=pi05_unitree_g1_brainco_pick_place --policy.dir=checkpoints/pi05_unitree_g1_brainco_pick_place/pi05_unitree_g1_brainco_grasp_oreo_experiment/19999

python examples/unitree_real/main.py \
  --args.host 10.3.0.241 \
  --args.port 14090 \
  --args.use_dataset \
  --args.repo_id unitreerobotics/G1_Brainco_GraspOreo_Dataset \
  --args.visualization \
  --args.episode_index 10
```


pi05_unitree_g1_dex3_pick_place:
```bash
python scripts/serve_policy.py policy:checkpoint --policy.config=pi05_unitree_g1_dex3_pick_place --policy.dir=checkpoints/pi05_unitree_g1_dex3_pick_place/pi05_unitree_g1_dex3_pick_place_experiment/19999

python examples/unitree_real/main.py \
  --args.host 10.3.0.241 \
  --args.port 14090 \
  --args.use_dataset \
  --args.repo_id unitreerobotics/G1_Dex3_PickPlace_Merge_Dataset \
  --args.visualization \
  --args.episode_index 10
```


pi05_unitree_z1_fold_clothes_merge:
```bash
python scripts/serve_policy.py policy:checkpoint --policy.config=pi05_unitree_z1_fold_clothes_merge --policy.dir=checkpoints/pi05_unitree_z1_fold_clothes_merge/pi05_unitree_z1_fold_clothes_merge_experiment/19999

python examples/unitree_real/main.py \
  --args.host 10.3.0.241 \
  --args.port 14090 \
  --args.use_dataset \
  --args.repo_id unitreerobotics/Z1_Dual_Dex1_FoldClothes_Dataset_Merge \
  --args.visualization \
  --args.episode_index 10
```


pi0_unitree_z1_new_fold_clothes:
```bash
python scripts/serve_policy.py policy:checkpoint --policy.config=pi0_unitree_z1_new_fold_clothes --policy.dir=checkpoints/pi0_unitree_z1_new_fold_clothes/pi0_unitree_z1_new_fold_clothes_experiment/19999

python examples/unitree_real/main.py \
  --args.host 10.3.0.241 \
  --args.port 14090 \
  --args.use_dataset \
  --args.repo_id /home/unitree/datasets/lerobot/Henry-Ellis/new_z1_fold_cloth_old \
  --args.visualization \
  --args.episode_index 10
```


pi0_unitree_g1_dex1_pick_coffeebottle:
```bash
python scripts/serve_policy.py policy:checkpoint --policy.config=pi0_unitree_g1_dex1_pick_coffeebottle --policy.dir=checkpoints/pi0_unitree_g1_dex1_pick_coffeebottle/pi0_unitree_g1_dex1_pick_coffeebottle_experiment/19999

python examples/unitree_real/main.py \
  --args.host 10.3.0.241 \
  --args.port 14090 \
  --args.use_dataset \
  --args.repo_id unitreerobotics/G1_Dex1_PickCoffeeBottle_Dataset \
  --args.visualization \
  --args.episode_index 10


pi0_unitree_g1_dex1_pick_3task:
```bash
python scripts/serve_policy.py policy:checkpoint --policy.config=pi0_unitree_g1_dex1_pick_3task --policy.dir=checkpoints/pi0_unitree_g1_dex1_pick_3task/pi0_unitree_g1_dex1_pick_3task_experiment/19999

python examples/unitree_real/main.py \
  --args.host 10.3.0.241 \
  --args.port 14090 \
  --args.use_dataset \
  --args.repo_id unitreerobotics/G1_Dex1_3Picktask_Dataset_Merge_fix \
  --args.visualization \
  --args.episode_index 10
```



Terminal window 3:

```bash
uv run scripts/serve_policy.py --env ALOHA --default_prompt='take the toast out of the toaster'
```


