# CREF with TargetDiff

Run all commands from this directory. Install the scientific environment described in the root README and consult the [original TargetDiff repository](https://github.com/guanjq/targetdiff) for data preparation and original model weights.

## Required files

CrossDocked processed LMDB and `data/crossdocked_pocket10_pose_split.pt`; receptor structures under `data/test_set/`.

Provide `pretrained_models/pretrained_diffusion.pt`, `rejection/best_model.pt`, and `rejection/normalization_params.npz`.

The native dataset reader derives its LMDB filename from the configured prefix. Auxiliary training-set proteins are also accessed by the recovered sampler, so test receptors alone are insufficient. Keep receptor files under `data/test_set/` for the internal Vina reward.

## Generate one molecule

```bash
python sample.py --config configs/sampling.yml --start_index 0 --end_index 0 --batch_size 1 --device cuda:0 --data_root data/crossdocked_v1.1_rmsd1.0_pocket10 --split_path data/crossdocked_pocket10_pose_split.pt
```

Results are written under `results/`. To generate more molecules, edit `sample.num_samples` in `configs/sampling.yml`. This example selects test pocket 0; the defaults and this one-pocket example are not a claim of full manuscript-result reproduction.

## Evaluate generated molecules

```bash
python evaluate.py --sample_path results --protein_root data/test_set --docking_mode vina_score
```

Evaluation writes `evaluation_status.json` and exits with an error when no molecule was evaluated successfully. The evaluator can be used independently of the sampling command; internal property scoring is still required by CREF generation.

## Classifier training (optional)

```bash
python rejection/train.py --labeled_dir /your/local/vina_labeled_trajectories --save_path rejection/best_model.pt --normalization_path rejection/normalization_params.npz
```

Inputs are `protein_*.pt` lists containing position/type trajectories, protein features and Vina-based binary labels. Their construction rule must match the intended training experiment. No labeled trajectories or classifier weights are bundled.

## Recovered input identities

The following SHA-256 values identify the weights and splits used in preparation. They do not provide download access or relicense those artifacts.

| File | SHA-256 |
|---|---|
| `pretrained_models/pretrained_diffusion.pt` | `8977821339ba31b7103b46835ef4ddbf7f15f3f2efc5cb1f2a405e7905b09db6` |
| `rejection/normalization_params.npz` | `41900af246308509ab497bb972e3de3543305d4877d9a357e4f7bc047c73a9de` |
| `rejection/best_model.pt` | `91797960b6288e92f4bfac3adf4c25a6964e9d7152ce7c09c581397ac13f6c9f` |
| `data/crossdocked_pocket10_pose_split.pt` | `fc2e32512985e1f01f70b95ca58c0f5abf1ab0a6a7985cabcfedd969ac730bc4` |

## Upstream citation

Guan et al. *3D Equivariant Diffusion for Target-Aware Molecule Generation and Affinity Prediction*. ICLR 2023.
