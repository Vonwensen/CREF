# CREF with DecompDiff

Run all commands from this directory. Install the scientific environment described in the root README and consult the [original DecompDiff repository](https://github.com/bytedance/DecompDiff) for data preparation and original model weights.

## Required files

Decomposition-aware LMDB, name-to-ID mapping, `data/split_by_name.pt`, `data/test_index.pkl`, and `data/test_set/`. Also provide `utils/evaluation/arm_num_config.pkl` and `scaffold_num_config.pkl`.

Provide `pretrained_models/uni_o2_bond.pt`, `rejection/best_model.pt`, and `rejection/normalization_params.npz`.

The native dataset reader derives its LMDB filename from the configured prefix. Auxiliary training-set proteins are also accessed by the recovered sampler, so test receptors alone are insufficient. Keep receptor files under `data/test_set/` for the internal Vina reward.

## Generate one molecule

```bash
python sample.py --config configs/sampling.yml --start_index 0 --end_index 0 --batch_size 1 --device cuda:0 --data_root data/crossdocked_v1.1_rmsd1.0_processed --split_path data/split_by_name.pt --outdir results
```

Results are written under `results/`. To generate more molecules, edit `sample.num_samples` in `configs/sampling.yml`. This example selects test pocket 0; the defaults and this one-pocket example are not a claim of full manuscript-result reproduction.

## Evaluate generated molecules

```bash
python evaluate.py results --aggregate_meta True --docking_mode vina_score --result_path results/evaluation --num_workers 1
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
| `pretrained_models/uni_o2_bond.pt` | `9d3daf75ff37fbc314f2dd43f899c7c104a8f68992cad5c0e8446574fac39c17` |
| `rejection/normalization_params.npz` | `41900af246308509ab497bb972e3de3543305d4877d9a357e4f7bc047c73a9de` |
| `rejection/best_model.pt` | `91797960b6288e92f4bfac3adf4c25a6964e9d7152ce7c09c581397ac13f6c9f` |
| `data/split_by_name.pt` | `7ba210bc9b8a89034b8e189f26dd055141b0cbcc7aab66898d68e84140ad6e3b` |
| `data/test_index.pkl` | `18f802c6a2224a04204710f20dc61fbf377b221bc50716fda3c26663ce542052` |

## Upstream citation

Guan et al. *DecompDiff: Diffusion Models with Decomposed Priors for Structure-Based Drug Design*. ICML 2023.
