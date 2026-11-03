# Small generation examples

Each backbone has one held-out pocket and ten auxiliary protein-ligand records, extracted from the recovered CrossDocked2020 representations. `manifest.json` records the original identities, source indices, changes and file hashes. These are small execution fixtures, not a classifier-training dataset or a new benchmark split.

The examples contain the processed LMDB entries, local split files and the receptor PDBs required by internal Vina scoring. DecompDiff also includes its name-to-ID mapping, single-entry test index and atom-count statistics. Source-specific absolute paths have been replaced with relative paths. They do not require the full benchmark LMDB or any private server path.

Install the scientific environment and obtain the upstream model checkpoint(s) described in each backbone's README. Download the CREF classifier pair from the [v0.2.0 release](https://github.com/Vonwensen/CREF/releases/download/v0.2.0/cref_classifier.zip) and extract it into the chosen backbone's `rejection/` directory. These examples contain no model weights. The IPDiff example requires both its diffusion checkpoint and `pretrained_models/ipnet`.

The commands below start from the CREF repository root. Each generates one molecule with the original 1000-step schedule, then evaluates QED, SA and Vina. Use the existing sampling configuration; shortening the diffusion schedule is not supported by the current time-indexed CREF logic.

## TargetDiff

```bash
cd targetdiff
cp -R ../examples/targetdiff/data .
python sample.py --config configs/sampling.yml --data_root data/example --split_path data/example_split.pt --max_data_id 10 --num_extra_proteins 10 --start_index 0 --end_index 0 --batch_size 1 --device cuda:0 --result_path results/example
python evaluate.py --sample_path results/example --protein_root data/test_set --docking_mode vina_score
cd ..
```

## DecompDiff

```bash
cd decompdiff
cp -R ../examples/decompdiff/data .
cp ../examples/decompdiff/utils/evaluation/arm_num_config.pkl utils/evaluation/
cp ../examples/decompdiff/utils/evaluation/scaffold_num_config.pkl utils/evaluation/
python sample.py --config configs/sampling.yml --data_root data/example --split_path data/example_split.pt --index_path data/example_test_index.pkl --start_index 0 --end_index 0 --batch_size 1 --device cuda:0 --outdir results/example --result_path results/example
python evaluate.py results/example --aggregate_meta True --docking_mode vina_score --protein_root data/test_set --result_path results/example/evaluation --num_workers 1
cd ..
```

The sampler bounds the auxiliary-protein pool to the available example entries. On full datasets with at least 99,901 training entries, its original 99,901-entry pool and ten selections are preserved.

## IPDiff

```bash
cd ipdiff
cp -R ../examples/ipdiff/data .
python sample.py --config configs/sampling.yml --train_config ../examples/ipdiff/training.yml --max_data_id 10 --num_extra_proteins 10 --start_index 0 --end_index 0 --batch_size 1 --device cuda:0 --result_path results/example
python evaluate.py --sample_path results/example --protein_root data/test_set --docking_mode vina_score
cd ..
```

`ipdiff/training.yml` here preserves the original network specification and changes only the data prefix and split path for the example. It does not start a training job.

## Expected output

Each generation command writes its native result file under `results/example`. Successful evaluation writes `evaluation_status.json` with `attempted_pockets: 1` and `evaluated: 1`. TargetDiff and IPDiff also report `reconstructed: 1` and `connected: 1`. The evaluation must exit successfully; zero evaluated molecules is a failure. Generated structures and numerical scores may vary with GPU kernels and scientific-library versions.

All three small examples were run in fresh backbone directories with only the bundled example inputs, external pretrained weights and the existing scientific environment. Each completed 1000 diffusion steps and successfully evaluated one molecule. See `reference_results.json` for tested commands, counts and illustrative runtimes. No full benchmark LMDB was present in those test directories.

## Data attribution and terms

The [CrossDocked2020 data license](https://bits.csb.pitt.edu/files/crossdock2020/LICENSE) declares CC0-1.0 for the data files. Cite Francoeur et al., *Three-Dimensional Convolutional Neural Networks and a Cross-Docked Data Set for Structure-Based Drug Design*, Journal of Chemical Information and Modeling, 2020, DOI: 10.1021/acs.jcim.0c00411.

TargetDiff/IPDiff processed fields and this subset's reindexing retain their upstream MIT source attribution where applicable. DecompDiff-derived decomposition fields, indices and atom-count statistics retain its CC BY-NC 4.0 attribution and conditions; they are not relabeled as raw CC0 coordinates. See the upstream licenses, root `THIRD_PARTY_NOTICES.md`, and `LICENSES/CrossDocked-CC0.txt`.

Version 0.2.1 uses the new reconstruction and ligand-preparation adapters documented in the root README. The inherited reconstruction source has been replaced; upstream DecompDiff attribution and data terms remain effective. The backend change can alter molecular structures and property scores, including scores used during sampling. Completion of these small examples establishes execution, not numerical reproduction of the manuscript benchmark. See `reconstruction_comparison.json` for a small same-coordinate comparison.
