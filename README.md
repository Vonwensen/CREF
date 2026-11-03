# CREF

Generation code for **CREF: A Critique-Refinement Sampling Framework for Target-aware Molecule Diffusion Models** (unpublished manuscript).

The three backbones are independent native-style projects. Each provides `sample.py`, `evaluate.py`, `models/cref.py`, `rejection/train.py`, and `configs/sampling.yml`. Run commands inside the chosen backbone directory. There is no package installation, unified launcher, or required check suite.

## Environment

Use a Linux scientific environment with PyTorch, PyG and their matching extensions, RDKit, Open Babel's `obabel` command, Vina, PDB2PQR and the AutoDockTools receptor-preparation script. `environment.yml` is a starting specification, not a clean-install lock. Real generation was tested in existing Python 3.8 / PyTorch 1.13.1+cu116 environments. TargetDiff/IPDiff used PyG 2.2.0; DecompDiff used PyG 2.6.1. Keep the environment's `bin` directory on PATH for subprocesses. This version does not import the Open Babel Python bindings or Meeko. If needed, set `CREF_OBABEL` to the executable path and `CREF_PREPARE_RECEPTOR` to the external `prepare_receptor4.py` script.

SA scoring requires RDKit's standard `fpscores.pkl.gz`. If it is not installed with RDKit, set its location:

```bash
export CREF_SA_SCORES=/your/local/SA_Score/fpscores.pkl.gz
```

## Data and weights

Use the public CrossDocked2020 data and each backbone's required processed representation. The full benchmark, classifier-training trajectories and pretrained weights are not bundled. Small generation inputs are provided in `examples/`.

- [TargetDiff data and checkpoints](https://github.com/guanjq/targetdiff#data)
- [DecompDiff data and checkpoints](https://github.com/bytedance/DecompDiff#preprocess)
- [IPDiff data and checkpoints](https://github.com/YangLing0818/IPDiff#data-and-preparation)

Follow the selected backbone's local README for filenames and commands. Backbone checkpoints are downloaded from the upstream projects linked above.

### CREF classifier

Download the [CREF classifier weights for v0.2.0](https://github.com/Vonwensen/CREF/releases/download/v0.2.0/cref_classifier.zip). The archive contains the Vina-supervised classifier `best_model.pt` and its paired `normalization_params.npz`. The same pair is used by all three backbones.

Run the following commands inside the chosen backbone directory (`targetdiff`, `decompdiff`, or `ipdiff`):

```bash
curl -fL https://github.com/Vonwensen/CREF/releases/download/v0.2.0/cref_classifier.zip -o cref_classifier.zip
mkdir -p rejection
unzip cref_classifier.zip -d rejection
```

The resulting files must be `rejection/best_model.pt` and `rejection/normalization_params.npz`. If using more than one backbone, place the pair in each backbone's `rejection/` directory. This archive supplies the CREF classifier; the upstream diffusion checkpoints and datasets are still required.

SHA-256 of `cref_classifier.zip`: `db45b0a1d25c2d641de101266287f74ae9531c3fd0c7ea791c5bcce47bc4380c`. The public download was checked against the source files before documenting this link.

The CREF-owned classifier weights and paired normalization file are licensed under MIT; see [the explicit weights grant](LICENSES/CREF-Weights-MIT.txt) for the covered payload hashes. Upstream diffusion checkpoints retain their own terms.

## Generation

For a small execution example without downloading the full benchmark, follow [examples/README.md](examples/README.md). Each backbone has one held-out pocket, ten auxiliary records, a local split and the required receptor inputs. Upstream model checkpoints and the CREF classifier pair remain external downloads.

Choose `targetdiff`, `decompdiff`, or `ipdiff`, prepare its external files, and run the command in that directory's README. The supplied sampling configuration generates one molecule with the original 1000-step schedule. Increase `sample.num_samples` in the YAML for more samples. Keep the 1000-step schedule for the current time-indexed intervention logic.

The classifier is trained with Vina-based binary labels. The Vina/QED/SA product reward used during sampling is a separate mechanism. Training code is retained for reference; the release's supported purpose is pretrained generation, not reproduction of every paper table or classifier-training dataset.

## Reconstruction and numerical scope

Version 0.2.1 replaces the formerly bundled reconstruction implementation with CREF-authored MIT adapters. Coordinate-only inputs are exchanged with the separately installed `obabel` executable through XYZ/SDF files; explicit predicted bonds use RDKit and, when needed, an adjacency-preserving SDF normalization through `obabel`. Input atom order and coordinates are checked and retained. Ligand PDBQT preparation also uses the external command. The legacy custom bond-pruning and aromatic/valence-repair implementation is not included.

This changes the reconstruction and ligand-preparation backend, including molecular identity, aromaticity, charges and resulting Vina/QED/SA values in some cases. The same-coordinate comparison in `examples/reconstruction_comparison.json` records observed differences. Because these scores also influence CREF candidate selection, generated trajectories can change. This generation release has not been validated as numerically equivalent to the manuscript's original pipeline; reproducing its tables requires a corresponding benchmark rerun or appropriately licensed original implementation. Cite the specific source version/backend used for new experiments.

## License and attribution

Original CREF contributions are licensed under MIT; see the root `LICENSE`. Original and adapted third-party code retains its own terms. In particular, DecompDiff-derived code remains subject to its CC BY-NC 4.0 conditions. The root MIT license does not remove those conditions. See `THIRD_PARTY_NOTICES.md` and the license in each backbone directory. Model weights and datasets are separate artifacts and must be used under their applicable terms.

The previously identified GPL/CC BY-NC overlap in the inherited reconstruction source is addressed by replacing those source files in this version, rather than changing their original notices. New adapters are explicitly MIT; retained DecompDiff source keeps its original CC BY-NC terms. Open Babel and receptor-preparation tools are installed and run separately and retain their own licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for component scope and provenance. This package is not uniformly MIT.

The copyright label “CREF contributors” denotes the rights holders of the original CREF contributions, not the authors of the upstream models.

## Paper and version

The CREF manuscript has not been published. Add the formal paper citation when public metadata is available; retain citations to the original dataset and backbone papers.

Source version: **0.2.1**. Includes small generation fixtures, component license records, and the standard-format reconstruction/ligand-preparation adapters described above. The DecompDiff auxiliary pool supports the small example split while preserving the original selection range for full datasets. This source version is prepared locally; GitHub/Zenodo publication is a separate step. Single-pocket generation/evaluation is not a full benchmark reproduction.

The standardized filenames were re-tested with one molecule from one pocket per backbone: all three completed generation and evaluated one molecule successfully. The validation scripts and logs are intentionally not included in this compact source archive.

The bundled small examples were separately tested in fresh directories without the full benchmark LMDB. All three completed the 1000-step generation schedule and evaluated one molecule with Vina/QED/SA. Input identities and reference completion counts are in `examples/manifest.json` and `examples/reference_results.json`.
