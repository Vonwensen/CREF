# CREF

Generation code for **CREF: A Critique-Refinement Sampling Framework for Target-aware Molecule Diffusion Models** (unpublished manuscript).

The three backbones are independent native-style projects. Each provides `sample.py`, `evaluate.py`, `models/cref.py`, `rejection/train.py`, and `configs/sampling.yml`. Run commands inside the chosen backbone directory. There is no package installation, unified launcher, or required check suite.

## Environment

Use a Linux scientific environment with PyTorch, PyG and their matching extensions, RDKit, Open Babel, Meeko, Vina, PDB2PQR and AutoDockTools. `environment.yml` is a starting specification, not a clean-install lock. Real generation was tested in existing Python 3.8 / PyTorch 1.13.1+cu116 environments. TargetDiff/IPDiff used PyG 2.2.0; DecompDiff used PyG 2.6.1. Keep the environment's `bin` directory on PATH for receptor preparation subprocesses.

SA scoring requires RDKit's standard `fpscores.pkl.gz`. If it is not installed with RDKit, set its location:

```bash
export CREF_SA_SCORES=/your/local/SA_Score/fpscores.pkl.gz
```

## Data and weights

Use the public CrossDocked2020 data and each backbone's required processed representation. The original data, derived trajectory datasets, pretrained weights and generated structures are not bundled.

- [TargetDiff data and checkpoints](https://github.com/guanjq/targetdiff#data)
- [DecompDiff data and checkpoints](https://github.com/bytedance/DecompDiff#preprocess)
- [IPDiff data and checkpoints](https://github.com/YangLing0818/IPDiff#data-and-preparation)

Follow the selected backbone's local README for filenames and commands. CREF requires the Vina-supervised classifier `rejection/best_model.pt` together with `rejection/normalization_params.npz`. A public CREF checkpoint download endpoint has not yet been provided; backbone download links do not supply this classifier.

## Generation

Choose `targetdiff`, `decompdiff`, or `ipdiff`, prepare its external files, and run the command in that directory's README. The supplied sampling configuration generates one molecule with the original 1000-step schedule. Increase `sample.num_samples` in the YAML for more samples. Keep the 1000-step schedule for the current time-indexed intervention logic.

The classifier is trained with Vina-based binary labels. The Vina/QED/SA product reward used during sampling is a separate mechanism. Training code is retained for reference; the release's supported purpose is pretrained generation, not reproduction of every paper table or classifier-training dataset.

## License and attribution

Original CREF contributions are licensed under MIT; see the root `LICENSE`. Original and adapted third-party code retains its own terms. In particular, DecompDiff-derived code remains subject to its CC BY-NC 4.0 conditions. The root MIT license does not remove those conditions. See `THIRD_PARTY_NOTICES.md` and the license in each backbone directory. Model weights and datasets are separate artifacts and must be used under their applicable terms.

The anonymous copyright label “CREF contributors” denotes the rights holders of the original CREF contributions, not the authors of the upstream models. Confirm the final rights-holder attribution before an identified public release.

## Paper and version

The CREF manuscript has not been published. No DOI, publication venue, or author list is invented in this anonymous package. Add the formal paper citation when public metadata is available; retain citations to the original dataset and backbone papers.

Source release: **0.2.0**. This release standardizes filenames and removes legacy entry points and release-check tooling. Single-pocket generation/evaluation is checked separately during preparation; it is not a full benchmark reproduction.

The standardized filenames were re-tested with one molecule from one pocket per backbone: all three completed generation and evaluated one molecule successfully. The validation scripts and logs are intentionally not included in this compact source archive.
