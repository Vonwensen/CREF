# Third-party notices

The root MIT license covers only original CREF contributions for which the CREF contributors hold licensing rights. It does not replace licenses on original or adapted upstream material. Existing source-file notices remain effective.

| Directory/component | Upstream | Retained license |
|---|---|---|
| `targetdiff/` | [TargetDiff](https://github.com/guanjq/targetdiff), Jiaqi Guan and contributors | MIT, in `targetdiff/LICENSE` |
| `ipdiff/` | [IPDiff](https://github.com/YangLing0818/IPDiff), Zhilin Huang and contributors | MIT, in `ipdiff/LICENSE` |
| DecompDiff-derived files under `decompdiff/` | [DecompDiff](https://github.com/bytedance/DecompDiff), ByteDance and contributors | CC BY-NC 4.0, in `decompdiff/LICENSE`; new CREF adapters listed separately below have explicit MIT headers |
| `*/utils/reconstruct.py`, `*/utils/obabel_cli.py` | New CREF adapters written for this release using public RDKit APIs and standard XYZ/SDF/PDBQT interfaces | MIT, stated in each new file; these replace the inherited reconstruction implementation |
| `*/utils/evaluation/sascorer.py` | Novartis Institutes for BioMedical Research Inc., 2013; RDKit SA scorer | Embedded BSD 3-clause-style terms retained; reproduced in `LICENSES/SA-Score-BSD.txt` |
| CrossDocked coordinates/features in `examples/` | [CrossDocked2020](https://bits.csb.pitt.edu/files/crossdock2020/) | Upstream data files declared CC0-1.0, see `LICENSES/CrossDocked-CC0.txt` |
| DecompDiff example decomposition metadata and atom-count statistics | DecompDiff preprocessing representation | Retain the upstream CC BY-NC 4.0 attribution and conditions; not relabeled as raw CC0 coordinates |
| RDKit | [RDKit](https://www.rdkit.org/) | Separately installed BSD-licensed dependency; license reference in `LICENSES/RDKit-BSD.txt` |
| Open Babel command-line tool | [Open Babel](https://openbabel.org/docs/Command-line_tools/babel.html) | Separately installed GPL tool, invoked through standard file formats; not bundled or imported through Python bindings. GPL reference text is retained in `LICENSES/GPL-2.0.txt`. |
| Receptor-preparation utilities | AutoDockTools and PDB2PQR | Separately installed external programs with their own terms; no tool source or executable is bundled |

CREF integrates critique gating, rollback, reward-based candidate selection and refinement into the native samplers. Release edits also standardize module names, make paths portable, export classifier normalization statistics, and make zero-success evaluation fail explicitly. These modifications do not erase upstream copyright or license conditions. The precise upstream commit for historical recovered files has not been established; repository links identify origins, not a certified commit match.

Please cite the original backbone papers named in their respective READMEs and the CrossDocked2020 dataset when using those components. No pretrained weights are bundled. The only molecular inputs included are the small fixtures documented in `examples/README.md` and `examples/manifest.json`.

## Reconstruction-source replacement

The current official TargetDiff root license is spelled `LICIENCE`: https://github.com/guanjq/targetdiff/blob/main/LICIENCE. It is MIT. IPDiff's official root `LICENSE` is MIT; DecompDiff's official root `LICENSE` is CC BY-NC 4.0. These root-level findings do not override more specific source-file notices.

Earlier recovered `utils/reconstruct.py` files explicitly identified GPL v2 code from liGAN. The old DecompDiff file also contained a ByteDance CC BY-NC 4.0 header. The source-overlap finding concerned those inherited files. In version 0.2.1, all three files have been wholly replaced with newly written adapters; none of the old private bond-pruning, reachability, aromatic-repair or valence-repair functions is retained. Existing third-party notices remain intact in all retained upstream-derived files. This replacement does not alter or retroactively relicense historical source versions.

The new adapter constructs explicit bond graphs through public RDKit APIs. For coordinate-only perception, it sends XYZ text to a separately installed `obabel` program and receives SDF text. Explicit aromatic graphs that require normalization are exchanged as SDF, with atom identities and adjacency verified before accepting the result. Ligand preparation is a separate SDF-to-PDBQT conversion. This version contains no Open Babel/Pybel/Meeko library imports. The retained DecompDiff SDF-to-PDB utility was adapted to the same external interface. The receptor-preparation script is located without importing AutoDockTools and runs in its own process.

External tools remain separate installations under their respective terms; the archive does not relicense or distribute their implementations. The design uses ordinary chemical-file interchange, not a copied GPL implementation or a custom bridge carrying internal tool objects. Relevant references are the [Open Babel command-line documentation](https://openbabel.org/docs/Command-line_tools/babel.html), the [Open Babel library/bindings license notice](https://openbabel.org/docs/UseTheLibrary/intro.html), and the [FSF discussion of separate programs and aggregation](https://www.gnu.org/licenses/gpl-faq.en.html#MereAggregation). The component statements describe what this archive includes and how it calls dependencies; they do not certify arbitrary downstream combinations or redistribute external tools under MIT.

The replacement changes scientific behavior. It does not implement the previous custom reconstruction heuristics, and its ligand preparation may assign different charges and torsions. Same-coordinate comparisons demonstrate differences in molecular identity and QED; the complete manuscript benchmark has not been rerun. See the root README and `examples/reconstruction_comparison.json`. The source overlap is addressed for this new archive; numerical equivalence to older source releases is not claimed.

The CREF rights holder has explicitly authorized an MIT grant for the separately hosted classifier weights. The exact two payloads and hashes covered by this grant are listed in `LICENSES/CREF-Weights-MIT.txt`; a standalone copy is prepared for attachment to the existing public weights release. This source archive contains the grant and a download link, not the weights themselves. The grant does not relicense upstream diffusion checkpoints or datasets.
