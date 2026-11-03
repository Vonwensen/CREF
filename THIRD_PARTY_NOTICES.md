# Third-party notices

The root MIT license covers only original CREF contributions for which the CREF contributors hold licensing rights. It does not replace licenses on original or adapted upstream material. Existing source-file notices remain effective.

| Directory/component | Upstream | Retained license |
|---|---|---|
| `targetdiff/` | [TargetDiff](https://github.com/guanjq/targetdiff), Jiaqi Guan and contributors | MIT, in `targetdiff/LICENSE` |
| `ipdiff/` | [IPDiff](https://github.com/YangLing0818/IPDiff), Zhilin Huang and contributors | MIT, in `ipdiff/LICENSE` |
| `decompdiff/` | [DecompDiff](https://github.com/bytedance/DecompDiff), ByteDance and contributors | CC BY-NC 4.0, in `decompdiff/LICENSE` |
| SA scorer and reconstruction utilities | Original notices embedded in source files | Respective source-file terms |

CREF integrates critique gating, rollback, reward-based candidate selection and refinement into the native samplers. Release edits also standardize module names, make paths portable, export classifier normalization statistics, and make zero-success evaluation fail explicitly. These modifications do not erase upstream copyright or license conditions. The precise upstream commit for historical recovered files has not been established; repository links identify origins, not a certified commit match.

Please cite the original backbone papers named in their respective READMEs and the CrossDocked2020 dataset when using those components. Third-party attribution is preserved in this anonymous release. No pretrained weights or molecular datasets are redistributed.
