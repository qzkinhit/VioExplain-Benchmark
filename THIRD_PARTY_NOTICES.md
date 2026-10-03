# Third-party notices

The root [MIT license](LICENSE) applies to this project's own code. Third-party datasets, separately installed libraries, baseline implementations, and model weights retain their own terms. No external dataset, model weight, or baseline source checkout is bundled in this release.

| Material | Source evidence | Release handling |
|---|---|---|
| SMD | [OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly/tree/7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4), repository MIT [notice](https://github.com/NetManAIOps/OmniAnomaly/blob/7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4/LICENSE) | Downloaded separately, revision checked, upstream notice retained |
| SKAB | [Official repository](https://github.com/waico/SKAB/tree/b2c0d46c2971dcbfe71e26087b6d231998bb91c2), GPL-3.0 [notice](https://github.com/waico/SKAB/blob/b2c0d46c2971dcbfe71e26087b6d231998bb91c2/LICENSE) | Data downloaded separately; no upstream code copied; dataset-specific scope should be checked with its owner |
| Exathlon | [Official source](https://github.com/exathlonbenchmark/exathlon) | No trace redistribution; verify separate code and data terms for chosen version |
| Tennessee Eastman | [Classic MIT/Braatz archive](https://web.mit.edu/braatzgroup/TE_process.zip), SHA256 pinned in data registry | Archive downloaded separately; retain exact UIUC source/binary notice in its readme |
| Chronos-2 | [Official repository](https://github.com/amazon-science/chronos-forecasting) and [model card](https://huggingface.co/amazon/chronos-2) | No weights or upstream source bundled; model terms and revision recorded by each GPU experiment |
| TranAD | [Official locked source](https://github.com/imperial-qore/TranAD/tree/7ffb98d0c18189cc3d9ab732b4cb0278200a0af0), BSD-3-Clause [notice](https://github.com/imperial-qore/TranAD/blob/7ffb98d0c18189cc3d9ab732b4cb0278200a0af0/LICENSE) | Download separately with its notice; the runner checks source hashes and executes the unchanged network classes |
| SARAD | [Official locked source](https://github.com/daidahao/SARAD/tree/24854d9723b4eed31b547344061671c08fbfb3e2), MIT [notice](https://github.com/daidahao/SARAD/blob/24854d9723b4eed31b547344061671c08fbfb3e2/LICENSE) | Download separately with its notice; network, loss and score adaptations are documented and numerically checked |
| Scientific Python dependencies | Installed through `pyproject.toml` | Retain their independently distributed licenses; not relicensed by this repository |

Upstream source and license evidence above were inspected on 2026-10-03. Public accessibility does not itself establish permission to redistribute an arbitrary processed copy. This notice does not assign a new license to any third-party material.
