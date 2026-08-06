# Tenkai

Tenkai is an atom-loss-aware surface-code design that combines a Walking-SE
syndrome-extraction circuit with a loss-shadow-guided decoder. This repository
provides the reference implementation of Tenkai and lossy-DEM, together with a
command-line runner for reproducible single-point simulations.

## Installation

Tenkai supports CPython 3.12 and uses
[`uv`](https://docs.astral.sh/uv/) 0.10.2 for dependency management.

```bash
git clone https://github.com/Chenning-Tao/MICRO2026-Tenkai.git
cd MICRO2026-Tenkai
uv sync --locked
uv run --locked python -m tenkai --help
```

## Quick start

Run Tenkai and lossy-DEM on the same sampled shots:

```bash
uv run --locked python -m tenkai run-point \
  --method both \
  --distance 5 \
  --physical-error-rate 0.004 \
  --loss-fraction 0.5 \
  --shots 16 \
  --seed 2026 \
  --workers 1 \
  --output-dir tenkai-output/example
```

`--method` accepts:

- `tenkai` for the Tenkai decoder;
- `lossy-dem` for the lossy-DEM decoder;
- `both` to evaluate both decoders on the same shots.

The command writes a machine-readable `result.json` and resumable shot
fragments under the selected output directory.

## Supported configurations

| Parameter | Supported value |
| --- | --- |
| Code | Rotated surface-code memory |
| Basis | Z |
| Distance | 5, 7, 9, or 11 |
| QEC rounds | Equal to the distance |
| Entangling gate | CX |
| Python | CPython 3.12 |

For a physical error rate \(p\) and loss fraction \(L\), the simulator uses

\[
p_{\mathrm{loss}} = Lp,
\qquad
p_{\mathrm{Pauli}} = (1-L)p.
\]

## Calibration

Pre-calibrated parameters for each supported distance are bundled with the
package, so `run-point` works immediately. To generate calibration parameters
for a specific point, inspect the workload first:

```bash
uv run --locked python -m tenkai calibrate \
  --distance 5 \
  --physical-error-rate 0.004 \
  --loss-fraction 0.5 \
  --shots-per-lifecycle 4 \
  --rho-grid 0.5 1.0 \
  --seed 42 \
  --workers 2 \
  --output-dir tenkai-output/calibration \
  --dry-run
```

Remove `--dry-run` to execute the calibration. Interrupted runs resume from
completed lifecycle fragments. The generated artifact can then be supplied to
an exactly matching point:

```bash
uv run --locked python -m tenkai run-point \
  --method tenkai \
  --distance 5 \
  --physical-error-rate 0.004 \
  --loss-fraction 0.5 \
  --shots 16 \
  --routing tenkai-output/calibration/routing/<sha256>.json \
  --output-dir tenkai-output/custom-point
```

## Results and reproducibility

`result.json` records the input configuration, derived loss and Pauli rates,
circuit and calibration identities, accepted shot ranges, logical-error counts,
logical error rates, per-round rates, and confidence intervals.

Every shot is derived from its global shot index and experiment seed. Changing
the worker count or output path does not change the scientific result.

## Development

```bash
uv sync --locked --extra dev
uv run --locked --extra dev python -m pytest -q
uv build
```

## License

Tenkai is licensed under the [Apache License 2.0](LICENSE).

## Citation

If you use Tenkai in your research, please cite:

```bibtex
@inproceedings{tao2026tenkai,
  author    = {Tao, Chenning and Zhou, Xiaoyi and Lu, Liqiang and Yin, Jianwei},
  title     = {{Tenkai}: Accurate Syndrome Extraction Circuit Design and Shadow-Guided Decoder for Tackling Atom Loss on Surface Code},
  booktitle = {Proceedings of the IEEE/ACM International Symposium on Microarchitecture ({MICRO})},
  year      = {2026}
}
```
