# VolgaIT 2026 PlateVision

Offline Russian vehicle plate recognition research project for the Volga-IT semifinal.

## Supported classes
- `type1`: standard single-row white plates
- `type1a`: two-row white plates
- `type1b`: yellow passenger-transport plates
- `other`: non-target signs

## Included
- `src/infer.py` — inference CLI
- `src/train.py` — detector training script
- `dataset/generator/generate.py` — reproducible synthetic sample generator
- `docs/EXPLANATORY_NOTE.md` — architecture and limitations

## Status
**Engineering baseline only.** The repository does not yet contain trained model weights, a complete licensed real-world dataset, or independently validated accuracy and latency results. Do not describe it as a competition-ready trained submission.

## Submission requirements
Provide trained weights, complete legally shareable dataset with `meta.csv` and labels, license/source records, official validator report, and benchmark results. The inference target is a semicolon-separated UTF-8 CSV with `image;plate_num;plate_type;confidence`.

See the Python sources for interfaces. Offline execution may require local dependencies and model files.
