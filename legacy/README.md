# Legacy implementation

This directory preserves the original SAM v1/two-Conda-environment prototype,
copy-based upstream patches, and its deployment scripts for reproducibility.

It is not the maintained runtime path. In particular, the deployment scripts
can overwrite files inside third-party repositories. Use the packaged entry
point documented in the repository root instead:

```bash
source scripts/activate_click_to_model.sh
python -m click_to_model --help
```
