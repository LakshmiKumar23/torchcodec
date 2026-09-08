# ROCm rocDecode and RPP Integration Notes

After PR #1642 lands (which adds rocJPEG pip-wheel layout support), the following
additions are needed for rocDecode and RPP:

## Changes to repair_wheel.py

Add these patterns to the auditwheel exclude list (near `librocjpeg*`):

```python
        # librocdecode: hardware video decoder, same treatment as librocjpeg.
        # Stays in user's ROCm install (_rocm_sdk_core/lib).
        "librocdecode*",
        # librpp: ROCm Performance Primitives for color space conversion.
        # Also stays in the ROCm install.
        "librpp*",
```

## RPATH

The RPATH patch that PR #1642 adds for `_rocm_sdk_core/lib` already covers
rocDecode and RPP since they live in the same directory as rocJPEG.

## Workflow

No additional workflow changes needed — the ROCm SDK init from PR #1642
installs the full `rocm[devel]` package which includes rocDecode and RPP.

## Deleted Files

- `packaging/install_rocdecode.sh` - No longer needed with pip-wheel layout
- `packaging/install_rpp.sh` - No longer needed with pip-wheel layout
