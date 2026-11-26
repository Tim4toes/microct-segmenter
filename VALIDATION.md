# MicroCT Segmenter validation

## Version 1.2.0

The GUI tests used napari with a virtual X display on Linux. 61 tests passed. The optional test that opens the real nnInteractive plugin was skipped because that plugin was absent from the test environment. The coverage tests verified persistence across app instances, selection of a named preset, both BMP bit depths, the default for new datasets, and restoration of a saved project's own coverage settings. Package versions are pinned to the same builds as in 1.1.0.

## Version 1.1.0

### Export comparison

Versions 1.0 and 1.1.0 loaded the supplied original slice `115813NEM__rec_Tra00001919.bmp` independently. The test applied the same synthetic segmentation to each version, including foreground near image boundaries, then exported with the original Y +1 shift and two XY dilations. For both bit depths the entire BMP files matched byte for byte.

| Output | Dimensions | File size | SHA-256 |
|---|---|---:|---|
| 1-bit BMP | 2752 × 2752 | 946,750 bytes | `a014934fd45beeb00c54a1520a28f8afb279273e0c34052ed39b2c84b9353b0d` |
| 8-bit BMP | 2752 × 2752 | 7,574,582 bytes | `35f8a1c2dab8b2fc43dd6018aa795a244de11351cfd2b362933f704c0a22466f` |

A separate stack test used mixed-case BMP extensions, padded numbering, rectangular slices and blank first and last masks. It checked filenames, dimensions, bit depth, foreground pixel counts and the sibling output location.

### Application and launcher

- 62 Python tests passed on Linux with a virtual X display. Coverage included project round-trips, geometry, alignment, the export checklist, removal of the two former controls and a real nnInteractive widget and scribble layer.
- Three native bootstrap tests passed on Linux. They exercised unpacking and reusing the interpreter, two or more simultaneous first launches and rejection of paths outside the destination.
- Release-manager tests exercised initial setup, update, rollback, interrupted installations, download resumption, checksum failures, release-folder discovery, offline startup lookup and session locking. One test ran the production venv creation, offline pip installation and dependency check with a small fixture package.
- A separate Linux environment with the real local nnInteractive backend and the matching CPU build of PyTorch imported the inference session. The nnInteractive dock recognised the selected test dataset. The plugin's own result-layer method created a layer from known labels, which the exporter used to produce three 37 × 23 BMPs.
- The Windows lock was resolved for CPython 3.12 on x86-64. All 173 dependency wheels, including the CUDA 12.8 PyTorch build, were downloaded, SHA-256 checked, and their wheel tags and declared dependency versions checked. The embedded Python release archive was also checked against its upstream SHA-256. The app wheel and release launcher each have a separate checksum.
- First setup runs a local smoke test for Qt, the napari viewer, the app dock, plugin registration and backend imports before activating the release. It also checks for a CUDA build and tries a small GPU operation when CUDA is available. Setup records this result in the version's ready.json.

### What remains hardware-dependent

Native Windows installation, Windows application-control policy, CUDA predictions on the lab GPUs, work on a 16 GB workstation at full dataset size, and import into CTAn have not been exercised here. The release has no Authenticode signature. Model selection and model weights are managed by nnInteractive. Existing checkpoint folders can be selected in the plugin.
