# MicroCT Segmenter for Windows

Open a reconstructed image series in napari, use nnInteractive to segment bones, and export a CTAn bitmap mask stack beside the original data. The selected dataset defines the slice count, names, dimensions and pixel grid.

## Install and open

Download `MicroCT_Segmenter_v1.2.0_Windows_x64.zip` from [Releases](https://github.com/Tim4toes/microct-segmenter/releases).

1. Extract the release ZIP on a 64-bit Windows 10 or 11 PC. Double-click **Start_MicroCT.bat** in the extracted folder and accept **Install** when prompted. The installer runs as the current Windows user. Python is included. The first setup downloads the locked Windows packages, checks their SHA-256 hashes, then tests the installed environment.
2. Watch the console for download and installation progress. On a fresh installation the packages total 3.78 GB. Updates reuse the packages already downloaded by setup. Allow approximately 15 GB free on the Windows profile drive for setup. Model weights, dataset caches and editable projects need additional space. Interrupted transfers can resume on the next launch.
3. For daily use, run `%LOCALAPPDATA%\MicroCTSegmenter\Start_MicroCT.bat`. To create a desktop shortcut, open `%LOCALAPPDATA%\MicroCTSegmenter` in File Explorer, right-click **Start_MicroCT.bat**, and choose **Send to → Desktop (create shortcut)**. On Windows 11, you may need to choose **Show more options** first.

Local nnInteractive requires a compatible NVIDIA GPU and driver. This release includes PyTorch 2.8.0 with CUDA 12.8. The on-PC setup check reports whether a small CUDA operation ran. A PC without local CUDA can still view images and export saved segmentations when Qt/OpenGL works. Installation on managed lab machines is subject to their application-control policy. IT may need to provide the GPU driver or approve the executable. The setup log is under `%LOCALAPPDATA%\MicroCTSegmenter\logs`.

## Segment and export

1. Choose **Open dataset folder**. Use **Choose one slice** when a folder has multiple series. The panel loads the numbered series in numerical order and verifies slice sizes.
2. Click **Open nnInteractive**, select the original image in the plugin, choose **Local** and your model, and click **Initialize**. The plugin downloads the selected model on first use, or accepts an existing compatible local checkpoint folder. Use its point or scribble tool to refine the segmentation. Use the plugin's next-object control to work on another bone.
3. In the MicroCT panel, check the result layers you want. Give finished layers descriptive names if helpful. Scribble prompts do not appear in the checklist. Choose a combined mask, a folder per layer or a folder per positive label ID. Leave **Label IDs** empty to include all foreground IDs. Complete the current prediction before exporting or saving.
4. Use **Preview current slice / alignment** to inspect the original labels and corrected mask. Choose your Mask coverage settings before export. Export applies any selected shift and dilation once to the editable segmentation.
5. Click **Export CTAn masks**. A source such as `D:\Study\Mouse813\Original` gets a new sibling folder such as `D:\Study\Mouse813\Original_masks_…`. Each original slice, including blank-mask slices, has a matching full-size BMP. BMP names are preserved, including zero padding. Other supported input formats keep their original stems and receive the BMP extension.
6. Use **Save editable project** to resume later. Reopen its `project.json` and locate the original dataset if it has moved. Projects from 1.0 remain compatible. Project files hold label volumes and IDs. A new nnInteractive session is needed for fresh prompts.

A **1-bit BMP** mask is uncompressed, with black background and white foreground. The 8-bit option contains the same decoded pixel values. The exporter checks source hashes, layer shape and spatial alignment, then reads each output BMP back and verifies its pixels, bit depth and BMP pixel-size fields. Each completed export includes `COMPLETE.txt`, `slice_mapping.csv`, `export.json` and `CTAN_IMPORT.txt`. The JSON records installed software versions and the local checkpoint folder when the open plugin session exposes it. This session information describes the time of saving or exporting. A restored project can contain results created in an earlier session.

## Save mask coverage settings

- **Save as default** saves the current X shift, Y shift, dilation and BMP format for new datasets in your Windows account. Until you save a default, new datasets use native pixels (zero shift and dilation) and 1-bit BMP.
- **Save preset…** adds the current values as a named entry in the **Preset** menu. Enter an existing name to replace that preset after confirmation.
- Reopening an editable project restores the coverage values saved with the project. Select **Saved default** or a named preset to apply it to the open project.

## CTAn

Open the original dataset in CTAn. Use **Region of Interest → Load** and select **Bitmap files (*.bmp)**. Open the matching BMP in the exported stack or the relevant per-bone folder. Use the original filename digit count and full-size bitmap import without centring or resizing. Check the overlay on the first, middle and last slices and around thin structures. The bundled CTAN_IMPORT.txt gives the same import steps for each export.

## Large datasets and 16 GB PCs

Leave **Use disk cache for large datasets** enabled and choose a fast local SSD with enough space under **Choose cache location**. The cache stores the full-resolution source on disk. For example, a 4,299 × 2,752 × 2,752 uint8 stack uses about 30.3 GiB for that source alone. The exporter works one slice at a time. nnInteractive also allocates its own image, segmentation and model buffers. Large stacks can exceed 16 GB system RAM even with disk caching. A suitable higher-memory workstation can run the same launcher.

## Updates on PC1 and PC2

Close open MicroCT Segmenter sessions on the PC being updated. Open **Update_MicroCT.cmd** in the installed folder and select **Install from a release ZIP**. Use the full ZIP from a trusted source. You can also run **Start_MicroCT.bat** from a newly extracted release. Both routes set up a separate environment and check it before switching to the new version. A failed installation leaves the active version available. The launcher retains the preceding version for **Roll_Back_MicroCT.cmd**. Allow space for both installed environments during and after an update. Downloads are cached for reuse.

For both PCs to discover the same releases, select **Choose folder** in each PC's update window once. Put future ZIPs named `MicroCT_Segmenter_vX.Y.Z_Windows_x64.zip` in that trusted shared folder. Each PC offers an available newer release at launch and asks before installing. **Check folder now** checks on demand. With no shared folder set, select an update ZIP manually. A disconnected share can be skipped at startup. There is no public update server configured for this release.

Use the release updater for napari, nnInteractive and PyTorch together. Installing a plugin upgrade in napari's console would change just that PC's active environment outside the release lock. The normal launcher works offline after initial setup. Local AI also needs its checkpoint cached or available on disk. Projects, exports, model caches and app preferences live outside the replaceable application environment.

## Included source

This repository contains the app, tests, locked wheel list and Windows launcher build script. The Windows release ZIP also includes a snapshot under **source**. To use the panel inside an existing compatible Python/napari environment, run `python launch_microct.py` from the repository root, or install it there with `python -m pip install --no-deps --no-build-isolation .` and open the MicroCT Segmenter plugin in napari. The portable installer works independently of that environment.

The plugin's installation and hardware documentation is published by the model developers. [nnInteractive for napari](https://github.com/MIC-DKFZ/napari-nninteractive)

The GPU wheel versions are listed by PyTorch. [PyTorch release combinations](https://pytorch.org/get-started/previous-versions/)

The bundled Windows Python is from Astral's python-build-standalone project. [Python runtime releases](https://github.com/astral-sh/python-build-standalone/releases)

## Credits and licence

MicroCT Segmenter uses nnInteractive by the MIC-DKFZ team for interactive segmentation through its napari plugin. MicroCT Segmenter adds a Windows launcher, mask coverage presets and CTAn-compatible export. If you use nnInteractive in research, please cite the original paper. [nnInteractive project and citation](https://github.com/MIC-DKFZ/nnInteractive#citation) · [napari plugin](https://github.com/MIC-DKFZ/napari-nninteractive)

MicroCT Segmenter’s own code is licensed under [Apache License 2.0](LICENSE). The nnInteractive code and napari plugin are also distributed upstream under Apache 2.0. The official pretrained nnInteractive weights are published separately under CC BY-NC-SA 4.0. [Model card and licence](https://huggingface.co/MIC-DKFZ/nnInteractive)
