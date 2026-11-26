"""Per-install Windows smoke test. Never downloads a checkpoint or loads a study."""
import importlib.metadata as md
import json
import sys
from pathlib import Path

import numpy as np
from qtpy.QtWidgets import QApplication
import torch
import torchvision
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from napari_nninteractive import nnInteractiveWidget
from npe2 import PluginManifest
import napari

from microct_segmenter import __version__
from microct_segmenter.gui import MicroCTWidget

app = QApplication.instance() or QApplication([])
manifest = PluginManifest.from_distribution("napari-nninteractive")
assert any(w.display_name == "nnInteractive" for w in manifest.contributions.widgets)
viewer = napari.Viewer(show=False)
dock = MicroCTWidget(viewer)
viewer.window.add_dock_widget(dock, name="MicroCT Segmenter")
viewer.add_image(np.zeros((2, 8, 8), dtype=np.uint8), rgb=False)
viewer.close()
dock.cleanup()
if not torch.version.cuda:
    raise RuntimeError("This release requires the CUDA build of PyTorch.")
report = {"app": __version__, "python": sys.version.split()[0],
          "packages": {name: md.version(name) for name in
                       ("napari", "napari-nninteractive", "nninteractive", "torch", "torchvision")},
          "cuda_build": torch.version.cuda, "gpu": None}
try:
    if torch.cuda.is_available():
        # A real kernel also detects unsupported GPU architectures.
        torch.ones(4, device="cuda").sum().item()
        report["gpu"] = torch.cuda.get_device_name()
except RuntimeError as exc:
    report["gpu_error"] = str(exc)
Path(sys.argv[1]).write_text(json.dumps(report, indent=2), encoding="utf-8")
