"""The embedded VTK view (pyvistaqt's ``QtInteractor``) every MeshDraw widget uses, in the CAD style."""

from __future__ import annotations

try:  # pragma: no cover - environment dependent
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except Exception:  # pragma: no cover
    pv = None
    PYVISTA_AVAILABLE = False

try:  # pragma: no cover - environment dependent
    from pyvistaqt import QtInteractor

    PYVISTAQT_AVAILABLE = True
except Exception:  # pragma: no cover
    QtInteractor = None
    PYVISTAQT_AVAILABLE = False


def make_interactor(parent):
    """The embedded VTK view, in the CAD style (``style.style_plotter``).

    pyvistaqt's defaults are wrong for an embedded, event-driven view:
    ``auto_update=5.0`` re-renders the whole scene every 200 ms forever (even
    hidden, even with nothing changed). Here nothing renders unless something
    changed or the user moves the camera. Anti-aliasing is this machine's
    choice (``style.antialiasing``): 4x MSAA by default -- crisp edges and
    silhouettes, its framebuffers sized once per resize -- FXAA (softer,
    free on resize) or none.
    """
    from meshdraw.style import MSAA_SAMPLES, antialiasing, style_plotter

    aa = antialiasing()
    plotter = QtInteractor(parent, auto_update=False, multi_samples=MSAA_SAMPLES if aa == "msaa" else 0)
    try:
        style_plotter(plotter, aa=aa)
    except Exception:  # pragma: no cover - renderer dependent
        pass
    return plotter


__all__ = ["PYVISTAQT_AVAILABLE", "PYVISTA_AVAILABLE", "make_interactor"]
