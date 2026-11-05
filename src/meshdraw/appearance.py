"""What the user changed about each object a 3D view draws: shown or hidden, its opacity, its colour (no Qt).

Objects are keyed by their LABEL (a body mesh's ``member/leaf``, a port's
``port:<name>``), so the choices outlive a re-render: a solve, a section
change or a variable edit redraws the scene and every object comes back
as the user left it. ``Isolate`` hides everything but the chosen objects;
``Show all`` unhides; ``Reset`` forgets every choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

#: the transparency steps the menu offers (opacity)
OPACITIES = ((1.0, "Opaque"), (0.75, "75 %"), (0.5, "50 %"), (0.25, "25 %"), (0.1, "10 %"))


@dataclass
class Choice:
    visible: bool = True
    opacity: Optional[float] = None  # None: the object's own
    color: Optional[str] = None  # None: the object's own


@dataclass
class Appearance:
    choices: Dict[str, Choice] = field(default_factory=dict)

    def get(self, label: str) -> Choice:
        return self.choices.get(label, Choice())

    def _edit(self, label: str) -> Choice:
        return self.choices.setdefault(label, Choice())

    def resolve(self, label: str, color: str, opacity: float) -> Tuple[bool, str, float]:
        """``(visible, color, opacity)`` of an object drawn with its own ``color`` / ``opacity``."""
        c = self.get(label)
        return c.visible, (c.color or color), (float(c.opacity) if c.opacity is not None else float(opacity))

    def hide(self, label: str) -> None:
        self._edit(label).visible = False

    def show(self, label: str) -> None:
        self._edit(label).visible = True

    def set_opacity(self, label: str, opacity: float) -> None:
        self._edit(label).opacity = max(0.02, min(1.0, float(opacity)))

    def set_color(self, label: str, color: str) -> None:
        self._edit(label).color = str(color)

    def isolate(self, keep: Iterable[str], labels: Iterable[str]) -> None:
        """Hide every one of ``labels`` but ``keep``; show those."""
        keep = set(keep)
        for lbl in labels:
            self._edit(lbl).visible = lbl in keep

    def show_all(self) -> None:
        for c in self.choices.values():
            c.visible = True

    def reset(self, label: Optional[str] = None) -> None:
        """Forget the choices for one object, or for all of them."""
        if label is None:
            self.choices.clear()
        else:
            self.choices.pop(label, None)

    def hidden(self, labels: Iterable[str]) -> List[str]:
        """Which of ``labels`` are hidden, in their order."""
        return [lbl for lbl in labels if not self.get(lbl).visible]


def display_name(label: str) -> str:
    """How a menu names an object: ``horn.section`` wall / exterior / port ``horn.p1``."""
    if label.startswith("port:"):
        return f"Port {label[5:]}"
    if label.startswith("air:"):  # a radiating part's open boundary (``subject.air_boxes``)
        member, _, at = label[4:].partition("@")
        return f"{member} open boundary" + (f" at {at}" if at else "")
    if label.startswith("mesh:"):
        return f"{label[5:]} mesh"
    if label.startswith("refine:"):  # a mesh refinement box: refine:<part>:<k> (the 3D editor's: refine:<k>)
        rest = label[7:]
        part, _, k = rest.rpartition(":")
        return f"{part} refinement box {k}".strip() if part else f"refinement box {rest}"
    if label.startswith("ref:"):
        return f"reference {label[4:]}"
    if label.startswith("check:"):
        return "thin spots"
    member, _, leaf = label.rpartition("/")
    names = {"wall": "wall", "exterior": "exterior", "fem_body": "air body", "fem_metal": "metal", "notional": "(notional)"}
    return f"{member} {names.get(leaf, leaf)}".strip() if member else names.get(leaf, leaf)


__all__ = ["Appearance", "Choice", "OPACITIES", "display_name"]
