"""A standalone SVG per frame - the same construction as `structureforge/api/static/app.js`'s
canvas (a flipped-y group so geometry-up renders as image-up, one path per layer with evenodd
fill for holes), so a script or notebook can look at a simulation result without the web GUI.
"""

from __future__ import annotations

from ..process.simulate import Frame


def _ring_path(ring: dict) -> str:
    def seg(points: list[list[float]]) -> str:
        return "M " + " L ".join(f"{x},{y}" for x, y in points) + " Z"

    d = seg(ring["exterior"])
    for hole in ring["holes"]:
        d += " " + seg(hole)
    return d


def frame_to_svg(frame: Frame, material_colors: dict[str, str], margin_nm: float = 5.0) -> str:
    """A self-contained `<svg>` document showing one frame, y-up (positive y renders at the top,
    matching how these cross-sections are normally drawn - substrate at the bottom, growth up).
    """
    layers = frame.layers
    xs = [0.0, frame.domain_width_nm]
    ys: list[float] = [0.0]
    for layer in layers:
        for ring in layer.rings():
            for _, y in ring["exterior"]:
                ys.append(y)
    y0, y1 = min(ys) - margin_nm, max(ys) + margin_nm
    x0, x1 = -margin_nm, frame.domain_width_nm + margin_nm

    paths = []
    for layer in layers:
        rings = layer.rings()
        if not rings:
            continue
        d = " ".join(_ring_path(r) for r in rings)
        color = material_colors.get(layer.material, "#999999")
        paths.append(f'<path d="{d}" fill="{color}" fill-rule="evenodd" stroke="rgba(0,0,0,0.25)" stroke-width="{(x1 - x0) * 0.002:g}"><title>{layer.material}</title></path>')

    width, height = x1 - x0, y1 - y0
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0} {-y1} {width} {height}" '
        f'width="800" height="{800 * height / width:.0f}">'
        f'<g transform="scale(1,-1)">{"".join(paths)}</g>'
        f"</svg>"
    )


def save_svg(path: str, frame: Frame, material_colors: dict[str, str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(frame_to_svg(frame, material_colors))
