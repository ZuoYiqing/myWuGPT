"""Generate architecture assets for reports."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src import export_onnx  # noqa: E402
from src import export_drawio_png  # noqa: E402


DRAWIO_XML = """<mxfile host="app.diagrams.net" modified="2025-01-01T00:00:00.000Z" agent="Codex" version="20.8.23" type="device">
  <diagram id="moe-routing" name="Page-1">
    <mxGraphModel dx="1400" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1600" pageHeight="900" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        <mxCell id="input" value="Input token x" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="40" y="180" width="120" height="50" as="geometry"/>
        </mxCell>
        <mxCell id="router" value="Router&#xa;Linear(n_embd -&gt; E)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F0FE;strokeColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="210" y="160" width="160" height="80" as="geometry"/>
        </mxCell>
        <mxCell id="softmax" value="Softmax&#xa;probabilities over E" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF7E6;strokeColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="410" y="160" width="170" height="80" as="geometry"/>
        </mxCell>
        <mxCell id="top2" value="Top-2 gating" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F1F8E9;strokeColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="610" y="160" width="140" height="80" as="geometry"/>
        </mxCell>
        <mxCell id="dispatch" value="Dispatch" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="790" y="160" width="120" height="80" as="geometry"/>
        </mxCell>
        <mxCell id="expert1" value="Expert FFN 1" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="960" y="60" width="160" height="60" as="geometry"/>
        </mxCell>
        <mxCell id="expert2" value="Expert FFN 2" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="960" y="150" width="160" height="60" as="geometry"/>
        </mxCell>
        <mxCell id="expert3" value="Expert FFN 3" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="960" y="240" width="160" height="60" as="geometry"/>
        </mxCell>
        <mxCell id="expert4" value="Expert FFN 4" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="960" y="330" width="160" height="60" as="geometry"/>
        </mxCell>
        <mxCell id="combine" value="Combine&#xa;(weighted sum)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF3E0;strokeColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="1160" y="160" width="150" height="80" as="geometry"/>
        </mxCell>
        <mxCell id="output" value="Output" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="1330" y="180" width="100" height="50" as="geometry"/>
        </mxCell>
        <mxCell id="note" value="E=4, Top-2&#xa;避免坍缩：关注 expert share / entropy / max_share" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFDE7;strokeColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="40" y="300" width="300" height="80" as="geometry"/>
        </mxCell>
        <mxCell id="legend" value="Legend:&#xa;Green = selected paths&#xa;Gray dashed = other paths" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F5F5F5;strokeColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="40" y="400" width="300" height="90" as="geometry"/>
        </mxCell>
        <mxCell id="e1" style="endArrow=block;html=1;strokeColor=#000000;" edge="1" parent="1" source="input" target="router">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="e2" style="endArrow=block;html=1;strokeColor=#000000;" edge="1" parent="1" source="router" target="softmax">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="e3" style="endArrow=block;html=1;strokeColor=#000000;" edge="1" parent="1" source="softmax" target="top2">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="e4" style="endArrow=block;html=1;strokeColor=#000000;" edge="1" parent="1" source="top2" target="dispatch">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="e5" style="endArrow=block;html=1;strokeColor=#2E7D32;strokeWidth=2;" edge="1" parent="1" source="dispatch" target="expert1">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="e6" style="endArrow=block;html=1;strokeColor=#B0B0B0;dashed=1;" edge="1" parent="1" source="dispatch" target="expert2">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="e7" style="endArrow=block;html=1;strokeColor=#2E7D32;strokeWidth=2;" edge="1" parent="1" source="dispatch" target="expert3">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="e8" style="endArrow=block;html=1;strokeColor=#B0B0B0;dashed=1;" edge="1" parent="1" source="dispatch" target="expert4">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="e9" style="endArrow=block;html=1;strokeColor=#2E7D32;strokeWidth=2;" edge="1" parent="1" source="expert1" target="combine">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="e10" style="endArrow=block;html=1;strokeColor=#B0B0B0;dashed=1;" edge="1" parent="1" source="expert2" target="combine">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="e11" style="endArrow=block;html=1;strokeColor=#2E7D32;strokeWidth=2;" edge="1" parent="1" source="expert3" target="combine">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="e12" style="endArrow=block;html=1;strokeColor=#B0B0B0;dashed=1;" edge="1" parent="1" source="expert4" target="combine">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="e13" style="endArrow=block;html=1;strokeColor=#000000;" edge="1" parent="1" source="combine" target="output">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""


def write_drawio(path: Path) -> None:
    path.write_text(DRAWIO_XML, encoding="utf-8")


def write_readme(path: Path) -> None:
    text = (
        "# Model Graph Assets\n\n"
        "Netron screenshot steps:\n"
        "1) Install Netron (https://netron.app) and open it.\n"
        "2) File -> Open -> select `assets/model.onnx`.\n"
        "3) Capture the graph screenshot for your report.\n\n"
        "No Netron option:\n"
        "- Open `assets/model_graph.txt` and use it as a lightweight structural screenshot alternative.\n\n"
        "MoE routing diagram:\n"
        "- Open `assets/moe_routing.drawio` in draw.io (diagrams.net) and export to PNG if needed.\n"
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate architecture assets")
    parser.add_argument("--export-png", action="store_true", help="Try to export draw.io PNG")
    args = parser.parse_args()

    assets_dir = Path(ROOT_DIR) / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    drawio_path = assets_dir / "moe_routing.drawio"
    write_drawio(drawio_path)

    onnx_path, graph_path = export_onnx.export_assets(assets_dir)

    readme_path = assets_dir / "README_graph.md"
    write_readme(readme_path)

    png_path = assets_dir / "moe_routing.png"
    png_exported = False
    if args.export_png:
        png_exported = export_drawio_png.export_drawio_png(drawio_path, png_path)

    print("Generated assets:")
    print(f"- {drawio_path}")
    if png_exported:
        print(f"- {png_path}")
    else:
        print(f"- {png_path} (manual export if needed)")
    print(f"- {onnx_path}")
    print(f"- {graph_path}")
    print(f"- {readme_path}")


if __name__ == "__main__":
    main()
