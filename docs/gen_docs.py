"""Generate the parts of the documentation that come from the code.

Called from ``conf.py`` on every build, so the dataset and model tables can never drift from the
code: the metadata lives next to the code (``DATASET_INFO`` in each datamodule, ``MODELS_INFO`` in
``SARIAD.models``). The output goes to ``docs/_generated`` (not tracked by git).
"""

import shutil
from pathlib import Path

DOCS = Path(__file__).parent
ROOT = DOCS.parent


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def dataset_table() -> str:
    from SARIAD.datasets import DATASETS_INFO

    rows = ["| Dataset | Anomaly | Normal data | Masks | Source |", "|---|---|---|---|---|"]
    details = []
    for key, info in DATASETS_INFO.items():
        rows.append(
            f"| `{key}` | {_cell(info['anomaly'])} | {_cell(info['normal_data'])} | {_cell(info['masks'])} | [{_cell(info['name'])}]({info['source']}) |"
        )
        lines = [f"**`{key}`**", "", info["summary"], "", f"- **Download:** {info['download']}"]
        if info.get("collections") and info["collections"] != "-":
            lines.append(f"- **Variants:** {info['collections']}")
        if info.get("notes"):
            lines.append(f"- **Notes:** {info['notes']}")
        details.append("\n".join(lines))
    return "\n".join(rows) + "\n\n" + "\n\n".join(details) + "\n"


def model_table() -> str:
    from SARIAD.models import MODELS_INFO

    rows = ["| Model | What it does | Training | Needs | Paper | Code |", "|---|---|---|---|---|---|"]
    for key, info in MODELS_INFO.items():
        rows.append(
            f"| `{key}` | {_cell(info['summary'])} | {_cell(info['training'])} | {_cell(info.get('requires', '-'))} | [link]({info['paper']}) | [link]({info['code']}) |"
        )
    return "\n".join(rows) + "\n"


def generate() -> None:
    out = DOCS / "_generated"
    out.mkdir(exist_ok=True)
    (out / "dataset_table.md").write_text(dataset_table())
    (out / "model_table.md").write_text(model_table())
    static = DOCS / "_static"
    static.mkdir(exist_ok=True)
    shutil.copy(ROOT / "figs" / "overall.svg", static / "overall.svg")


if __name__ == "__main__":
    generate()
