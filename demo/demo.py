"""Run the experiments described in a YAML file (default: demo.yaml next to this script).

    python demo/demo.py [config.yaml]
"""
import sys
from pathlib import Path

from SARIAD.config.run import main

if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("demo.yaml"))
    raise SystemExit(main(["--config", config]))
