"""运行独立 embedding → SID → Transformer 生成召回主流程。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_demo import main


if __name__ == "__main__":
    main()
