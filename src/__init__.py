from pathlib import Path
def get_project_root() -> Path:
    return Path(__file__).parent.parent



from src.engine.datasets import *
from src.model.dgm.model import *