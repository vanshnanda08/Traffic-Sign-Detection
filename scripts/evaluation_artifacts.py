"""Render only stored, complete executed evaluator outputs; never run inference."""
import os
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parents[1]/'.cache/matplotlib')
from p57.evaluation_artifacts import make_artifacts
if __name__=='__main__':make_artifacts()
