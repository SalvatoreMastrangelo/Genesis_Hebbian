from __future__ import annotations

import ast
import importlib.util
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

DATA_PROCESSING_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = DATA_PROCESSING_DIR.parent
