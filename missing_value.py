import pandas as pd
import numpy as np


def missing_value(dataset, mv):
    if mv == "zero":
        dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset.replace(to_replace=np.nan, value=0, inplace=True)
