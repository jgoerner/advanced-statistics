# Imports
import os
import pandas as pd

# Config
os.chdir("/home/jovyan/work")

# Calculations
df_mtcars = pd.read_csv("./data/mtcars.csv")
df_mtcars.cov()