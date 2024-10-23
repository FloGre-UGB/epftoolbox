import pandas as pd
import numpy as np
import argparse
import os

from epftoolbox.data import read_data
from epftoolbox.evaluation import MAE, sMAPE
from epftoolbox.models import LEAR

#%%
os.chdir('C:/Users/flori/github/epftoolbox/examples')

data = pd.read_excel('C:/Users/flori/github/epftoolbox/custom_data/Input.xlsx')

#%%

#%%
