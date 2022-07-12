# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 18:38:07 2021

@author: user
"""

import os
import numpy as np
import scipy
import pandas as pd
from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi
import copy
from collections import Counter

# Flag to query neurons from the database
DB_ACCESS = False # Set True to query 

#%%

if DB_ACCESS:
    ctc = CellTypesCache()
    
    cells = ctc.get_cells(species=[CellTypesApi.MOUSE], require_reconstruction=True)
    
    # 509 cells total
    
    for i in range(len(cells)):
        morphology = ctc.get_reconstruction(cells[i]['id']) 
    

