#!/bin/env python

import streamlit as st
import numpy as np
<<<<<<< Updated upstream


import streamlit as st
import numpy as np

import streamlit as st
import numpy as np
import pandas as pd
import streamlit as st
import time

'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)
=======
from app.core.system import ArtifactRegistry
from autoop.core.database import Database
from autoop.core.storage import LocalStorage
from autoop.core.ml.dataset import Dataset

storage = LocalStorage()
database = Database(storage)
registry = ArtifactRegistry(
    storage=storage,
    database=database
)

dataset = Dataset()

>>>>>>> Stashed changes

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'