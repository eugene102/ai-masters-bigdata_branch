#!/opt/conda/envs/dsenv/bin/python
from model import model, fields, col_deleter

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from joblib import dump
import numpy as np

import os, sys
import logging

logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))


try:
  proj_id = sys.argv[1] 
  train_path = sys.argv[2]
except:
  logging.critical("Need to pass both project_id and train dataset path")
  sys.exit(1)


logging.info(f"TRAIN_ID {proj_id}")
logging.info(f"TRAIN_PATH {train_path}")

read_table_opts = dict(sep="\t", names=fields, index_col=False)
df = pd.read_table(train_path, **read_table_opts)

logging.info(f"Splitting X and y...")
y = df.label
df.drop(columns=['label'], inplace=True)

col_deleter.transform(df)
print(df.columns)

logging.info(f"Reducing dataset")
# mask = np.random.rand(df.shape[0]) < 0.5
# df = df.loc[mask, :]
# y = y.loc[mask]
logging.info(f"Splitting to train and test..")


mask = np.random.rand(df.shape[0]) < 0.95

logging.info(f"Check if all cat_fea")

logging.info(f"Learning...")
model.fit(df.loc[mask, :], y.loc[mask])

logging.info(f"Calculating score")
model_score = log_loss(y.loc[~mask], model.predict_proba(df.loc[~mask, :]))

logging.info(f"model score: {model_score:.3f}")

# save the model
dump(model, "{}.joblib".format(proj_id))