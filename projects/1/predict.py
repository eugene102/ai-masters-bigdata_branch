#!/opt/conda/envs/dsenv/bin/python

import sys, os
import logging
from joblib import load
import pandas as pd

sys.path.append('.')
from model import fields, fields_types, col_deleter

del fields[fields.index('label')]
del fields_types['label']

#
# Init the logger
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#load the model
model = load("tut1.joblib")

#fields = """doc_id,hotel_name,hotel_url,street,city,state,country,zip,class,price,
#num_reviews,CLEANLINESS,ROOM,SERVICE,LOCATION,VALUE,COMFORT,overall_ratingsource""".replace("\n",'').split(",")

#read and infere
read_opts=dict(
        sep='\t', names=fields, index_col=False, header=None,
        iterator=True, chunksize=100, dtype=fields_types
)

for df in pd.read_csv(sys.stdin, **read_opts):
    temp = df.id
    col_deleter.transform(df)
    pred = model.predict_proba(df).iloc[:, 1]
    out = zip(temp, pred)
    print("\n".join(["{0}\t{1}".format(*i) for i in out]))

