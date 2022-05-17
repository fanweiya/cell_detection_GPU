import os

import pandas as pd
from imutils.paths import list_files
for i in list(list_files("H:/test")):
    name=i.split(os.sep)[-1]
    if name.endswith(".csv"):
        df=pd.read_csv(i)
        print(i,(df["团属性"][df["团属性"]>1].count())/(df["团属性"].count())*100)