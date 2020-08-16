import pandas as pd
import numpy as np
pf = pd.DataFrame(np.arange(12).reshape(3,4), columns=['A', 'B', 'C', 'D'])
print(pf)
index_0=pf[pf.A==4].index.tolist()
print(index_0)
pf.drop(index=index_0,inplace=True)
pf.drop(['B', 'C'], axis=1,inplace=True)
print(pf)