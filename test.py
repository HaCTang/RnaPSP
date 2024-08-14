import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(7, 4),
                  index = pd.date_range('1/1/2020', periods=7),
                  columns = ['A', 'B', 'C', 'D'])
print(df)


print(df.rolling(window=3).mean())
print(df.expanding(min_periods=3).mean())


print(df.rolling(window=3, min_periods=1).mean())    