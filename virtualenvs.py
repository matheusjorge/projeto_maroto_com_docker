from pandas import __version__ as pd_version
from numpy import __version__ as np_version
from sklearn import __version__ as sk_version

if __name__ == "__main__":
    print(f"Pandas version: {pd_version}")
    print(f"Numpy version: {np_version}")
    print(f"Sklearn version: {sk_version}")