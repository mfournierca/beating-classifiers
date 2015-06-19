from urllib import urlretrieve
from os.path import join, dirname
from zipfile import ZipFile
from pandas import read_csv
from sklearn.cross_validation import train_test_split

DATA_ROOT = join(dirname(__file__), "..", "data")
SOURCE_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
              "spambase/spambase.zip")
DATA_ARCHIVE = join(DATA_ROOT, "spambase.zip")
DATA_DIR = join(DATA_ROOT, "spambase/")


def download_spambase(url=SOURCE_URL, dest=DATA_ARCHIVE):
    print("downloading data set ...")
    urlretrieve(url, dest)
    print("download complete, unzipping data ...")
    with ZipFile(dest) as z:
        z.extractall(DATA_DIR)


def load_spambase(
        data_root=DATA_DIR,
        data_file="spambase.data",
        names_file="spambase.names"):
    names = []
    with open(join(data_root, names_file), "r") as f:
        for i, e in enumerate(f):
            if 32 < i < 90:
                names.append(e[:e.index(":")])
    names.append("spam")
    return read_csv(join(data_root, data_file), header=None, names=names)


def split_spambase(df, test_ratio=0.33, random_seed=1):
    x = df[df.columns - ["spam"]]
    y = df["spam"]
    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=test_ratio, random_state=random_seed)
    return xtrain, xtest, ytrain, ytest
