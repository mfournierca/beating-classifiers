from urllib import urlretrieve
from os.path import join, dirname
from zipfile import ZipFile

DATA_ROOT = join(dirname(__file__), "..", "..", "data")
SOURCE_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
              "spambase/spambase.zip")
DATA_ARCHIVE = join(DATA_ROOT, "spambase.zip")
DATA_FOLDER = join(DATA_ROOT, "spambase/")


def download(url=SOURCE_URL, dest=DATA_ARCHIVE):
    print("downloading data set ...")
    urlretrieve(url, dest)
    print("download complete, unzipping data ...")
    with ZipFile(dest) as z:
        z.extractall(DATA_FOLDER)

