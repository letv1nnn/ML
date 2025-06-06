import pandas as pd
from pathlib import Path
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
    with tarfile.open(tarball_path) as housing_tarball:
        housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()
# QUICK LOOK AT THE DATA STRUCTURE AND THE DATA IN GENERAL
# print(housing.head())
# print(housing.info())
# print(housing["ocean_proximity"].value_counts())
# print(housing.describe())

# DATA VISUALIZATION
def visualize_initial_dataset():
    housing.hist(bins=50, figsize=(12, 8))
    plt.show()

# visualize_initial_dataset()

# SPLITTING DATA INTO TRAIN AND TEST SETS
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# manual implementation
def shuffle_and_split_data(data, test_ration):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ration)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# splitting the median house district into 6 categories
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
def visualize_income_cat():
    housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
    plt.xlabel("Income category")
    plt.ylabel("Number of districts")
    plt.show()

# visualize_income_cat()

# Ensures all important classes are represented in both sets
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

# print(strat_test_set.info())
# Income category proportion int the test set
# print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# copy of the original training set, so I could do various transformations with it without change the initial one
housing = strat_train_set.copy()

# The area of each circle represents the district's population
# and the color represents the price of the house
def visualize_long_lat():
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2,
                 s=housing["population"] / 100, label="population",
                 c="median_house_value", cmap="jet", colorbar=True,
                 legend=True, sharex=False, figsize=(10, 7)
                 )
    plt.show()

# visualize_long_lat()



