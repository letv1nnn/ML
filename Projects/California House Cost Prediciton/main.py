import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import tarfile
import urllib.request

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (OrdinalEncoder, OneHotEncoder,
                                   MinMaxScaler, StandardScaler,
                                   FunctionTransformer)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------------------------------------------------|

# LOADING DATA

# ---------------------------------------------------------------------------------------------------------------------|

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

# ---------------------------------------------------------------------------------------------------------------------|

# EXPLORE DATA AND GAIN INSIGHTS

# ---------------------------------------------------------------------------------------------------------------------|

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


# scatter matrix that plots every numerical attribute against every other numerical attribute.
def visualize_correlations():
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    plt.show()

# visualize_correlations()

# ---------------------------------------------------------------------------------------------------------------------|

# EXPERIMENT WITH ATTRIBUTE COMBINATIONS AND PREPARE DATA FOR MACHINE LEARNING ALGORITHMS

# ---------------------------------------------------------------------------------------------------------------------|

housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]


housing = housing.drop(columns=["median_house_value"], axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

imputer = SimpleImputer(strategy="median")
housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
# print(imputer.statistics_)
# print(housing_num.median().values)

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
housing_cat = housing[["ocean_proximity"]]

ordinal_encoder = OrdinalEncoder()
housing_cat_encoding = ordinal_encoder.fit_transform(housing_cat)

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# print(housing_cat_1hot) sparse matrix, really efficient way while using OneHotEncoding method
housing_cat_1hot = housing_cat_1hot.toarray()
# print(housing_cat_1hot)

# Data scaling
min_max_scaler = MinMaxScaler()
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

# scaling the data using standardisation
std_scaler = StandardScaler(with_mean=False) # with mean is equal to false for sparse matrices
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

# Using radial bias function to find housing ages that are close to 35
age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)

# Scaling the target
target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

# Creating a Linear Regression model
model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)
some_new_data = housing[["median_income"]].iloc[:5]

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)

# Alternatively
model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)

# Creating a log-transformer and applying it to the population feature
log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])


# Custom transformer that acts much as like the Standard Scaler
class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean: bool=True):
        self.with_mean = with_mean

    def fit(self, X, y=None):
        X = check_array(X)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_
    # add get_features_names_out() and inverse_transform() methods


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]], sample_weight=housing_labels)


# Many data transformation steps that need to be executed in the right order.
# Pipeline class helps with such sequences of transformation.
# Here is a small pipeline for numerical attributes, which will first impute then scale the input features
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])
# Alternatively, it's possible to use make_pipeline function instead

housing_num_prepared = num_pipeline.fit_transform(housing_num)

# ---------------------------------------------------------------------------------------------------------------------|


