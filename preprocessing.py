import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV


def print_df_metadata(df, id_column=None):
    if not id_column:
        id_column = 'new_id_col'
        df = df.copy()
        df['new_id_col'] = range(1, len(df) + 1)

    df.info()
    for col_name in df.columns:
        if col_name in (id_column):
            continue

        n_values = df[col_name].nunique()
        print("Column: {}  |  Type = {}  |  {} Unique Values  ".format(col_name, df[col_name].dtype.name, n_values))
        if is_numeric_column(df, col_name):
            n_nan = np.count_nonzero(np.isnan(df[col_name]))
            n_neg = np.count_nonzero(df[col_name] < 0)
            print('\t Negative Count = {}  |  NaN count = {}'.format(n_neg, n_nan))
        else:
            print("\t" + str(df.groupby(col_name)[id_column].nunique()).replace("\n", "\n\t"))
        # df[col_name].value_counts()


def count_value_pairs_between_columns(df, col1, col2, cond_on_1=None, cond_on_2=None):
    df2 = df[[col1, col2]]

    if cond_on_1:
        df2 = df2[cond_on_1(df[col1])]
    if cond_on_2:
        df2 = df2[cond_on_2(df2[col2])]

    col1, col2 = df[col1], df[col2]
    df2.insert(0, "new_col", list(zip(col1, col2)))
    return df2["new_col"].value_counts().sort_index()


def add_column_by_f_on_columns(df, new_col_name, f, *col_names):
    cols = []
    for name in col_names:
        cols.append(df[name])

    df[new_col_name] = f(*cols)

    # df["experience"] = f(np.maximum(df["age"].values - df["education.num"].values))


def extract_target_column(df, target_name):
    # target = (df["income"] == ">50K") * 1
    # df.drop("income", axis=1, inplace=True)
    target = df[target_name]
    df.drop(target_name, axis=1, inplace=True)
    return target


def normalize_numeric_columns(df):
    for col_name in df.columns:
        if not is_numeric_column(df, col_name):
            col = df[col_name]
            df[col_name] = (col - col.min()) / col.std()


def one_11hot(df, drop_origin_columns=True):
    for col_name in df.columns:
        if not is_numeric_column(df, col_name):
            df[col_name] = pd.Categorical(df[col_name])
            dfDummies = pd.get_dummies(df[col_name], prefix=col_name).astype(int, False)
            df = pd.concat([dfDummies, df], axis=1)
            if drop_origin_columns:
                df.drop(col_name, axis=1, inplace=True)

    return df


def get_train_test_random_mask(N, part=.2):
    if isinstance(N, pd.DataFrame):
        N = N.shape[0]

    np.random.seed(999)
    df = pd.DataFrame(np.random.randn(N, 2))
    return np.random.rand(len(df)) < part


def is_numeric_column(df, col: [int, str, pd.DataFrame]):
    if isinstance(col, str):
        col = df[col]
    elif isinstance(col, int):
        col = df.iloc[:, col]

    return np.issubdtype(col.dtype, np.number)


def get_train_test_data(df, target):
    from sklearn.model_selection import train_test_split

    X = df.values
    y = target.values

    X, X_test, y, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    return X, X_test, y, y_test


def drop_columns(df, *col_names):
    for name in col_names:
        df.drop(name, axis=1, inplace=True)


def get_train_test_data_split_on_col_value(df, target, col_name, col_value):
    yes_indices = df[col_name] == col_value
    no_indices = df[col_name] != col_value

    set_1 = (df[yes_indices], target[yes_indices])
    set_2 = (df[no_indices], target[no_indices])

    return set_1, set_2


def train_xgb_classifier_with_parameters(X, X_test, y, y_test, classifier_parameters=None, fit_parameters=None):
    from xgboost.sklearn import XGBClassifier

    if not classifier_parameters:
        classifier_parameters = {"max_depth": 5,
                                 "min_child_weight": 1,
                                 "learning_rate": 0.1,
                                 "n_estimators": 500,
                                 "n_jobs": -1}

    if not fit_parameters:
        fit_parameters = {}

    clf = XGBClassifier(**classifier_parameters)
    clf.fit(X, y, **fit_parameters)
    return clf.predict(X_test)


def recursive_xgb_param_search(reg, X, y, recursion_iter=2):
    param_grid = {
        'max_depth': np.array(list(range(1, 5))),
        'learning_rate': 10 ** (np.array(range(-10, 0))),
        'subsample': np.array([0.4, 0.6, 0.8, 1.0]),
        'colsample_bytree': np.array([0.4, 0.6, 0.8, 1.]),
        'colsample_bylevel': np.array([0.4, 0.6, 0.8, 1.]),
        'min_child_weight': .01 * np.array(list(range(1, 100, 20)), dtype=float),
        'gamma': np.array(list(range(0, 11, 2)), dtype=float),
        'reg_lambda': [0, .5, 1],
        'n_estimators': 5 * 10 ** (np.array(range(1, 4)))
    }

    param_grid_recursion_op = {
        'max_depth': lambda best: int(np.maximum(1, [best - 1, best, best + 1])),
        'learning_rate': lambda best: best * np.array([.2, .5, 1., 1.5, 2., 4.]),
        'subsample': lambda best: best * np.array([0.1, 0.5, 1.0, 1.5, 2.]),
        'colsample_bytree': lambda best: np.unique(np.minimum(1., best * np.array([0.1, 0.5, 1.0, 1.5, 1.]))),
        'colsample_bylevel': lambda best: np.unique(np.minimum(1., best * np.array([0.1, 0.5, 1.0, 1.5, 1.]))),
        'min_child_weight': lambda best: best * np.array([0.1, 0.5, 1.0, 1.5, 2.]),
        'gamma': lambda best: best * np.array([0.1, 0.5, 1.0, 1.5, 2.]),
        'reg_lambda': lambda best: best * np.array([.2, .5, 1., 1.5, 2., 4.]),
        'n_estimators': lambda best: int(np.maximum(1, (best * np.array([0.1, 0.5, 1.0, 2, 3.])).astype(np.int64)))
    }

    for i in range(recursion_iter):
        rs_regr = RandomizedSearchCV(reg, param_grid, n_iter=100, refit=True, random_state=42, scoring="accuracy")
        rs_regr.fit(X, y)

        param_grid = rs_regr.best_params_.copy()
        for par in param_grid_recursion_op:
            param_grid[par] = param_grid_recursion_op[par](param_grid[par])

    return rs_regr.best_params_


def get_stacked_combiner_classifier(X, y, combiner, *stacked_classifiers):
    def stack_transform(X):
        results = pd.DataFrame()
        for i, clf in enumerate(stacked_classifiers):
            result = clf.predict(X)
            results.insert(0, "classifier_" + str(i + 1), result)

        return np.concatenate([X, results.values], axis=1)

    combiner.stack_transform = stack_transform
    combiner.stack_transform_and_fit = lambda X, y: combiner.fit(combiner.stack_transform(X), y)
    combiner.stack_transform_and_predict = lambda X: combiner.predict(combiner.stack_transform(X))

    combiner.stack_transform_and_fit(X, y)

    return combiner


def bin_data_in_column(df, col_name, conditions_dict, new_col_name, remove_col=True, default_value="default"):
    df.insert(df.columns.get_loc(col_name), new_col_name, default_value)
    for data_val in conditions_dict:
        range = conditions_dict[data_val]
        df[new_col_name][(df[col_name] >= range[0]) & (df[col_name] < range[1])] = data_val

    df[new_col_name] = df[new_col_name].astype(np.str)

    if remove_col:
        df.drop(col_name, axis=1, inplace=True)


def get_ratios_of_classes(df, column_name, target_col_name):
    v = df.groupby(column_name)[target_col_name].value_counts().unstack()
    v.fillna(0, inplace=True)
    v.insert(0, "tot", v.iloc[:, 0] + v.iloc[:, 1])
    return v.iloc[:, 2] / v.iloc[:, 0]


class OrderByDisributionTransformer:
    """
    Transforms values in columns to values ordered by their relative probability.
    """

    def __init__(self, target_col, supported_column_values, temp_col_name="temp_col", only_non_numeric_columns=True):
        self.target_col = target_col.copy()
        self.temp_col_name = temp_col_name
        self.only_non_numeric_columns = only_non_numeric_columns
        self._dictionaries = None
        self.supported_column_values = supported_column_values

    def fit(self, df):
        self._dictionaries = {}
        df = df.copy()
        df.insert(df.shape[1], self.temp_col_name, self.target_col)
        for col_name in df.columns:
            if col_name == self.temp_col_name or (self.only_non_numeric_columns and is_numeric_column(df, col_name)):
                continue
            ratios = get_ratios_of_classes(df, col_name, self.temp_col_name)
            if col_name in self.supported_column_values:
                for val in self.supported_column_values[col_name]:
                    if val not in ratios.index:
                        val = pd.Series([0], [val])
                        ratios = ratios.append(val)
            ordered_ratios = ratios.sort_values().index.values
            dict = {}
            for i, val in enumerate(ordered_ratios):
                dict[val] = i
            self._dictionaries[col_name] = dict

    def transform(self, df):
        if not self._dictionaries:
            raise Exception("You must fit before transforming.")

        df_new = df.copy()
        for col in self._dictionaries:
            df_new[col] = df_new[col].apply(self._dictionaries[col].__getitem__)
        return df_new

    def fit_and_transform(self, df):
        self.fit(df)
        return self.transform(df)


def get_all_column_values(df, only_non_numeric_columns=True):
    dict = {}
    for col_name in df.columns:
        if only_non_numeric_columns and is_numeric_column(df, col_name):
            continue
        dict[col_name] = sorted(df[col_name].unique())

    return dict


class PolynomializationTransformer:
    """

    """

    def __init__(self, deg=2, at_position=0):
        self.at_position = at_position
        self.degree = deg
        self._polynomial_parts = None

    def fit(self, df):
        self._polynomial_parts = set()
        for d in range(self.degree):
            poly_parts = self._polynomial_parts.copy()
            for i in range(df.shape[1]):
                if not is_numeric_column(df, i):
                    continue
                if len(self._polynomial_parts) == i:
                    self._polynomial_parts.add((i,))
                else:
                    for poly in poly_parts:
                        self._polynomial_parts.add(tuple(sorted(poly + (i,))))

    def transform(self, df):
        if not self._polynomial_parts:
            raise Exception("You must fit before transforming.")

        df = df.copy()

        col_names = df.columns.values
        for p in self._polynomial_parts:
            p = list(p)
            pname = str(p)
            new_col = np.ones((1, df.shape[0]))
            counter = 2
            while len(p):
                counter -= 1
                new_col *= df[col_names[p.pop()]].values

            if counter <= 0:
                df.insert(self.at_position, pname, new_col[0])

        return df

    def fit_and_transform(self, df):
        self.fit(df)
        return self.transform(df)


class DataSplitClassiffier:

    def __init__(self, class_restricted_to, class_other_than, col_index, split_on_value):
        self.class_restricted_to = class_restricted_to
        self.class_other_than = class_other_than
        self.col_index = col_index
        self.split_on_value = split_on_value

    def fit(self, X, y, fit_params1={}, fit_params2={}):

        X1, X2, y1, y2 = DataSplitClassiffier.split_on_column_value([X, y], self.col_index, self.split_on_value)

        self.class_restricted_to.fit(X1.values, y1.values, **fit_params1)
        self.class_other_than.fit(X2.values, y2.values, **fit_params2)

    @staticmethod
    def split_on_column_value(data, col_index, split_on_value):
        ret = []
        indices = not_indices = None
        for X in data:
            dfX = pd.DataFrame(X)
            if indices is None:
                indices = dfX.iloc[:, col_index] == split_on_value
                not_indices = ~indices
            ret.append(dfX[indices])
            ret.append(dfX[not_indices])

        return tuple(ret)

    def predict(self, X):
        dfX = pd.DataFrame(X)
        dfX.reset_index()

        indices = dfX[self.col_index] == self.split_on_value
        X1 = dfX[indices].values

        indices = ~indices
        X2 = dfX[indices].values

        y1 = self.class_restricted_to.predict(X1)
        y2 = self.class_other_than.predict(X2)

        y = indices.copy()
        y[indices] = y2
        y[~indices] = y1

        return y


class OneHotTransformer:

    def __init__(self, supported_column_values, drop_original_columns=True):
        self.drop_original_columns = drop_original_columns
        self.supported_column_values = supported_column_values
        self._dictionaries = None

    def fit(self, df):
        self._dictionaries = self.supported_column_values
        for col_name in df.columns:
            if not is_numeric_column(df, col_name):
                old = np.empty(0)
                if self._dictionaries[col_name]:
                    old = self._dictionaries[col_name]
                self._dictionaries[col_name] = sorted(np.union1d(df[col_name].unique(), old))

    def transform(self, df):
        if not self._dictionaries:
            raise Exception("You must fit before transforming.")

        df = df.copy()
        for col_name in df.columns:
            if col_name not in self._dictionaries:
                continue
            pos = df.columns.get_loc(col_name)
            dfDummies = pd.get_dummies(df[col_name], prefix=col_name).astype(int, False)
            dfDummies = dfDummies.reindex(sorted(dfDummies.columns), axis=1)
            # check column order and that nothing is missing
            i = 0
            for val in self._dictionaries[col_name]:
                col_formatted_name = "{}_{}".format(col_name, val)
                if dfDummies.columns[i] != col_formatted_name:
                    dfDummies.insert(0, col_formatted_name, np.zeros((dfDummies.shape[0], 1)))
                i += 1

            for i, c_name in enumerate(dfDummies.columns):
                df.insert(pos + i, c_name, dfDummies[c_name])

            if self.drop_original_columns:
                df.drop(col_name, axis=1, inplace=True)

        return df

    def fit_and_transform(self, df):
        self.fit(df)
        return self.transform(df)