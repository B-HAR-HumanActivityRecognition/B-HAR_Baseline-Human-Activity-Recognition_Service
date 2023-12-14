import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN, KMeansSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss, EditedNearestNeighbours
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import LinearSVC, SVR


def drop_data(
        data: [pd.DataFrame],
        info: [],
        to_drop: dict
):
    """
    Drop some information from the dataset, these info can be label values, subject or sessions
    :param data: the dataset
    :param info: the labels, subjects and/or sessions
    :param to_drop: a dictionary containing the info to be dropped, for example {'label': ['Walking', 'Sitting']}
    :return: dataset and info without unwanted ones
    """
    # Rebuild the dataframe
    merged_data = pd.concat(data)
    merged_info = pd.concat([pd.concat(info_i) for info_i in info if len(info_i) > 0])
    merged_info.index = merged_data.index

    # Keep track of information
    info_columns = list(merged_info.columns)

    # Merge data and info
    merged = pd.concat([merged_data, merged_info], axis=1)

    # Free up space
    del merged_data, merged_info

    # Drop unwanted values
    for drop_key in to_drop.keys():
        for drop_val in list(to_drop[drop_key]):
            merged = merged[merged[drop_key] != drop_val]

    selected_info = merged[info_columns]
    selected_data = merged.drop(info_columns, axis=1)

    return selected_data, selected_info


def train_test_splitting(
        data: [pd.DataFrame],
        info: [],
        method: str = 'normal',
        test_size: float = .2,
        filter_dict: dict = None
):
    """
    Split the dataset into two subset one for training and one for testing
    :param data: sensor data
    :param info: labels, subjects, and/or sessions
    :param method: the methodology to accomplish the train test split, can be: normal, intra or leave-out
    :param test_size: used when method is set to 'normal', it represents the percentage of data in the test set
    :param filter_dict: used when method is set to 'leave-out', put in train set the following rows
                        {'subject': [194], 'label': ['Running', 'Stairs']}
    :return:
    """
    if method == 'normal':
        if type(data) is list:
            data = pd.concat(data)
            info = pd.concat([pd.concat(info_i) for info_i in info if len(info) > 0])

        X_train, X_test, y_train, y_test = train_test_split(data, info, test_size=test_size, random_state=28)
        return X_train, X_test, y_train, y_test

    elif method == 'intra':
        pass

    elif method == 'leave-out':
        X_train, X_test, y_train, y_test = __leave_out(data, info, leave_out=filter_dict)
        return X_train, X_test, y_train, y_test


def normalization(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        n_data_cols: int,
        method: str = 'min-max',
        representation: str = 'segmentation'
):
    """
    Normalise the data using training set as reference
    :param train_df: training dataset
    :param test_df: testing dataset
    :param segment_length: number of sample in a time window (sf x tw_duration) [Hz x sec]
    :param n_data_cols: number of columns containing data
    :param method: normalization method, {min-max, robust, standard}
    :param representation: data representation chosen {segmentation, features, raw}
    :return:
    """
    if representation == 'segmentation':
        # Reshape to normalize
        tmp_train = train_df.values.reshape((-1, n_data_cols))
        tmp_test = test_df.values.reshape((-1, n_data_cols))

        # Normalize the data
        tmp_train, tmp_test = __normalizer(tmp_train, tmp_test, method)

        # Reshape back
        train_df = pd.DataFrame(tmp_train.reshape(train_df.shape), columns=train_df.columns)
        test_df = pd.DataFrame(tmp_test.reshape(test_df.shape), columns=test_df.columns)

    else:
        # No reshape needed in case of raw data or features, normalize as is
        train_df, tmp_test = __normalizer(train_df, test_df, method)

    return train_df, test_df


def features_selection(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        method: str,
        train_info: pd.DataFrame = None
):

    if method == 'variance':
        # Remove low variance features
        selector = VarianceThreshold()
        train_df = selector.fit_transform(train_df)
        test_df = selector.transform(test_df)

    elif method == 'l1':
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(train_df, train_info)
        model = SelectFromModel(lsvc, prefit=True)
        train_df = model.fit_transform(train_df)
        test_df = model.transform(test_df)

    elif method == 'tree-based':
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(train_df, train_info)
        model = SelectFromModel(clf, prefit=True)
        train_df = model.fit_transform(train_df)
        test_df = model.transform(test_df)

    elif method == 'recursive':
        estimator = SVR(kernel="linear")
        selector = RFE(estimator, n_features_to_select=5, step=1)
        train_df = selector.fit_transform(train_df, train_info)
        test_df = selector.transform(test_df)

    return train_df, test_df


def balancing(
        train_df: pd.DataFrame,
        train_info: pd.DataFrame,
        method: str
):
    available_methods = {
        # Under-sample
        'random_under': RandomUnderSampler(),
        'near_miss': NearMiss(),
        'edited_nn': EditedNearestNeighbours(),

        # Over-sampling
        'smote': SMOTE(),
        'adasyn': ADASYN(),
        'kmeans_smote': KMeansSMOTE(k_neighbors=3),
        'random_over': RandomOverSampler()
    }

    if method in available_methods.keys():
        train_df, train_info = available_methods[method].fit_resample(train_df, train_info)

    return train_df, train_info


def __leave_out(x: pd.DataFrame, y: pd.DataFrame, leave_out: dict):
    if type(x) is list:
        x = pd.concat(x)
        y = pd.concat([pd.concat(info_i) for info_i in y if len(y) > 0])

    # Reset index
    x.index = range(x.shape[0])
    y.index = range(y.shape[0])

    df = pd.concat([x, y], axis=1)
    info_cols = list(y.columns)

    # Create filter dataframe
    filter_df = pd.DataFrame.from_dict(leave_out.values()).T
    filter_df.columns = list(leave_out.keys())
    filter_df.ffill(inplace=True)

    # Use merge to filter from both inclusion and exclusion
    test_df = df.merge(filter_df)
    train_df = df.merge(filter_df, how='left', indicator=True)
    train_df = train_df[train_df['_merge'] == 'left_only']
    train_df.drop('_merge', axis=1, inplace=True)

    train_info = train_df[info_cols]
    test_info = test_df[info_cols]
    train_df.drop(info_cols, axis=1, inplace=True)
    test_df.drop(info_cols, axis=1, inplace=True)

    return train_df, test_df, train_info, test_info


def __normalizer(train, test, method):
    scaler = None
    if method == 'min-max':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'standard':
        scaler = StandardScaler()

    if scaler is not None:
        train = scaler.fit_transform(train)
        test = scaler.transform(test)

    return train, test
