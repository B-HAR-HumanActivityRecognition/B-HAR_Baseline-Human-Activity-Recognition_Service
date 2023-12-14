from random import randint
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from modules.representation import __apply_segmentation, __to_compact_form
import tsfel as ts


def correlation_matrix(df: pd.DataFrame, save_to: str):
    df = pd.concat(df)
    corrMatrix = df.corr(numeric_only=True)
    plt.figure(figsize=(int(df.shape[1] * 2.2), df.shape[1] * 2))
    sn.heatmap(corrMatrix, annot=True)
    plt.title('Correlation Matrix')
    plt.savefig('{}/Correlation_Matrix.png'.format(save_to), dpi=200)
    #plt.show()
    plt.close()


def feature_distribution(df: pd.DataFrame, save_to: str):
    df = pd.concat(df)
    plt.title('Features Distribution')
    df['label'].value_counts().plot(kind='barh')
    plt.savefig('{}/Features_Distribution.png'.format(save_to), dpi=200)
    #plt.show()
    plt.close()


def feature_importance(df: pd.DataFrame, save_to: str):
    model = RandomForestClassifier()
    df = pd.concat(df)
    data_columns = [col for col in df.columns if 'data' in col]
    model.fit(df[data_columns], df['label'])

    plt.title('Features Importance')
    (pd.Series(model.feature_importances_, index=data_columns).plot(kind='barh'))
    plt.savefig('{}/Features_Importance.png'.format(save_to), dpi=200)
    #plt.show()
    plt.close()


def _apply_pca(d: pd.DataFrame, labels: list, n_components: int, save_to: str):
    columns = [f'pc_{i}' for i in range(n_components)]
    # PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(d)
    principal_df = pd.DataFrame(
        data=principal_components,
        columns=columns
    )
    principal_df['label'] = labels

    var = pca.explained_variance_ratio_.sum() * 100
    title = f'Total explained variance {var:.2f}%'
    graph = sn.pairplot(principal_df, hue='label')
    graph.fig.suptitle(title)
    plt.savefig('{}/PCA_{}D.png'.format(save_to, n_components), dpi=200)
    #plt.legend()
    #plt.show()
    plt.close()


def _metadata_to_labels_sa(metadata, num_labels, num_cols):
    labels = []
    for label in metadata:
        for lab in label:
            labels.extend([lab['label'].values[0] for i in range(num_labels // num_cols)])

    return labels

def _metadata_to_labels_pca(metadata):
    labels = []
    for label in metadata:
        for lab in label:
            labels.append(lab['label'].values[0])

    return labels


def pca_plot(df: [pd.DataFrame], sampling_frequency: int, segment_duration: float, save_to: str):
    data, metadata = __apply_segmentation(
        df,
        sampling_frequency=sampling_frequency,
        segment_duration=segment_duration,
    )

    cfg = ts.get_features_by_domain()

    # Bring data in the correct shape
    data = __to_compact_form(data)
    n_cols = int(data.shape[1] / (sampling_frequency * segment_duration))
    data = data.values.reshape((-1, n_cols))

    # Convert labels
    labels = _metadata_to_labels_pca(metadata)

    # Start features extraction
    data = ts.time_series_features_extractor(
        cfg,
        data,
        fs=sampling_frequency,
        window_size=int(segment_duration * sampling_frequency),
        overlap=0,
        verbose=0
    )

    # Standardizing the features
    data = StandardScaler().fit_transform(data)

    # PCA 2D and 3D
    for dimension in [2, 3]:
        _apply_pca(data, labels, dimension, save_to)


def sensor_analysis(df: [pd.DataFrame], sampling_frequency: int, segment_duration: float, save_to: str):
    data, metadata = __apply_segmentation(
        df,
        sampling_frequency=sampling_frequency,
        segment_duration=segment_duration,
    )
    data = __to_compact_form(data)
    num_labels = data.shape[1]
    n_cols = int(data.shape[1] / (sampling_frequency * segment_duration))
    data = data.values.reshape((-1, n_cols))

    # Convert labels
    labels = _metadata_to_labels_sa(metadata, num_labels, n_cols)
    sensor_number = 0
    for i in range(n_cols):
        if i % 3 == 0:
            c_names = [f'data_{i}', f'data_{i+1}', f'data_{i+2}']
            _plot_sensor(data[:, i: i+3], labels, c_names, sensor_number, save_to)
            sensor_number += 1


def _plot_sensor(d: pd.DataFrame, labels: list, col_names: list, sensor: int, save_to: str):
    sensor_df = pd.DataFrame(
        data=d,
        columns=col_names
    )
    sensor_df['label'] = labels

    title = f'Sensor {sensor}'
    graph = sn.pairplot(sensor_df, hue='label')
    graph.fig.suptitle(title)
    plt.savefig('{}/Sensor_{}.png'.format(save_to, sensor), dpi=200)
    #plt.legend()
    #plt.show()
    plt.close()
