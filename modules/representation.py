import pandas as pd
from modules.cleaning import columns_whitelist
from statistics import mode
import tsfel as ts


def dataset_representation(df: [pd.DataFrame],
                           rep_type: str = 'segmentation',
                           sampling_frequency: int = None,
                           segment_duration: float = None,
                           segment_duration_overlap: int = 0,
                           create_transitions: bool = False
                           ):
    if rep_type == 'segmentation':
        data, info = __apply_segmentation(
            df=df,
            sampling_frequency=sampling_frequency,
            segment_duration=segment_duration,
            segment_duration_overlap=segment_duration_overlap,
            create_transitions=create_transitions
        )

    elif rep_type == 'features':
        data, info = __apply_feature_extraction(
            df=df,
            sampling_frequency=sampling_frequency,
            segment_duration=segment_duration,
            create_transitions=create_transitions
        )

    elif rep_type == 'raw':
        df = __to_compact_form(df)
        info_in_data = [col for col in df.columns if col in columns_whitelist]
        if 'timestamp' in info_in_data:
            info_in_data.remove('timestamp')
        data_columns = [col for col in df.columns if col not in columns_whitelist]
        info = df[info_in_data]
        data = df[data_columns]
        
    return data, info


def __apply_segmentation(df: [pd.DataFrame],
                         sampling_frequency: int,
                         segment_duration: float,
                         create_transitions: bool = False,
                         representation: str = 'segmentation',
                         segment_duration_overlap: float = 0) -> [pd.DataFrame]:
    """
    Segment the input dataset
    :param df: the dataset
    :param sampling_frequency: dataset sampling frequency
    :param segment_duration: number of seconds to study as a window
    :param segment_duration_overlap: overlap time between windows
    :return:
    """
    # Number of rows in a segment
    segment_length = int(segment_duration * sampling_frequency)

    hop_size = segment_length - int(segment_duration_overlap * sampling_frequency)

    segmented_df = []
    metadata_seg = []

    for df_i in df:
        if 'timestamp' in df_i.columns:
            df_i.drop('timestamp', axis=1, inplace=True)
        data_df_i = []
        info_df_i = []
        #print(len(df_i) - segment_length + 1)

        for i in range(0, len(df_i) - segment_length + 1, hop_size):
            # Creating a segment
            data_segment = pd.DataFrame()
            info_segment = pd.DataFrame()

            for column in df_i.columns:
                if column not in columns_whitelist:
                    # Here we are dealing with data
                    data_segment[column] = df_i[column].values[i: i + segment_length]
                else:
                    # Here we are dealing with metadata

                    # Creating transition labels
                    if create_transitions and column == "label":
                        labels = df_i[column].values[i: i + segment_length]
                        first_label = labels[0]
                        found_transition = False
                        for label in labels:
                            if label is not first_label:
                                info_segment[column] = [f"{first_label}_to_{label}"]
                                found_transition = True
                                break
                        # If there's only one label in the window
                        if not found_transition:
                            info_segment[column] = [df_i[column].values[i: i + segment_length][0]]
                    else:
                        info_segment[column] = [mode(df_i[column].values[i: i + segment_length])]
            # Append
            data_df_i.append(data_segment)
            info_df_i.append(info_segment)

        segmented_df.append(data_df_i)
        metadata_seg.append(info_df_i)

    if representation == 'features':
        return segmented_df, metadata_seg

    else:
        compliant_segmented_df = []
        for df_i in segmented_df:
            compact_df_i = __to_compact_form(df_i)
            if len(compact_df_i) > 0:
                new_names = list(compact_df_i.columns) * segment_length
                new_shape = (-1, compact_df_i.shape[1] * segment_length)
                compliant_segmented_df.append(pd.DataFrame(compact_df_i.values.reshape(new_shape), columns=new_names))

        return compliant_segmented_df, metadata_seg


def __apply_feature_extraction(df: [pd.DataFrame],
                               sampling_frequency: int,
                               segment_duration: float,
                               domain: str = 'statistical',
                               segment_duration_overlap: float = 0,
                               create_transitions: str = False):

    # Set up feature extractor
    cfg = ts.get_features_by_domain(domain=None if domain == 'all' else domain)

    # Save information for each time window
    data, info = __apply_segmentation(
        df=df,
        sampling_frequency=sampling_frequency,
        representation='features',
        segment_duration=segment_duration,
        segment_duration_overlap=segment_duration_overlap,
        create_transitions=create_transitions
    )

    features_df = []
    for df_i in data:
        compact_df_i = __to_compact_form(df_i)

        # Start feature extraction
        features_df_i = ts.time_series_features_extractor(
            cfg,
            compact_df_i,
            fs=sampling_frequency,
            window_size=int(segment_duration * sampling_frequency),
            overlap=segment_duration_overlap,
            verbose=0
        )

        features_df.append(features_df_i)
    return features_df, info


def __to_compact_form(segmented_df: [pd.DataFrame]) -> pd.DataFrame:
    compact_df_i = []
    for segment in segmented_df:
        compact_df_i.append(segment)

    # Create a compact dataframe from all segments
    if len(compact_df_i) > 0:
        compact_df_i = pd.concat(compact_df_i, axis=0)
    else:
        pass

    return compact_df_i
