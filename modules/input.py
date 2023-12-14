import os
import pandas as pd
import re


def __atoi(text):
    return int(text) if text.isdigit() else text


def __natural_keys(text):
    return [__atoi(c) for c in re.split('(\d+)', text)]


def __create_header(keyword, n_cols) -> [str]:
    """
    Creates an header for the dataset based on the given keyword, which represent the columns distributions
    :param keyword: string representing the order of the columns
    :param n_cols: number of columns in the dataframe
    :return: ana array of string containing the new columns names
    """
    header = []
    num_data_cols = 0
    for placeholder in keyword:
        if placeholder == 't':
            # Add time to the header
            header.append('timestamp')
        elif placeholder == 'l':
            # Add label to the header
            header.append('label')
        elif placeholder == 's':
            # Add subject to the header
            header.append('subject')
        elif placeholder == 'm':
            # Add session to the header
            header.append('session')
        elif placeholder == 'd':
            # Add data columns
            num_data_cols = n_cols - len(keyword) + 1
            for i in range(num_data_cols):
                header.append('data_{}'.format(i))

    return header, num_data_cols


def read_dataset(
        path: str,
        header_format: str, 
        sep: str = ',',
        derive_header: bool = True,
        each_file_is_subject: bool = False,
        ds_type: str = 'continuous'
) -> pd.DataFrame:
    """
    Read BHAR compliant datasets
    :param each_file_is_subject: if it is passed a directory, each file in it represents a subject
    :param ds_type: type of the recorded dataset, continuous or fragmented
    :param path: path to file or directory containing the dataset
    :param header_format: string representing the order of the columns
    :param sep: the character which separates the columns in the dataset file
    :param derive_header: use default header in the dataset or derive an header compliant to BHAR logic
    :return: the pandas dataframe
    """
    if os.path.isdir(path):
        # The dataset is contained in a folder
        df = []
        # Get the files in the directory and sort it
        files = os.listdir(path)
        files.sort(key=__natural_keys)

        # Keep only valuable files
        files = [f for f in files if (f.endswith('.csv') or f.endswith('.txt')) and not f.startswith('.')]

        for i, file in zip(range(len(files)), files):
            if (file.endswith('.csv') or file.endswith('.txt')) and not file.startswith('.'):
                df_i = pd.read_csv(os.path.join(path, file), sep=sep, header=None if derive_header else 'infer')

                if each_file_is_subject:
                    df_i[len(df_i.columns)] = i

                df.append(df_i)

        df = pd.concat(df, axis=0)

    elif os.path.isfile(path):
        # The whole dataset is contained in a file
        try:
            df = pd.read_csv(path, sep=sep)
        except pd.errors.EmptyDataError:
            raise Exception("The dataset is empty")

    else:
        raise FileNotFoundError("Dataset not found")

    # Change to derived header if necessary
    if derive_header:
        if each_file_is_subject and 's' not in header_format:
            header_format += 's'
        df.columns, data_shape = __create_header(header_format, df.shape[1])

    if 'm' in header_format:
        pass

    if ds_type == 'continuous':
        if 'm' in header_format:
            # There is also the session
            df = df.groupby(['subject', 'session'])
        else:
            # There are no sessions, the subject performs all activities in a unique record
            #df['subject'] = -1
            df = df.groupby('subject')

    else:
        if 'm' in header_format:
            # There is also the session
            df = df.groupby(['subject', 'label', 'session'])
        else:
            # There are no sessions, the subject performs each activity in a unique record
            df = df.groupby(['subject', 'label'])

    return [d[1] for s, d in enumerate(df)], data_shape
