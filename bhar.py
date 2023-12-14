from pathlib import Path
from datetime import datetime

from modules.input import *
from modules.cleaning import *
from modules.representation import dataset_representation
from modules.preprocessing import drop_data, train_test_splitting, normalization, features_selection, balancing
from modules.train_and_test import ModelsManager
from modules.data_exploration import feature_importance, correlation_matrix, feature_distribution, pca_plot, sensor_analysis
from models.machine_learning_hypermodels import *
from models.deep_learning_hypermodels import *
from models.user_models import *
from utility.reporter import Reporter

class BHAR(object):
    def __init__(
            self,
            # Input
            username: str,
            name: str,
            dataset_path: str,
            dataset_sampling_frequency: int,
            header_format: str,
            derive_header: bool = True,
            each_file_is_subject: bool = False,
            separator: str = ',',
            dataset_type: str = 'continuous',  # or fragmented

            # Data Cleaning
            replace_error_method: str = 'mean',
            apply_filtering: bool = True,
            filter_type: str = 'lowpass',  # or highpass, bandpass
            filter_order: int = 4,
            filter_cutoff: int = 20,

            # Data Representation
            representation: str = 'raw',  # or features, segmentation
            segment_time: float = 1.0,  # seconds
            overlap: float = 0.0,  # seconds
            feature_domain: str = 'all',  # or statistical, spectral, temporal
            create_transitions: bool = False,

            # Pre-processing
            to_drop: dict = None,
            train_test_split_method: str = 'normal',  # leave-out, normal
            leave_out_filter: dict = None,
            normalization_method: str = None,
            feature_selection_method: str = None,
            balancing_method: str = None,

            # Train and test
            # useless
            # user_models: list = None,
            target_feature: str = 'label',

            # Stages
            data_cleaning: bool = True,
            data_representation: bool = True,
            pre_processing: bool = True,
            model_training_and_testing: bool = True,

            # Export data after processing module
            export_data_error: bool = False,
            export_noise_removal: bool = False,
            export_data_representation: bool = False,
            export_pre_processing: bool = False,
            export_time_stats: bool = False,
    ):
        self._username = username
        self._name = name
        self._dataframe = None
        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None

        self._dataset_sampling_frequency = dataset_sampling_frequency
        self._dataset_path = dataset_path
        self._header_format = header_format
        self._derive_header = derive_header
        self._each_file_is_subject = each_file_is_subject
        self._separator = separator
        self._dataset_type = dataset_type
        self._columns_whitelist = ['timestamp', 'label', 'subject', 'session']

        self._replace_method = replace_error_method
        self._apply_filter = apply_filtering
        self._filter_type = filter_type
        self._filter_order = filter_order
        self._filter_cutoff = filter_cutoff

        self._representation = representation
        self._segment_time = segment_time
        self._overlap = overlap
        self._features_domain = feature_domain
        self._create_transitions = create_transitions

        self._data = None  # Sensors data are stored here
        self._info = None  # Info like: labels, session and subject are stored here

        self._to_drop = to_drop
        self._train_test_split_method = train_test_split_method
        self._leave_out_filter = leave_out_filter
        self._normalization_method = normalization_method
        self._features_selection_method = feature_selection_method
        self._balancing_method = balancing_method

        self._models_manager = None
        # useless
        # self._user_models = user_models
        self._target_feature = target_feature

        self._data_cleaning = data_cleaning
        self._data_representation = data_representation
        self._pre_processing = pre_processing
        self._model_training_and_testing = model_training_and_testing

        self._export_data_error = export_data_error
        self._export_noise_removal= export_noise_removal
        self._export_data_representation = export_data_representation
        self._export_pre_processing = export_pre_processing
        self._export_time_stats = export_time_stats

        self._reporter = Reporter(ref_file=dataset_path, bhar_name=self._name, bhar_username=self._username)

        # Input
        self._reporter.write_section('Input', print_elapsed_time=False)

        self._dataframe, self._data_shape = read_dataset(
            path=self._dataset_path,
            header_format=self._header_format,
            sep=self._separator,
            derive_header=self._derive_header,
            each_file_is_subject=self._each_file_is_subject,
            ds_type=self._dataset_type
        )
        self._reporter.write_body('Dataset header: {}'.format(list(self._dataframe[0].columns)))
        self._reporter.write_body('Dataset shape: {}'.format(self._dataframe[0].shape))

    def __eq__(self, other):
        return self._name == other._name and self._username == other._username
    
    def __hash__(self):
        return hash(self._name + self._username)
    
    def to_json(self):
        return {
            "username": self._username,
            "name": self._name,
            "dataset_path": self._dataset_path,
            "dataset_sampling_frequency": self._dataset_sampling_frequency,
            "header_format": self._header_format,
            "derive_header": self._derive_header,
            "each_file_is_subject": self._each_file_is_subject,
            "separator": self._separator,
            "dataset_type": self._dataset_type,
            "replace_error_method": self._replace_method,
            "apply_filtering": self._apply_filter,
            "filter_type": self._filter_type,
            "filter_order": self._filter_order,
            "filter_cutoff": self._filter_cutoff,
            "representation": self._representation,
            "segment_time": self._segment_time,
            "overlap": self._overlap,
            "feature_domain": self._features_domain,
            "create_transitions": self._create_transitions,
            "to_drop": self._to_drop,
            "train_test_split_method": self._train_test_split_method,
            "leave_out_filter": self._leave_out_filter,
            "normalization_method": self._normalization_method,
            "feature_selection_method": self._features_selection_method,
            "balancing_method": self._balancing_method,
            # useless
            # "user_models": self._user_models,
            "target_feature": self._target_feature,
            "data_cleaning": self._data_cleaning,
            "data_representation": self._data_representation,
            "pre_processing": self._pre_processing,
            "model_training_and_testing": self._model_training_and_testing,
            "export_data_error": self._export_data_error,
            "export_noise_removal": self._export_noise_removal,
            "export_data_representation": self._export_data_representation,
            "export_pre_processing": self._export_pre_processing,
            "export_time_stats": self._export_time_stats,
        }

    def get_reporter(self):
        return self._reporter
    
    def get_info(self):
        return self._info

    def _get_num_classes(self, target):
        return len(self._y_train[target].unique())

    def _get_shape_cnn(self):
        shape = (self._y_train.shape[0], -1, self._data_shape)
        return self._X_train.reshape(shape).shape[1:]

    def get_baseline(self):
        current_time = datetime.now()

        # Data Cleaning
        if self._data_cleaning:
            self._reporter.write_section('Data Cleaning')
            self._reporter.write_subsection('Handle Data Errors')
            self._dataframe, errors = remove_data_errors(
                df=self._dataframe,
                error_sub_method=self._replace_method
            )
            self._reporter.export_time_stats("Handle Data Errors", elapsed_time=(datetime.now()-current_time), write=self._export_time_stats)
            current_time = datetime.now()

            self._reporter.write_body('Removed {} errors'.format(errors))
            self._reporter.export_data("error_handling", df=self._dataframe, write=self._export_data_error)

            if self._apply_filter:
                self._reporter.write_subsection('Noise Removal')
                self._reporter.write_body('Applied {} filter'.format(self._filter_type))
                self._dataframe = remove_noise(
                    df=self._dataframe,
                    sample_rate=self._dataset_sampling_frequency,
                    filter_name=self._filter_type,
                    cutoff=self._filter_cutoff
                )
                self._reporter.export_time_stats("Noise Removal", elapsed_time=(datetime.now()-current_time), write=self._export_time_stats)
                current_time = datetime.now()
                self._reporter.export_data("noise_removal", df=self._dataframe, write=self._export_noise_removal)

        # Data Representation
        if self._data_representation:
            self._reporter.write_subsection('Data Representation')
            self._reporter.write_body('Applied {}'.format(self._representation))
            self._data, self._info = dataset_representation(
                df=self._dataframe,
                rep_type=self._representation,
                sampling_frequency=self._dataset_sampling_frequency,
                segment_duration=self._segment_time,
                segment_duration_overlap=self._overlap,
                create_transitions=self._create_transitions
            )
            self._reporter.export_time_stats("Data Representation", elapsed_time=(datetime.now()-current_time), write=self._export_time_stats)
            current_time = datetime.now()
            self._reporter.export_data("data_representation", df=self._data, write=self._export_data_representation)

        # Pre-processing
        if self._pre_processing:
            self._reporter.write_section('Pre-processing')
            if self._to_drop is not None:
                self._reporter.write_subsection('Drop Unnecessary Data')
                self._reporter.write_body('Excluded from the analysis: {}'.format(self._to_drop))
                self._data, self._info = drop_data(self._data, self._info, to_drop=self._to_drop)
                self._reporter.export_time_stats("Drop Unnecessary Data", elapsed_time=(datetime.now()-current_time), write=self._export_time_stats)
                current_time = datetime.now()

            self._reporter.write_subsection('Train Test Split')
            self._reporter.write_body('Applied technique: {}'.format(self._train_test_split_method))
            self._X_train, self._X_test, self._y_train, self._y_test = train_test_splitting(
                self._data,
                self._info,
                method=self._train_test_split_method,
                filter_dict=self._leave_out_filter
            )
            self._reporter.export_time_stats("Train Test Split", elapsed_time=(datetime.now()-current_time), write=self._export_time_stats)
            current_time = datetime.now()

            if self._normalization_method is not None:
                self._reporter.write_subsection('Data Normalization')
                self._reporter.write_body('The data has been normalized with: {}'.format(self._normalization_method))
                self._X_train, self._X_test = normalization(
                    self._X_train,
                    self._X_test,
                    n_data_cols=self._data_shape,
                    method=self._normalization_method,
                    representation=self._representation
                )
                self._reporter.export_time_stats("Data Normalization", elapsed_time=(datetime.now()-current_time), write=self._export_time_stats)
                current_time = datetime.now()

            if self._features_selection_method is not None:
                self._reporter.write_subsection('Feature Selection')
                self._reporter.write_body('Feature has been selected using {}'.format(self._features_selection_method))
                self._X_train, self._X_test = features_selection(
                    self._X_train,
                    self._X_test,
                    method=self._features_selection_method,
                    train_info=self._y_train
                )
                self._reporter.export_time_stats("Feature Selection", elapsed_time=(datetime.now()-current_time), write=self._export_time_stats)
                current_time = datetime.now()

            if self._balancing_method is not None:
                self._reporter.write_subsection('Data Balancing')
                self._reporter.write_body('Data has been balanced using {}'.format(self._balancing_method))
                self._X_train, self._y_train = balancing(
                    self._X_train,
                    self._y_train,
                    method=self._balancing_method
                )
                self._reporter.export_time_stats("Data Balancing", elapsed_time=(datetime.now()-current_time), write=self._export_time_stats)
                current_time = datetime.now()
            self._reporter.export_data("pre_processing", df=self._data, write=self._export_pre_processing)

        # Train and Testing
        if self._model_training_and_testing:
            self._reporter.write_section('Train and Test', print_elapsed_time=False)
            shape = (0, 0)
            if self._representation == 'raw':
                pass
            if self._representation == 'segmentation':
                self._X_train = self._X_train.values
                self._X_test = self._X_test.values
                shape = self._get_shape_cnn()
                self._reporter.export_time_stats("Segmentation", elapsed_time=(datetime.now()-current_time), write=self._export_time_stats) 
                current_time = datetime.now()

            num_classes = self._get_num_classes(self._target_feature)
            self._reporter.write_body(f'Total target classes: {num_classes}')
            models = {'ml': [], 'dl': []}
            # Edit here to remove or add some models
            # models['ml'].append(KNN(self._username))
            models['ml'].append(RandomForest(self._username))
            # models['ml'].append(SupportVectorClassification(self._username))
            # models['ml'].append(DecisionTree(self._username))
            # models['ml'].append(WKNN(self._username))
            models['dl'].append(LightCNNHyperModel(self._username, shape, num_classes, self._reporter.get_dataset_name()))
            # models['dl'].append(CNNHyperModel(self._username, shape, num_classes, self._reporter.get_dataset_name()))
            # models['dl'].append(RNNHyperModel(self._username, shape, num_classes, self._reporter.get_dataset_name()))
            # models['dl'].append(LSTMHyperModel(self._username, shape, num_classes, self._reporter.get_dataset_name()))
            self._models_manager = ModelsManager(shape, num_classes, self._reporter, models)

            # useless
            # if self._user_models is not None:
            #     for usr_model in self._user_models:
            #         self._models_manager.add_model(usr_model)

            self._models_manager.start_train_and_test(
                x_train=self._X_train,
                x_test=self._X_test,
                y_train=self._y_train[self._target_feature].values,
                y_test=self._y_test[self._target_feature].values,
                representation=self._representation
            )
            self._reporter.export_time_stats("Train and Testing", elapsed_time=(datetime.now()-current_time), write=self._export_time_stats)
            current_time = datetime.now()


        self._reporter.write_end()

    def exploratory_analysis(self):
        current_time = datetime.now()
        self._reporter.write_section('Data Cleaning')
        self._reporter.write_subsection('Handle Data Errors')
        self._dataframe, errors = remove_data_errors(
            df=self._dataframe,
            error_sub_method=self._replace_method
        )
        self._reporter.export_time_stats("Handle Data Errors", elapsed_time=(datetime.now()-current_time), write=self._export_time_stats)
        current_time = datetime.now()
        self._reporter.write_body('Removed {} errors'.format(errors))
        self._reporter.export_data("error_handling", df=self._dataframe, write=self._export_data_error)

        # Folder path
        folder = '{}_exploratory_data_analysis'.format(self._reporter.get_dataset_name())

        # Create stats directory
        Path(folder).mkdir(parents=True, exist_ok=True)

        # Number of patients

        # Number of features

        # Number of patients per target label
        feature_distribution(self._dataframe, folder)

        # Heatmap correlation variable
        correlation_matrix(self._dataframe, folder)

        # Bar plot of features importance
        feature_importance(self._dataframe, folder)

        # PCA 2D plot
        pca_plot(self._dataframe, self._dataset_sampling_frequency, self._segment_time)

        # Sensor analysis with plots
        sensor_analysis(self._dataframe, self._dataset_sampling_frequency, self._segment_time)
