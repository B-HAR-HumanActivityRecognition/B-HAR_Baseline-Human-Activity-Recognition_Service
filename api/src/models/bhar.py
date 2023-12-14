from pydantic import ConfigDict, BaseModel

class BHARModel(BaseModel):
    # Input
    username: str
    name: str
    dataset_path: str
    dataset_sampling_frequency: int
    header_format: str
    derive_header: bool = True
    each_file_is_subject: bool = False
    separator: str = ','
    dataset_type: str = 'continuous' # or fragmente
    # Data Cleaning
    replace_error_method: str = 'mean'
    apply_filtering: bool = True
    filter_type: str = 'lowpass' # or highpass, bandpass
    filter_order: int = 4
    filter_cutoff: int = 20
    # Data Representation
    representation: str = 'raw'  # or features, segmentation
    segment_time: float = 1.0  # seconds
    overlap: float = 0.0 # seconds
    feature_domain: str = 'all'  # or statistical, spectral, temporal
    create_transitions: bool = False
    # Pre-processing
    to_drop: dict = None
    train_test_split_method: str = 'normal' # leave-out, normal
    leave_out_filter: dict = None
    normalization_method: str = None
    feature_selection_method: str = None
    balancing_method: str = None
    # Train and test
    # useless
    # user_models: list = None
    target_feature: str = 'label'
    # Stages
    data_cleaning: bool = True
    data_representation: bool = True
    pre_processing: bool = True
    model_training_and_testing: bool = True
    # Export data after processing module
    export_data_error: bool = False
    export_noise_removal: bool = False
    export_data_representation: bool = False
    export_pre_processing: bool = False
    export_time_stats: bool = False

    model_config = ConfigDict(
        protected_namespaces=(),
    )