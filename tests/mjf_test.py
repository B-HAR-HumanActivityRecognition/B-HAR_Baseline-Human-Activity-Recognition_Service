import unittest
import json
from bhar import BHAR


class MyTestCase(unittest.TestCase):

    paths = json.load(open('datasets_path.json'))

    # Exploratory Analysis
    def test_exp_analysis_mjf_task(self):
        file_path = self.paths['mjf-task']

        bhar = BHAR(
            dataset_path=file_path,
            dataset_type='fragmented',
            each_file_is_subject=True,
            dataset_sampling_frequency=50,  # Hz
            segment_time=1,  # seconds
            header_format='tdl',  # time, data, label
            representation='segmentation',
            train_test_split_method='normal',
            apply_filtering=False,
            normalization_method='min-max'
        )

        bhar.exploratory_analysis()

        self.assertEqual(True, True)

    def test_exp_analysis_mjf_status(self):
        file_path = self.paths['mjf-status']

        bhar = BHAR(
            dataset_path=file_path,
            dataset_type='fragmented',
            each_file_is_subject=True,
            dataset_sampling_frequency=50,  # Hz
            segment_time=1,  # seconds
            header_format='tdl',  # time, data, label
            representation='segmentation',
            train_test_split_method='normal',
            apply_filtering=False,
            normalization_method='min-max'
        )

        bhar.exploratory_analysis()

        self.assertEqual(True, True)

    # Raw Data Normal Train/Test split
    def test_baseline_mjf_task(self):
        file_path = self.paths['mjf-task']

        bhar = BHAR(
            dataset_path=file_path,
            dataset_type='fragmented',
            each_file_is_subject=True,
            dataset_sampling_frequency=50,  # Hz
            segment_time=1,  # seconds
            header_format='tdl',  # time, data, label
            representation='segmentation',
            train_test_split_method='normal',
            apply_filtering=False,
            normalization_method='min-max'
        )

        bhar.get_baseline()

        self.assertEqual(True, True)

    def test_baseline_mjf_status(self):
        file_path = self.paths['mjf-status']

        bhar = BHAR(
            dataset_path=file_path,
            dataset_type='fragmented',
            each_file_is_subject=True,
            dataset_sampling_frequency=50,  # Hz
            segment_time=1,  # seconds
            header_format='tdl',  # time, data, label
            representation='segmentation',
            train_test_split_method='normal',
            apply_filtering=False,
            normalization_method='min-max'
        )

        bhar.get_baseline()

        self.assertEqual(True, True)

    # Raw Data Leave-Out
    def test_leave_out_baseline_mjf_task(self):
        file_path = self.paths['mjf-task']

        bhar = BHAR(
            dataset_path=file_path,
            dataset_type='fragmented',
            each_file_is_subject=True,
            dataset_sampling_frequency=50,  # Hz
            segment_time=1,  # seconds
            header_format='tdl',  # time, data, label
            representation='segmentation',
            train_test_split_method='leave-out',
            leave_out_filter={'subject': [10]},
            apply_filtering=False,
            normalization_method='min-max'
        )

        bhar.get_baseline()

        self.assertEqual(True, True)

    def test_leave_out_baseline_mjf_status(self):
        file_path = self.paths['mjf-status']

        bhar = BHAR(
            dataset_path=file_path,
            dataset_type='fragmented',
            each_file_is_subject=True,
            dataset_sampling_frequency=50,  # Hz
            segment_time=1,  # seconds
            header_format='tdl',  # time, data, label
            representation='segmentation',
            train_test_split_method='leave-out',
            leave_out_filter={'subject': [10]},
            apply_filtering=False,
            normalization_method='min-max'
        )

        bhar.get_baseline()

        self.assertEqual(True, True)

    # Features Normal Train/Test split
    def test_features_baseline_mjf_task(self):
        file_path = self.paths['mjf-task']

        bhar = BHAR(
            dataset_path=file_path,
            dataset_type='fragmented',
            each_file_is_subject=True,
            dataset_sampling_frequency=50,  # Hz
            segment_time=1,  # seconds
            header_format='tdl',  # time, data, label
            representation='features',
            train_test_split_method='normal',
            apply_filtering=False,
            normalization_method='min-max'
        )

        bhar.get_baseline()

        self.assertEqual(True, True)

    def test_features_baseline_mjf_status(self):
        file_path = self.paths['mjf-status']

        bhar = BHAR(
            dataset_path=file_path,
            dataset_type='fragmented',
            each_file_is_subject=True,
            dataset_sampling_frequency=50,  # Hz
            segment_time=1,  # seconds
            header_format='tdl',  # time, data, label
            representation='features',
            train_test_split_method='normal',
            apply_filtering=False,
            normalization_method='min-max'
        )

        bhar.get_baseline()

        self.assertEqual(True, True)

    # Features Leave-Out
    def test_features_leave_out_baseline_mjf_task(self):
        file_path = self.paths['mjf-task']

        bhar = BHAR(
            dataset_path=file_path,
            dataset_type='fragmented',
            each_file_is_subject=True,
            dataset_sampling_frequency=50,  # Hz
            segment_time=1,  # seconds
            header_format='tdl',  # time, data, label
            representation='features',
            train_test_split_method='leave-out',
            leave_out_filter={'subject': [10]},
            apply_filtering=False,
            normalization_method='min-max'
        )

        bhar.get_baseline()

        self.assertEqual(True, True)

    def test_features_leave_out_baseline_mjf_status(self):
        file_path = self.paths['mjf-status']

        bhar = BHAR(
            dataset_path=file_path,
            dataset_type='fragmented',
            each_file_is_subject=True,
            dataset_sampling_frequency=50,  # Hz
            segment_time=1,  # seconds
            header_format='tdl',  # time, data, label
            representation='features',
            train_test_split_method='leave-out',
            leave_out_filter={'subject': [10]},
            apply_filtering=False,
            normalization_method='min-max'
        )

        bhar.get_baseline()

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
