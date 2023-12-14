import json
import os
import unittest

import pandas as pd

from bhar import BHAR


class PaperTestCase(unittest.TestCase):

#    paths = json.load(open('datasets_path.json'))

    def test_papam(self):
        file_path = self.paths['papam']

        bhar = BHAR(
            dataset_path=file_path,
            dataset_sampling_frequency=100,  # Hz
            segment_time=2,  # seconds
            header_format='tdls',  # time, data, label, subject
            representation='segmentation',
            train_test_split_method='normal',
            normalization_method='min-max'
        )

        bhar.get_baseline()

        self.assertEqual(True, True)

    def test_mhealth(self):
        file_path = self.paths['mhealth']

        bhar = BHAR(
            dataset_path=file_path,
            dataset_sampling_frequency=50,  # Hz
            segment_time=5,  # seconds
            overlap=2.5,
            header_format='dls',  # time, data, label, subject
            representation='segmentation',
            filter_type='lowpass',
            filter_cutoff=20,
            train_test_split_method='normal',
            normalization_method='min-max'
        )

        bhar.get_baseline()

        self.assertEqual(True, True)

    def test_hhar_watch(self):
        file_path = self.paths['hhar_watch']

        bhar = BHAR(
            dataset_path=file_path,
            dataset_sampling_frequency=200,  # Hz
            segment_time=2,  # seconds
            overlap=1,
            header_format='tdls',  # time, data, label, subject
            representation='segmentation',
            train_test_split_method='normal',
            normalization_method='min-max'
        )

        bhar.get_baseline()

        self.assertEqual(True, True)

    def test_hhar_phone(self):
        file_path = self.paths['hhar_phone']

        bhar = BHAR(
            dataset_path=file_path,
            dataset_sampling_frequency=200,  # Hz
            segment_time=2,  # seconds
            overlap=1,
            header_format='tdls',  # time, data, label, subject
            representation='segmentation',
            train_test_split_method='normal',
            normalization_method='min-max'
        )

        bhar.get_baseline()

        self.assertEqual(True, True)

    def test_daphnet(self):
        file_path = self.paths['daphnet']

        bhar = BHAR(
            dataset_path=file_path,
            dataset_sampling_frequency=64,  # Hz
            segment_time=4,  # seconds
            overlap=0.5,
            header_format='tdls',  # time, data, label, subject
            representation='segmentation',
            filter_type='lowpass',
            filter_cutoff=20,
            train_test_split_method='normal',
            normalization_method='min-max'
        )

        bhar.get_baseline()

        self.assertEqual(True, True)

    def test_WISDM_v1(self):
        file_path = self.paths['wisdm_v1']

        bhar = BHAR(
            dataset_path=file_path,
            dataset_sampling_frequency=20,  # Hz
            segment_time=10,  # seconds
            header_format='tdls',  # time, data, label, subject
            representation='segmentation',
            train_test_split_method='normal',
            normalization_method='min-max'
        )

        bhar.get_baseline()

        self.assertEqual(True, True)

    def test_WISDM_v2(self):
        file_path = self.paths['wisdm_v2']

        bhar = BHAR(
            dataset_path=file_path,
            dataset_sampling_frequency=20,  # Hz
            segment_time=10,  # seconds
            header_format='tdls',  # time, data, label, subject
            representation='segmentation',
            train_test_split_method='normal',
            normalization_method='min-max'
        )

        bhar.get_baseline()

        self.assertEqual(True, True)

    # --- New Datasets ---

    def test_daliac(self):
        file_path = self.paths['daliac']

        bhar = BHAR(
            dataset_path=file_path,
            dataset_sampling_frequency=204,  # Hz
            segment_time=1,  # seconds
            header_format='dl',  # time, data, label, subject
            representation='segmentation',
            train_test_split_method='normal',
            normalization_method='min-max'
        )

        bhar.exploratory_analysis()
        bhar.get_baseline()

        self.assertEqual(True, True)

    def test_basa(self):
        file_path = self.paths['basa']

        bhar = BHAR(
            dataset_path=file_path,
            dataset_sampling_frequency=204,  # Hz
            segment_time=1,  # seconds
            header_format='sld',  # time, data, label, subject
            representation='segmentation',
            train_test_split_method='normal',
            normalization_method='min-max'
        )

        bhar.get_baseline()

        self.assertEqual(True, True)

    def test_biofeedback(self):
        file_path = self.paths['bio_feedback']

        bhar = BHAR(
            dataset_path=file_path,
            dataset_sampling_frequency=1000,  # Hz
            segment_time=5,  # seconds
            overlap=0,
            header_format='tdslm',  # time, data, subject, label, session
            representation='segmentation',
            train_test_split_method='normal',
            normalization_method='min-max'
        )

        bhar.get_baseline()

        self.assertEqual(True, True)

    def test_mjf_task(self):
        file_path = self.paths['mjf-task']

        bhar = BHAR(
            dataset_path=file_path,
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
        bhar.get_baseline()

        self.assertEqual(True, True)

    def test_mjf_status(self):
        file_path = self.paths['mjf-status']

        bhar = BHAR(
            dataset_path=file_path,
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
        bhar.get_baseline()

        self.assertEqual(True, True)

    def test_tewabe_A(self):
        file_path = '../dataset/A/nnn_reindex25A.csv'
        # df = []
        # for file in os.listdir(file_path):
        #     if file.endswith('A.csv'):
        #        df.append(pd.read_csv(f"{file_path}/{file}", index_col=0))
        #
        # df = pd.concat(df, ignore_index=True)
        # df.index = list(range(len(df)))
        # df.to_csv("A.csv")

        # df = pd.read_csv('../dataset/A/nnn_reindex22A.csv')

        bhar = BHAR(
            dataset_path=file_path,
            dataset_sampling_frequency=1000,  # Hz
            segment_time=0.001,  # seconds
            overlap=0,
            header_format='tdslm',  # time, data, subject, label, session
            representation='raw',
            apply_filtering=False,
            train_test_split_method='normal',
            normalization_method='min-max'
        )

        bhar.get_baseline()

        self.assertEqual(True, True)

    def test_tewabe_B(self):
        file_path = 'B.csv'

        bhar = BHAR(
            dataset_path=file_path,
            dataset_sampling_frequency=1000,  # Hz
            segment_time=0.25,  # seconds
            overlap=0,
            header_format='tdslm',  # time, data, subject, label, session
            representation='segmentation',
            apply_filtering=False,
            train_test_split_method='normal',
            normalization_method='min-max'
        )

        bhar.get_baseline()

        self.assertEqual(True, True)

    def test_tewabe_A_statistical(self):
        file_path = 'A.csv'

        bhar = BHAR(
            dataset_path=file_path,
            dataset_sampling_frequency=1000,  # Hz
            segment_time=0.25,  # seconds
            overlap=0,
            header_format='tdslm',  # time, data, subject, label, session
            representation='segmentation',
            feature_domain='statistical',
            apply_filtering=False,
            train_test_split_method='normal',
            normalization_method='min-max'
        )

        bhar.get_baseline()

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
