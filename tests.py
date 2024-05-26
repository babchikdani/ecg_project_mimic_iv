from pathlib import Path
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import unittest
from ecg_dataset import ECGDataset
#  These diagnostic ECGs use 12 leads and are 10 seconds in length. They are sampled at 500 Hz.


class SignalDataTestCase(unittest.TestCase):
    def test_missing_data(self):
        self._run_test_with_data_loader(self._check_missing_data)

    def test_sig_mat_rows(self):
        self._run_test_with_data_loader(self._check_sig_mat_rows)

    def test_sig_mat_cols(self):
        self._run_test_with_data_loader(self._check_sig_mat_cols)

    def test_sig_names(self):
        self._run_test_with_data_loader(self._check_sig_names)

    def test_sig_fs(self):
        self._run_test_with_data_loader(self._check_sig_fs)

    def test_sig_units(self):
        self._run_test_with_data_loader(self._check_sig_units)

    def test_sig_leads(self):
        self._run_test_with_data_loader(self._check_sig_leads)

    def _run_test_with_data_loader(self, test_function):
        files_directory = Path("./files")
        ecg_dataset = ECGDataset(files_directory)
        batch_size = 32
        data_loader = DataLoader(ecg_dataset, batch_size=batch_size, shuffle=False, collate_fn=ecg_dataset.collate_fn)

        for batch in data_loader:
            test_function(batch)

    def _check_missing_data(self, batch):
        signal_data, signal_metadata_list, metadata_list = batch
        for signal_data_item, signal_metadata, metadata in zip(signal_data, signal_metadata_list, metadata_list):
            subject_id = get_subject_id(signal_metadata)
            signal_id = get_signal_id(metadata)
            self.assertFalse(any(value is None for value in signal_data_item), f"Missing data found in signal. - Subject id is: {subject_id}, Signal id is {signal_id}.")

    def _check_sig_mat_rows(self, batch):
        signal_data, signal_metadata_list, metadata_list = batch
        for signal_data_item, signal_metadata, metadata in zip(signal_data, signal_metadata_list, metadata_list):
            excepted_rows_num = 12
            subject_id = get_subject_id(signal_metadata)
            signal_id = get_signal_id(metadata)
            self.assertEqual(signal_data_item.shape[0],excepted_rows_num, f"Irregular signal rows found in signal. - Subject id is: {subject_id}, Signal id is {signal_id}.")

    def _check_sig_mat_cols(self, batch):
        signal_data, signal_metadata_list, metadata_list = batch
        for signal_data_item, signal_metadata, metadata in zip(signal_data, signal_metadata_list, metadata_list):
            excepted_rows_num = 5000
            subject_id = get_subject_id(signal_metadata)
            signal_id = get_signal_id(metadata)
            self.assertEqual(signal_data_item.shape[1],excepted_rows_num, f"Irregular signal columns found in signal. - Subject id is: {subject_id}, Signal id is {signal_id}.")


    def _check_sig_names(self, batch):
        signal_data, signal_metadata_list, metadata_list = batch
        expected_signal_names = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        for signal_data_item, signal_metadata, metadata in zip(signal_data, signal_metadata_list, metadata_list):
            signal_names = signal_metadata["sig_name"]
            subject_id = get_subject_id(signal_metadata)
            signal_id = get_signal_id(metadata)
            self.assertEqual(signal_names, expected_signal_names, f"Incorrect signal names found in signal - Subject id is: {subject_id}, Signal id is {signal_id}.")

    def _check_sig_units(self, batch):
        signal_data, signal_metadata_list, metadata_list = batch
        expected_signal_units = ['mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV', 'mV']
        for signal_data_item, signal_metadata, metadata in zip(signal_data, signal_metadata_list, metadata_list):
            signal_units = signal_metadata["units"]
            subject_id = get_subject_id(signal_metadata)
            signal_id = get_signal_id(metadata)
            self.assertEqual(signal_units, expected_signal_units, f"Incorrect signal names found in signal - Subject id is: {subject_id}, Signal id is {signal_id}.")

    def _check_sig_fs(self, batch):
        signal_data, signal_metadata_list, metadata_list = batch
        expected_signal_fs = 500
        for signal_data_item, signal_metadata, metadata in zip(signal_data, signal_metadata_list, metadata_list):
            signal_fs = signal_metadata["fs"]
            subject_id = get_subject_id(signal_metadata)
            signal_id = get_signal_id(metadata)
            self.assertEqual(signal_fs, expected_signal_fs, f"Incorrect signal fs found in signal - Subject id is: {subject_id}, Signal id is {signal_id}.")
            
    def _check_sig_leads(self, batch):
        signal_data, signal_metadata_list, metadata_list = batch
        expected_signal_fs = 12
        for signal_data_item, signal_metadata, metadata in zip(signal_data, signal_metadata_list, metadata_list):
            signal_leads = signal_metadata["n_sig"]
            subject_id = get_subject_id(signal_metadata)
            signal_id = get_signal_id(metadata)
            self.assertEqual(signal_leads, expected_signal_fs, f"Incorrect signal Leads found in signal - Subject id is: {subject_id}, Signal id is {signal_id}.") 

def get_subject_id(signal_metadata):
    subject_id = signal_metadata["comments"]
    
    return subject_id

def get_signal_id(metadata):
    return metadata.record_name