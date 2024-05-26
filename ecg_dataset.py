import wfdb
import torch
from torch.utils.data import Dataset
import numpy as np

class ECGDataset(Dataset):
    def __init__(self, patients_group_directory):
        self.patients_group_directory = patients_group_directory
        self.signal_paths = list(patients_group_directory.rglob("**/*.dat"))
        self.header_paths = list(patients_group_directory.rglob("**/*.hea"))

        self.image_ids = []
        self.study_ids = []
        self.subject_ids = []
    
    def __len__(self):
        return len(self.signal_paths)
    
    def __getitem__(self, idx):
        signal_path = self.signal_paths[idx]
        header_path = self.header_paths[idx]
        signal_data, signal_metadata = wfdb.rdsamp(f'{signal_path.parent}/{signal_path.stem}')
        metadata = wfdb.rdheader(f'{header_path.parent}/{header_path.stem}')

        subject_id = metadata.comments
        subject_id = [column.split(":")[1].strip() for column in subject_id][0]
        study_id = metadata.record_name


        self.subject_ids.append(subject_id)
        self.study_ids.append(study_id)

        image_id = f"{subject_id}_{study_id}"
        self.image_ids.append(image_id)

        return (signal_data.T,signal_metadata,metadata)
    
    def collate_fn(self, batch):
        signal_data_list, signal_metadata_list, metadata_list = zip(*batch)
        signal_data_array = np.array(signal_data_list)
        signal_data_tensor = torch.tensor(signal_data_array)
        return signal_data_tensor, signal_metadata_list, metadata_list