from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import sys
import matplotlib.pyplot as plt
import wfdb
import scipy.stats.mstats as mstats
from scipy.io import loadmat
from scipy import signal
from tqdm import tqdm
import neurokit2 as nk
# from keras.preprocessing.sequence import pad_sequences
from pathlib import Path
from scipy.signal import butter, filtfilt
from ECGMetaData import ECGMetaData
from ECGGenerator import ECGGenerator


SAMPLE_RATE = 500

def get_relevant_indexes(dr, indexes):
    return [i for i in indexes if dr.load_data(i).shape[1] >= dr.get_fs(i) * 9.5]


def load_indexes_file(path):
    indexes = []
    with open(path, 'r') as indexes_file:
        indexes.extend(int(line.strip()) for line in indexes_file)
    return indexes


def is_signal_good(signal):
    # computes the variance for each lead. if the variance is below 0.004
    # then the lead is considered bad
    count_num_bad_leads = 0
    for lead in range(len(signal)):
        describe = mstats.describe(signal[lead])
        if describe.variance < 0.004:
            count_num_bad_leads += 1
    return count_num_bad_leads < 3


def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file, 'r') as f:
        header_data = f.readlines()
    return data, header_data


def import_ecg_data(directory, ecg_len=5000, trunc="post", pad="post"):
    """
    get ecg data from a directory after padding or truncating if needed.
    in WPW all the data is at least 5000 and without any Nans
    :param directory:
    :param ecg_len:
    :param trunc:
    :param pad:
    :return:
    """
    print("Starting ECG import..")
    ecgs = []
    name_mapping = []
    for i, ecgfilename in enumerate(tqdm(sorted(os.listdir(directory)))):
        # if i not in [167, 171, 317, 319]: #the file numbers with the longer duration
        #     continue
        filepath = directory + os.sep + ecgfilename
        if filepath.endswith(".mat"):
            data, header_data = load_challenge_data(filepath)
            # if data.shape != (12, 5000):  # TODO: note: there are larger files, can we use them for more samples?
            #     print(f"file num {i}, {ecgfilename}, need to be padded or truncated, original size is {data.shape}")
            # data = pad_sequences(data, maxlen=ecg_len, truncating=trunc, padding=pad)
            ecgs.append(data)
            name_mapping.append(ecgfilename.split('.')[0])
    print("Finished!")
    return np.asarray(ecgs), name_mapping

def process_files(file_pair):
    header_path, signal_path = file_pair
    study_file = header_path.stem
    metadata = wfdb.rdheader(f'{header_path.parent}/{study_file}')
    patient_id = metadata.comments
    patient_id = [column.split(":")[1].strip() for column in patient_id][0]
    study_signal = signal_path.stem
    signal_data, signal_metadata = wfdb.rdsamp(f'{signal_path.parent}/{study_signal}')
    return signal_data.T, (patient_id, study_file)


def import_ecg_mimic_data(directory, ecg_len=5000, trunc="post", pad="post", num_of_ecgs_to_test=None):
    print("Starting ECG import..")
    ecgs = []
    name_mapping = []
    header_paths = directory.rglob("*.hea")
    signal_paths = directory.rglob("*.dat")

    ecgs = []
    name_mapping = []
    
    # Pair the header and signal paths together
    file_pairs = zip(header_paths, signal_paths)
    if num_of_ecgs_to_test is not None:
        file_pairs = islice(file_pairs, num_of_ecgs_to_test)

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_files, file_pairs)
        
        for ecg_data, ecg_name_mapping in results:
            ecgs.append(ecg_data)
            name_mapping.append(ecg_name_mapping)
            
            if num_of_ecgs_to_test and len(ecgs) >= num_of_ecgs_to_test:
                break

    print("Finished reading all data!")
    return np.asarray(ecgs), name_mapping


def resample_beats(beats):
    rsmp_beats = []
    for i in beats:
        i = np.asarray(i)

        # i = i[~np.isnan(i)]
        f = signal.resample(i, 250)
        rsmp_beats.append(f)
    rsmp_beats = np.asarray(rsmp_beats)
    return rsmp_beats


def median_beat(beat_dict):
    beats = []
    for i in beat_dict.values():
        # print(i['Signal'])
        beats.append(i['Signal'])
    beats = np.asarray(beats)
    rsmp_beats = resample_beats(beats)
    med_beat = np.median(rsmp_beats, axis=0)
    return med_beat


def process_ecgs(raw_ecg):
    processed_ecgs = []
    for i in tqdm(range(len(raw_ecg))):
        leadII = raw_ecg[i][1]
        leadII_clean = nk.ecg_clean(leadII, sampling_rate=SAMPLE_RATE, method="neurokit")
        r_peaks = nk.ecg_findpeaks(leadII_clean, sampling_rate=SAMPLE_RATE, method="neurokit", show=False)
        twelve_leads = []
        for lead_index, raw_lead_signal in enumerate(raw_ecg[i]):
            try:
                beats = nk.ecg_segment(raw_lead_signal, rpeaks=r_peaks['ECG_R_Peaks'], sampling_rate=SAMPLE_RATE,
                                       show=False)
                # note: the beats dict sometimes misses his last bit making it only Nans and zeroes. it should pad with
                # nans the last beats but sometimes it is just zeroes and nans
                last_bit_key = str(np.max(np.array(list(beats.keys()), dtype=int)))
                last_bit_signal_values = np.sort(beats[last_bit_key]['Signal'].unique())
                if np.array_equal(last_bit_signal_values, np.array([0, np.nan]), equal_nan=True):
                    print(f"problematic last heart beat in ECG num {i}, lead {lead_index}")
                med_beat = median_beat(beats)
                twelve_leads.append(med_beat)
                # print(f"try success in file {i}, lead {raw_lead_signal}")
            except:
                beats = np.ones(250) * np.nan
                twelve_leads.append(beats)
                # print(f"try failed in file {i}, lead {raw_lead_signal}") #in wpw, all beats succeed in the try
        # twelve_leads = np.asarray(twelve_leads)
        processed_ecgs.append(twelve_leads)
    processed_ecgs = np.asarray(processed_ecgs)
    return processed_ecgs


def remove_some_ecgs(ecg_arr):
    delete_list = []
    for i in tqdm(range(len(ecg_arr))):
        if np.all(ecg_arr[i].T[0] == 1):
            delete_list.append(i)
    ecg_arr = np.delete(ecg_arr, delete_list, axis=0)
    return ecg_arr


def remove_nans(ecg_arr):
    new_arr = []
    for i in tqdm(ecg_arr):
        twelve_lead = []
        for j in i:
            if j[0] != j[0]:
                j = np.ones(250)
            twelve_lead.append(j)
        new_arr.append(twelve_lead)
    new_arr = np.asarray(new_arr)
    return new_arr

def main(ecg_formats,input_data_dir=Path("./files")):
    # np.seterr(all='raise')
    SAMPLE_RATE = 500
    ECG_LEN =  SAMPLE_RATE*10
    ECG_IMAGE_SIZE = (1650, 880)
    TO_FILTER = False
    # input_data_dir = Path("./files")
    for ecg_format in ecg_formats:
        if ecg_format == 0:
            output_images_dir_format_0 = os.path.join(os.getcwd(), 'images_format_0')
            os.makedirs(output_images_dir_format_0, exist_ok=True)
        elif ecg_format == 1:
            output_images_dir_format_1 = os.path.join(os.getcwd(), 'images_format_1')
            os.makedirs(output_images_dir_format_1, exist_ok=True)
        elif ecg_format == 2:
            output_images_dir_format_2 = os.path.join(os.getcwd(), 'images_format_2')
            os.makedirs(output_images_dir_format_2, exist_ok=True)
        elif ecg_format == 3:
            output_images_dir_format_3 = os.path.join(os.getcwd(), 'images_format_3')
            os.makedirs(output_images_dir_format_3, exist_ok=True)
        elif ecg_format == 4:
            output_images_dir_format_4 = os.path.join(os.getcwd(), 'images_format_4')
            os.makedirs(output_images_dir_format_4, exist_ok=True)

    lead_index = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    ECGMetaDataOptionsLocal = [
        ECGMetaData(ecg_len=ECG_LEN, long_lead_indexes=[6, 1, 10], format_id=0, sample_rate=SAMPLE_RATE,
                    lead_index=lead_index),
        ECGMetaData(ecg_len=ECG_LEN, long_lead_indexes=[1], format_id=1, sample_rate=SAMPLE_RATE,
                    lead_index=lead_index),
        ECGMetaData(ecg_len=ECG_LEN, long_lead_indexes=[1, 6], format_id=2, sample_rate=SAMPLE_RATE,
                    lead_index=lead_index),
        ECGMetaData(ecg_len=ECG_LEN, columns=2, format_id=3, sample_rate=SAMPLE_RATE,
                    lead_index=lead_index),
        ECGMetaData(ecg_len=ECG_LEN, columns=2, long_lead_indexes=[1], format_id=4, sample_rate=SAMPLE_RATE,
                    lead_index=lead_index),
    ]

    num_of_ecgs_to_test = None #1000  # None for all
    full_ecg, full_name_mapping = import_ecg_mimic_data(input_data_dir, ecg_len=ECG_LEN, num_of_ecgs_to_test=num_of_ecgs_to_test)
    ecg_signals = full_ecg[:num_of_ecgs_to_test]
    name_mapping = full_name_mapping[:num_of_ecgs_to_test]

    for ecg_sample_index, (ecg_sample, ecg_signal_name) in enumerate(zip(ecg_signals, name_mapping)):
        ecg_signal_patiend_id = ecg_signal_name[0]
        ecg_signal_study_id = ecg_signal_name[1]
        clean_ecg_sample = []
        problematic_ecg_lead = 0  # id zero, no problems in sample, else an index of a problematic lead

        for ecg_lead_index, ecg_lead in enumerate(ecg_sample):
            #This is Filters we dosent need
            # ecg_signal = nk.signal_sanitize(ecg_lead)
            # ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=SAMPLE_RATE, method="neurokit")

            if TO_FILTER:
                # filter bad signals using signal quality and signal variance and r peaks
                # with lower then 3 r peaks cannot calculate heart rate
                if np.var(ecg_cleaned) < 0.004: #Was 0.004
                    problematic_ecg_lead = ecg_lead_index
                    break #can also continue instead of break to skip the lead and keep the rest of the leads

                (instant_peaks, rpeaks,) = nk.ecg_peaks(
                    ecg_cleaned=ecg_cleaned, sampling_rate=SAMPLE_RATE, method="neurokit", correct_artifacts=True
                )
                if len(rpeaks['ECG_R_Peaks']) <= 3:
                    problematic_ecg_lead = ecg_lead_index
                    break

                print(len(rpeaks['ECG_R_Peaks']))
                quality_rank = nk.ecg_quality(ecg_cleaned, rpeaks=rpeaks["ECG_R_Peaks"], sampling_rate=SAMPLE_RATE,
                                            method="zhao2018")

                quality = nk.ecg_quality(ecg_cleaned, rpeaks=rpeaks["ECG_R_Peaks"], sampling_rate=SAMPLE_RATE)

                print(f"ECG file {ecg_signal_name} ({ecg_sample_index}) with lead_index: {ecg_lead_index} "
                    f"has a {quality_rank} quality")
            # clean_ecg_sample.append(ecg_cleaned)
        if problematic_ecg_lead != 0:
            print(f"while processing ECG file {ecg_signal_name}, indexes: {ecg_sample_index}, "
                  f"found problematic lead number {problematic_ecg_lead}, name {lead_index[problematic_ecg_lead]}"
                  f". Did not create image")
            continue
        # clean_ecg_sample = np.array(clean_ecg_sample)
        clean_ecg_sample = ecg_sample
        for ecg_format in ecg_formats:
            try:
                if(ecg_format == 0):
                    output_images_dir = output_images_dir_format_0
                elif(ecg_format == 1):
                    output_images_dir = output_images_dir_format_1
                elif(ecg_format == 2):
                    output_images_dir = output_images_dir_format_2
                elif(ecg_format == 3):
                    output_images_dir = output_images_dir_format_3
                elif(ecg_format == 4):
                    output_images_dir = output_images_dir_format_4
                ecg_meta_data = ECGMetaDataOptionsLocal[ecg_format]
                img_generator = ECGGenerator(clean_ecg_sample, ecg_meta_data, to_preprocess=True)
                img = img_generator.get_numpy_array()
                resized_img = cv2.resize(img, ECG_IMAGE_SIZE)
                Image.fromarray(resized_img).save(
                    f'{output_images_dir}/{ecg_signal_patiend_id}_{ecg_signal_study_id}.png')
            except ValueError:
                print(f'Could not render ecg {ecg_signal_name} with format {ecg_format}')

if __name__ == '__main__':
    ecg_formats = [0]
    main(ecg_formats)