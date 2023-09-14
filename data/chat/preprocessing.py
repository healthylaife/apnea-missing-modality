import glob
import os
from datetime import datetime
from itertools import compress

import mne
import numpy as np
import pandas as pd
from mne import make_fixed_length_events

THRESHOLD = 3
NUM_WORKER = 1
SN = 3984  # STUDY NUMBER
FREQ = 128.0
EPOCH_LENGTH = 30.0
OUT_FOLDER = '/media/hamed/CHAT Dataset/chat_b_30x64/'

channels = [
    'E1',  # 0
    'E2',  # 1
    'F3',  # 2
    'F4',  # 3
    'C3',  # 4
    'C4',  # 5
    'M1',  # 6
    'M2',  # 7
    'O1',  # 8
    'O2',  # 9
    'ECG1',  # 10
    'ECG3',  # 11

    'CANNULAFLOW',  # 12
    'AIRFLOW',  # 13
    'CHEST',  # 14
    'ABD',  # 15
    'SAO2',  # 16
    'CAP',  # 17
]

APNEA_EVENT_DICT = {
    "Obstructive apnea": 2,
    "Central apnea": 2,
}

HYPOPNEA_EVENT_DICT = {
    "Hypopnea": 1,
}

POS_EVENT_DICT = {
    "Hypopnea": 1,

    "Obstructive apnea": 2,
    "Central apnea": 2,
}

NEG_EVENT_DICT = {
    'Stage 1 sleep': 0,
    'Stage 2 sleep': 0,
    'Stage 3 sleep': 0,
    'REM sleep': 0,
}

WAKE_DICT = {
    "Wake": 10
}

mne.set_log_file('log.txt', overwrite=False)


########################################## Annotation Modifier functions ##########################################
def identity(df):
    return df


def apnea2bad(df):
    df = df.replace(r'.*pnea.*', 'badevent', regex=True)
    print("bad replaced!")
    return df


def wake2bad(df):
    return df.replace("Sleep stage W", 'badevent')


def change_duration(df, label_dict=POS_EVENT_DICT, duration=EPOCH_LENGTH):
    for key in label_dict:
        df.loc[df.description == key, 'duration'] = duration
    print("change duration!")
    return df


def load_study_chat(edf_path, annotation_path, annotation_func, preload=False, exclude=[], verbose='CRITICAL'):
    raw = mne.io.edf.edf.RawEDF(input_fname=edf_path, exclude=exclude, preload=preload, verbose=verbose)

    df = annotation_func(pd.read_csv(annotation_path, sep='\t'))
    annotations = mne.Annotations(df.onset, df.duration, df.description)  # ,orig_time=new_datetime)

    raw.set_annotations(annotations)

    raw.rename_channels({name: name.upper() for name in raw.info['ch_names']})

    return raw


def preprocess(path, annotation_modifier, out_dir):
    is_apnea_available, is_hypopnea_available = True, True
    raw = load_study_chat(path[0], path[1], annotation_modifier, verbose=True)
    ########################################   CHECK CRITERIA FOR SS   #################################################
    if not all([name in raw.ch_names for name in channels]):
        print([name in raw.ch_names for name in channels])
        print("study " + os.path.basename(path[0]) + " skipped since insufficient channels")
        return 0

    try:
        apnea_events, event_ids = mne.events_from_annotations(raw, event_id=POS_EVENT_DICT, chunk_duration=1.0,
                                                              verbose=None)
    except ValueError:
        print("No Chunk found!")
        return 0
    ########################################   CHECK CRITERIA FOR SS   #################################################
    print(str(datetime.now().time().strftime("%H:%M:%S")) + ' --- Processing %s' % os.path.basename(path[0]))

    try:
        apnea_events, event_ids = mne.events_from_annotations(raw, event_id=APNEA_EVENT_DICT, chunk_duration=1.0,
                                                              verbose=None)
    except ValueError:
        is_apnea_available = False

    try:
        hypopnea_events, event_ids = mne.events_from_annotations(raw, event_id=HYPOPNEA_EVENT_DICT, chunk_duration=1.0,
                                                                 verbose=None)
    except ValueError:
        is_hypopnea_available = False

    wake_events, event_ids = mne.events_from_annotations(raw, event_id=WAKE_DICT, chunk_duration=1.0, verbose=None)
    ####################################################################################################################
    sfreq = raw.info['sfreq']
    tmax = EPOCH_LENGTH - 1. / sfreq

    raw = raw.pick_channels(channels, ordered=True)
    fixed_events = make_fixed_length_events(raw, id=0, duration=EPOCH_LENGTH, overlap=0.)
    epochs = mne.Epochs(raw, fixed_events, event_id=[0], tmin=0, tmax=tmax, baseline=None, preload=True, proj=False, verbose=None)
    epochs.load_data()
    if sfreq != FREQ:
        epochs = epochs.resample(FREQ, npad='auto', n_jobs=8, verbose=None)
    data = epochs.get_data()
    ####################################################################################################################
    if is_apnea_available:
        apnea_events_set = set((apnea_events[:, 0] / sfreq).astype(int))
    if is_hypopnea_available:
        hypopnea_events_set = set((hypopnea_events[:, 0] / sfreq).astype(int))
    wake_events_set = set((wake_events[:, 0] / sfreq).astype(int))

    starts = (epochs.events[:, 0] / sfreq).astype(int)

    labels_apnea = []
    labels_hypopnea = []
    labels_not_awake = []
    total_apnea_event_second = 0
    total_hypopnea_event_second = 0

    for seq in range(data.shape[0]):
        epoch_set = set(range(starts[seq], starts[seq] + int(EPOCH_LENGTH)))
        if is_apnea_available:
            apnea_seconds = len(apnea_events_set.intersection(epoch_set))
            total_apnea_event_second += apnea_seconds
            labels_apnea.append(apnea_seconds)
        else:
            labels_apnea.append(0)

        if is_hypopnea_available:
            hypopnea_seconds = len(hypopnea_events_set.intersection(epoch_set))
            total_hypopnea_event_second += hypopnea_seconds
            labels_hypopnea.append(hypopnea_seconds)
        else:
            labels_hypopnea.append(0)

        labels_not_awake.append(len(wake_events_set.intersection(epoch_set)) == 0)
    ####################################################################################################################
    data = data[labels_not_awake, :, :]
    labels_apnea = list(compress(labels_apnea, labels_not_awake))
    labels_hypopnea = list(compress(labels_hypopnea, labels_not_awake))
    ####################################################################################################################

    new_data = np.zeros_like(data)
    for i in range(data.shape[0]):

        new_data[i, 0, :] = data[i, 0, :] - data[i, 7, :]  # E1 - M2
        new_data[i, 1, :] = data[i, 1, :] - data[i, 6, :]  # E2 - M1

        new_data[i, 2, :] = data[i, 2, :] - data[i, 7, :]  # F3 - M2
        new_data[i, 3, :] = data[i, 3, :] - data[i, 6, :]  # F4 - M1
        new_data[i, 4, :] = data[i, 4, :] - data[i, 7, :]  # C3 - M2
        new_data[i, 5, :] = data[i, 5, :] - data[i, 6, :]  # C4 - M1
        new_data[i, 6, :] = data[i, 8, :] - data[i, 7, :]  # O1 - M2
        new_data[i, 7, :] = data[i, 9, :] - data[i, 6, :]  # O2 - M1

        new_data[i, 8, :] = data[i, 10,:] - data[i, 11,:]  # ECG

        new_data[i, 9, :] = data[i, 12, :]  # CANULAFLOW
        new_data[i, 10, :] = data[i, 13, :]  # AIRFLOW
        new_data[i, 11, :] = data[i, 14, :]  # CHEST
        new_data[i, 12, :] = data[i, 15, :]  # ABD
        new_data[i, 13, :] = data[i, 16, :]  # SAO2
        new_data[i, 14, :] = data[i, 17, :]  # CAP
    data = new_data[:, :15, :]
    ####################################################################################################################

    np.savez_compressed(
        out_dir + '/' + os.path.basename(path[0]) + "_" + str(total_apnea_event_second) + "_" + str(
            total_hypopnea_event_second),
        data=data, labels_apnea=labels_apnea, labels_hypopnea=labels_hypopnea)

    return data.shape[0]


if __name__ == "__main__":
    root = "/media/hamed/CHAT Dataset/chat/polysomnography/edfs/baseline/"
    for edf_file in glob.glob(root + "*.edf"):
        annot_file = edf_file.replace(".edf", "-nsrr.tsv")

        preprocess((edf_file, annot_file), identity, OUT_FOLDER)
