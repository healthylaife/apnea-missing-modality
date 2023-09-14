import concurrent.futures
from datetime import datetime
import pandas as pd
import mne
import numpy as np
from mne import make_fixed_length_events
from itertools import compress
from data import sleep_study as ss

THRESHOLD = 3
NUM_WORKER = 8
SN = 3984  # STUDY NUMBER
FREQ = 64.0
CHUNK_DURATION = 30.0
OUT_FOLDER = 'D:\\nch_30x64'

# channels = [
#     "EOG LOC-M2",  # 0
#     "EOG ROC-M1",  # 1
#     "EEG F3-M2",  # 2
#     "EEG F4-M1",  # 3
#     "EEG C3-M2",  # 4
#     "EEG C4-M1",  # 5
#     "EEG O1-M2",  # 6
#     "EEG O2-M1",  # 7
#     "EEG CZ-O1",  # 8
#     "ECG EKG2-EKG",  # 9
#     "RESP PTAF",  # 10
#     "RESP AIRFLOW",  # 11
#     "RESP THORACIC",  # 12
#     "RESP ABDOMINAL",  # 13
#     "SPO2",  # 14
#     "RATE",  # 15
#     "CAPNO",  # 16
#     "RESP RATE",  # 17
# ]


channels = [
    "EOG LOC-M2",  # 0
    "EOG ROC-M1",  # 1
    "EEG C3-M2",  # 2
    "EEG C4-M1",  # 3
    "ECG EKG2-EKG",  # 4

    "RESP PTAF",  # 5
    "RESP AIRFLOW",  # 6
    "RESP THORACIC",  # 7
    "RESP ABDOMINAL",  # 8
    "SPO2",  # 9
    "CAPNO",  # 10
]

APNEA_EVENT_DICT = {
    "Obstructive Apnea": 2,
    "Central Apnea": 2,
    "Mixed Apnea": 2,
    "apnea": 2,
    "obstructive apnea": 2,
    "central apnea": 2,
    "apnea": 2,
    "Apnea": 2,
}

HYPOPNEA_EVENT_DICT = {
    "Obstructive Hypopnea": 1,
    "Hypopnea": 1,
    "hypopnea": 1,
    "Mixed Hypopnea": 1,
    "Central Hypopnea": 1,
}

POS_EVENT_DICT = {
    "Obstructive Hypopnea": 1,
    "Hypopnea": 1,
    "hypopnea": 1,
    "Mixed Hypopnea": 1,
    "Central Hypopnea": 1,

    "Obstructive Apnea": 2,
    "Central Apnea": 2,
    "Mixed Apnea": 2,
    "apnea": 2,
    "obstructive apnea": 2,
    "central apnea": 2,
    "Apnea": 2,
}

NEG_EVENT_DICT = {
    'Sleep stage N1': 0,
    'Sleep stage N2': 0,
    'Sleep stage N3': 0,
    'Sleep stage R': 0,
}

WAKE_DICT = {
    "Sleep stage W": 10
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


def change_duration(df, label_dict=POS_EVENT_DICT, duration=CHUNK_DURATION):
    for key in label_dict:
        df.loc[df.description == key, 'duration'] = duration
    print("change duration!")
    return df

def preprocess(i, annotation_modifier, out_dir, ahi_dict):
    is_apnea_available, is_hypopnea_available = True, True
    study = data.sleep_study.data.study_list[i]
    raw = data.sleep_study.data.load_study(study, annotation_modifier, verbose=True)
    ########################################   CHECK CRITERIA FOR SS   #################################################
    if not all([name in raw.ch_names for name in channels]):
        print("study " + str(study) + " skipped since insufficient channels")
        return 0

    if ahi_dict.get(study, 0) < THRESHOLD:
        print("study " + str(study) + " skipped since low AHI ---  AHI = " + str(ahi_dict.get(study, 0)))
        return 0

    try:
        apnea_events, event_ids = mne.events_from_annotations(raw, event_id=POS_EVENT_DICT, chunk_duration=1.0,
                                                              verbose=None)
    except ValueError:
        print("No Chunk found!")
        return 0
    ########################################   CHECK CRITERIA FOR SS   #################################################
    print(str(i) + "---" + str(datetime.now().time().strftime("%H:%M:%S")) + ' --- Processing %d' % i)

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
    tmax = CHUNK_DURATION - 1. / sfreq

    raw = raw.pick_channels(channels, ordered=True)
    fixed_events = make_fixed_length_events(raw, id=0, duration=CHUNK_DURATION, overlap=0.)
    epochs = mne.Epochs(raw, fixed_events, event_id=[0], tmin=0, tmax=tmax, baseline=None, preload=True, proj=False, verbose=None)
    epochs.load_data()
    if sfreq != FREQ:
        epochs = epochs.resample(FREQ, npad='auto', n_jobs=4, verbose=None)
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
    labels_wake = []
    total_apnea_event_second = 0
    total_hypopnea_event_second = 0

    for seq in range(data.inp_dim[0]):
        epoch_set = set(range(starts[seq], starts[seq] + int(CHUNK_DURATION)))
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

        labels_wake.append(len(wake_events_set.intersection(epoch_set)) == 0)
    ####################################################################################################################
    print(study + "    HAMED    " + str(len(labels_wake) - sum(labels_wake)))
    data = data[labels_wake, :, :]
    labels_apnea = list(compress(labels_apnea, labels_wake))
    labels_hypopnea = list(compress(labels_hypopnea, labels_wake))

    np.savez_compressed(
        out_dir + '\\' + study + "_" + str(total_apnea_event_second) + "_" + str(total_hypopnea_event_second),
        data=data, labels_apnea=labels_apnea, labels_hypopnea=labels_hypopnea)

    return data.inp_dim[0]


if __name__ == "__main__":
    ahi = pd.read_csv(r"D:\Data\AHI.csv")
    ahi_dict = dict(zip(ahi.Study, ahi.AHI))
    ss.__init__()

    if NUM_WORKER < 2:
        for idx in range(SN):
            preprocess(idx, identity, OUT_FOLDER, ahi_dict)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKER) as executor:
            executor.map(preprocess, range(SN), [identity] * SN, [OUT_FOLDER] * SN, [ahi_dict] * SN)
