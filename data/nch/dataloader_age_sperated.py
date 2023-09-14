import glob
import os
import random
import numpy as np
import pandas as pd
from scipy.signal import resample
from biosppy.signals.ecg import hamilton_segmenter, correct_rpeaks
from biosppy.signals import tools as st
from scipy.interpolate import splev, splrep

# "EOG LOC-M2",  # 0
# "EOG ROC-M1",  # 1
# "EEG C3-M2",  # 2
# "EEG C4-M1",  # 3
# "ECG EKG2-EKG",  # 4
#
# "RESP PTAF",  # 5
# "RESP AIRFLOW",  # 6
# "RESP THORACIC",  # 7
# "RESP ABDOMINAL",  # 8
# "SPO2",  # 9
# "CAPNO",  # 10

######### ADDED IN THIS STEP #########
# RRI #11
# Ramp #12


SIGS = [0, 3, 5, 6, 9, 10, 4]
s_count = len(SIGS)

THRESHOLD = 3
PATH = "/media/hamed/NSSR Dataset/nch_30x64/"
FREQ = 64
EPOCH_DURATION = 30
ECG_SIG = 4
OUT_PATH = "/media/hamed/NSSR Dataset/age/nch_30x64_seperated_age"




def extract_rri(signal, ir, CHUNK_DURATION):
    tm = np.arange(0, CHUNK_DURATION, step=1 / float(ir))  # TIME METRIC FOR INTERPOLATION

    filtered, _, _ = st.filter_signal(signal=signal, ftype="FIR", band="bandpass", order=int(0.3 * FREQ),
                                      frequency=[3, 31], sampling_rate=FREQ, )
    (rpeaks,) = hamilton_segmenter(signal=filtered, sampling_rate=FREQ)
    (rpeaks,) = correct_rpeaks(signal=filtered, rpeaks=rpeaks, sampling_rate=FREQ, tol=0.05)

    if 4 < len(rpeaks) < 200:  # and np.max(signal) < 0.0015 and np.min(signal) > -0.0015:
        rri_tm, rri_signal = rpeaks[1:] / float(FREQ), np.diff(rpeaks) / float(FREQ)
        ampl_tm, ampl_signal = rpeaks / float(FREQ), signal[rpeaks]
        rri_interp_signal = splev(tm, splrep(rri_tm, rri_signal, k=3), ext=1)
        amp_interp_signal = splev(tm, splrep(ampl_tm, ampl_signal, k=3), ext=1)

        return np.clip(rri_interp_signal, 0, 2), np.clip(amp_interp_signal, -0.001, 0.002)
    else:
        return np.zeros((FREQ * EPOCH_DURATION)), np.zeros((FREQ * EPOCH_DURATION))


def load_data(path):
    age = pd.read_csv(r"/media/hamed/NSSR Dataset/nchsdb/health_data/SLEEP_STUDY.csv")
    age_dict = dict(zip(age.SLEEP_STUDY_ID, age.AGE_AT_SLEEP_STUDY_DAYS / 365))
    root_dir = os.path.expanduser(path)

    folds = [[], [], [], [], [], [], [], [], [], []]
    folds_age = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 16), (16, 18), (18, 99)]

    for key, value in age_dict.items():
        for i in range(len(folds_age)):
            if folds_age[i][0] <= value < folds_age[i][1]:
                folds[i].append(key)

    for idx, fold in enumerate(folds):
        first = True
        for study in fold:
            print(study, idx)
            if len(glob.glob(PATH + "*_" + str(study) + "_*")):
                study_data = np.load(glob.glob(PATH + "*_" + str(study) + "_*")[0])

                signals = study_data['data']
                labels_apnea = study_data['labels_apnea']
                labels_hypopnea = study_data['labels_hypopnea']

                y_c = labels_apnea + labels_hypopnea
                neg_samples = np.where(y_c == 0)[0]
                pos_samples = list(np.where(y_c > 0)[0])
                ratio = len(pos_samples) / len(neg_samples)
                neg_survived = []
                for s in range(len(neg_samples)):
                    if random.random() < ratio:
                        neg_survived.append(neg_samples[s])
                samples = neg_survived + pos_samples
                signals = signals[samples, :, :]
                labels_apnea = labels_apnea[samples]
                labels_hypopnea = labels_hypopnea[samples]

                data = np.zeros((signals.shape[0], EPOCH_DURATION * FREQ, s_count + 2))
                for i in range(signals.shape[0]):  # for each epoch
                    data[i, :, -1], data[i, :, -2] = extract_rri(signals[i, ECG_SIG, :], FREQ, float(EPOCH_DURATION))
                    for j in range(s_count):  # for each signal
                        data[i, :, j] = resample(signals[i, SIGS[j], :], EPOCH_DURATION * FREQ)

                if first:
                    aggregated_data = data
                    aggregated_label_apnea = labels_apnea
                    aggregated_label_hypopnea = labels_hypopnea
                    first = False
                else:
                    aggregated_data = np.concatenate((aggregated_data, data), axis=0)
                    aggregated_label_apnea = np.concatenate((aggregated_label_apnea, labels_apnea), axis=0)
                    aggregated_label_hypopnea = np.concatenate((aggregated_label_hypopnea, labels_hypopnea), axis=0)

        np.savez_compressed(OUT_PATH + "_" + str(idx), x=aggregated_data, y_apnea=aggregated_label_apnea,
                            y_hypopnea=aggregated_label_hypopnea)


if __name__ == "__main__":
    load_data(PATH)
