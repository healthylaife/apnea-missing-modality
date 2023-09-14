import glob
import os
import random
import numpy as np
import pandas as pd
from scipy.signal import resample
from biosppy.signals.ecg import hamilton_segmenter, correct_rpeaks
from biosppy.signals import tools as st
from scipy.interpolate import splev, splrep

# 0- E1 - M2
# 1- E2 - M1

# 2- F3 - M2
# 3- F4 - M1
# 4- C3 - M2
# 5- C4 - M1
# 6- O1 - M2
# 7- O2 - M1

# 8- ECG3 - ECG1

# 9- CANULAFLOW
# 10- AIRFLOW
# 11- CHEST
# 12- ABD

# 13- SAO2
# 14- CAP
######### ADDED IN THIS STEP #########
# 15- RRI
# 16 Ramp

SIGS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
s_count = len(SIGS)

THRESHOLD = 3
PATH = "/media/hamed/CHAT Dataset/chat_b_30x64/"
FREQ = 128
OUT_FREQ = 64
EPOCH_LENGTH = 30
ECG_SIG = 8
OUT_PATH = "/media/hamed/CHAT Dataset/chat_b_30x64_"

SIGS = [0, 5, 9, 10, 13, 14, 8]
s_count = len(SIGS)

def extract_rri(signal, ir, CHUNK_DURATION):
    tm = np.arange(0, CHUNK_DURATION, step=1 / float(ir))  # TIME METRIC FOR INTERPOLATION

    filtered, _, _ = st.filter_signal(signal=signal, ftype="FIR", band="bandpass", order=int(0.3 * FREQ),
                                      frequency=[3, 45], sampling_rate=FREQ, )
    (rpeaks,) = hamilton_segmenter(signal=filtered, sampling_rate=FREQ)
    (rpeaks,) = correct_rpeaks(signal=filtered, rpeaks=rpeaks, sampling_rate=FREQ, tol=0.05)

    if 4 < len(rpeaks) < 200:  # and np.max(signal) < 0.0015 and np.min(signal) > -0.0015:
        rri_tm, rri_signal = rpeaks[1:] / float(FREQ), np.diff(rpeaks) / float(FREQ)
        ampl_tm, ampl_signal = rpeaks / float(FREQ), signal[rpeaks]
        rri_interp_signal = splev(tm, splrep(rri_tm, rri_signal, k=3), ext=1)
        amp_interp_signal = splev(tm, splrep(ampl_tm, ampl_signal, k=3), ext=1)

        return np.clip(rri_interp_signal, 0, 2) * 100, np.clip(amp_interp_signal, -0.001, 0.002) * 10000, True
    else:
        return np.zeros((FREQ * EPOCH_LENGTH)), np.zeros((FREQ * EPOCH_LENGTH)), False


def load_data(path):
    # demo = pd.read_csv("../misc/result.csv")
    # ahi = pd.read_csv(r"C:\Data\AHI.csv")
    # ahi_dict = dict(zip(ahi.Study, ahi.AHI))
    root_dir = os.path.expanduser(path)
    file_list = os.listdir(root_dir)
    length = len(file_list)

    ################################### Fold the data based on number of respiratory events #########################
    study_event_counts = [i for i in range(0, length)]
    folds = []
    for i in range(5):
        folds.append(study_event_counts[i::5])

    counter = 0
    for idx, fold in enumerate(folds):
        first = True
        for patient in fold:
            rri_succ_counter = 0
            rri_fail_counter = 0
            counter += 1
            print(counter)
            # for study in glob.glob(PATH + patient[0] + "_*"):
            study_data = np.load(PATH + file_list[patient - 1])

            signals = study_data['data']
            labels_apnea = study_data['labels_apnea']
            labels_hypopnea = study_data['labels_hypopnea']

            # identifier = study.split('\\')[-1].split('_')[0] + "_" + study.split('\\')[-1].split('_')[1]
            # demo_arr = demo[demo['id'] == identifier].drop(columns=['id']).to_numpy().squeeze()

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

            data = np.zeros((signals.shape[0], EPOCH_LENGTH * OUT_FREQ, s_count + 2))
            for i in range(signals.shape[0]):  # for each epoch
                # data[i, :len(demo_arr), -3] = demo_arr
                rri, r_amp, status = extract_rri(signals[i, ECG_SIG, :], FREQ,float(EPOCH_LENGTH))
                data[i, :, -1] = resample(rri, EPOCH_LENGTH * OUT_FREQ)
                data[i, :, -2] = resample(r_amp, EPOCH_LENGTH * OUT_FREQ)
                if status:
                    rri_succ_counter += 1
                else:
                    rri_fail_counter += 1

                for j in range(s_count):  # for each signal
                    data[i, :, j] = resample(signals[i, SIGS[j], :], EPOCH_LENGTH * OUT_FREQ)

            if first:
                aggregated_data = data
                aggregated_label_apnea = labels_apnea
                aggregated_label_hypopnea = labels_hypopnea
                first = False
            else:
                aggregated_data = np.concatenate((aggregated_data, data), axis=0)
                aggregated_label_apnea = np.concatenate((aggregated_label_apnea, labels_apnea), axis=0)
                aggregated_label_hypopnea = np.concatenate((aggregated_label_hypopnea, labels_hypopnea), axis=0)
            print(rri_succ_counter, rri_fail_counter)

        np.savez_compressed(OUT_PATH + "_" + str(idx), x=aggregated_data, y_apnea=aggregated_label_apnea,
                            y_hypopnea=aggregated_label_hypopnea)



if __name__ == "__main__":
    load_data(PATH)
