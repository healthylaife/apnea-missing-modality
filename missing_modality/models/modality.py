import gc
import random

import numpy as np
from scipy import signal


class Modality:
    def __init__(self, name, index, inp_dim, z_dim, need_freq=False, need_reshape=False):
        self.name = name
        self.index = index
        self.inp_dim = inp_dim
        self.z_dim = z_dim
        self.dim = len(inp_dim) - 1
        self.inp = None
        self.dec = None
        self.enc = None
        self.cls = None
        self.q = None  # quality Score
        ### fusion variables ###
        self.f_enc = None
        self.f_enc_flat = None
        self.f_q = None
        self.f_q_c = None
        self.f_q_f = None
        self.f_h = None
        self.f_z = None
        self.f_zh = None
        ### data ###
        self.x_train = None
        self.x_test = None
        self.need_freq = need_freq
        self.need_reshape = need_reshape

def delete_data(ms):
    for m in ms:
        del m.x_test
        del m.x_train
        print("delete DATA")
        gc.collect()



def get_inps(ms):
    inps = []
    for m in ms:
        inps.append(m.inp)
    return inps


def get_encs(ms):
    encs = []
    for m in ms:
        encs.append(m.enc)
    return encs


def get_decs(ms):
    decs = []
    for m in ms:
        decs.append(m.dec)
    return decs


def get_clss(ms):
    clss = []
    for m in ms:
        clss.append(m.cls)
    return clss


def get_q_s(ms):
    a_s = []
    for m in ms:
        a_s.append(m.q)
    return a_s


def get_f_encs(ms):
    f_enc = []
    for m in ms:
        f_enc.append(m.f_enc)
    return f_enc


def get_f_enc_flats(ms):
    f_enc_flat = []
    for m in ms:
        f_enc_flat.append(m.f_enc_flat)
    return f_enc_flat


def get_f_a_s(ms):
    f_a_s = []
    for m in ms:
        f_a_s.append(m.f_q)
    return f_a_s


def get_f_enc_flats(ms):
    f_enc_flat = []
    for m in ms:
        f_enc_flat.append(m.f_enc_flat)
        f_enc_flat.append(m.f_q_f)
    return f_enc_flat


def get_x_train(ms):
    x_train = []
    for m in ms:
        x_train.append(m.x_train)
    return x_train


def get_x_test(ms):
    x_test = []
    for m in ms:
        x_test.append(m.x_test)
    return x_test


def generate_loss(m_list, dec_loss='mae', cls_loss='binary_crossentropy'):
    loss = {}
    for m in m_list:
        loss[m.name + '_dec'] = dec_loss
        loss[m.name + '_cls'] = cls_loss
    return loss


def generate_loss_weights(m_list):
    loss_weights = {}
    for m in m_list:
        loss_weights[m.name + '_dec'] = 1
        loss_weights[m.name + '_cls'] = 1
    return loss_weights


def load_data(m_list, x_train, x_test, miss_ratio, noise_ratio, noise_chance, miss_index, return_data=False):
    ###########   Miss Some Data     ##############################
    if miss_ratio > 0:
        print("miss ratio " + str(miss_ratio))
        for i in range(x_test.shape[0]):
            for j in range(x_test.shape[2]):
                if random.random() < miss_ratio:
                    x_test[i, :, j] = np.zeros_like(x_test[i, :, j])
                    # x_train[i, :, j] = np.zeros_like(x_train[i, :, j])
    ###########   Miss Specific Channels    #######################
    if miss_index is not None:
        print("miss channel " + str(miss_index))
        for i in range(x_test.shape[0]):
            for j in miss_index:
                    x_test[i, :, j] = np.zeros_like(x_test[i, :, j])
    ###########   ADD Some Noise     ##############################
    if noise_ratio > 0:
        print("noise ratio " + str(noise_ratio))
        add_noise_to_data(x_test, noise_ratio, noise_chance)
    ###############################################################
    ###############################################################
    if return_data:
        if x_train is not None:
            # for i in [0, 1, 3, 4, 5, 7, 8]:
            #     x_train[:, :, i] = normalize(x_train[:, :, i])
            x_train = x_train[:, :, [0, 1, 3, 4, 5, 7, 8]]
        # for i in [0, 1, 3, 4, 5, 7, 8]:
        #     x_test[:, :, i] = normalize(x_test[:, :, i])
        x_test = x_test[:, :, [0, 1, 3, 4, 5, 7, 8]]

        return normalize(x_train), normalize(x_test)
    else:
        for m in m_list:
            if m.need_freq:
                m.x_train = transform2freq(x_train, m.index)
                m.x_test = transform2freq(x_test, m.index)
            elif m.need_reshape:
                m.x_train = resize(x_train, m.index)
                m.x_test = resize(x_test, m.index)
            else:
                m.x_train = x_train[:, :, m.index]
                m.x_test = x_test[:, :, m.index]

            ###########################################################
            m.x_train = normalize(m.x_train)
            m.x_test = normalize(m.x_test)


def normalize(xx):
    if xx is None:
        return None

    if len(xx.shape) == 4:
        for i in range(xx.shape[-1]):
            x = xx[:, :, :, i]
            x = np.clip(x, np.percentile(x, 0.1), np.percentile(x, 99.9))
            if np.max(x) - np.min(x) != 0:
                x = (x - np.min(x)) / (np.max(x) - np.min(x))
                xx[:, :, :, i] = x
    else:
        for i in range(xx.shape[-1]):
            x = xx[:, :, i]
            x = np.clip(x, np.percentile(x, 0.1), np.percentile(x, 99.9))
            if np.max(x) - np.min(x) != 0:
                x = (x - np.min(x)) / (np.max(x) - np.min(x))
                xx[:, :, i] = x
    return xx


############################################## NOISE/MISSING MODALITY ##################################################
def add_noise_to_signal(signal, target_snr_db=20):
    signal_watts = signal ** 2
    # Calculate signal power and convert to dB
    sig_avg_watts = np.mean(signal_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    y_noise = np.random.normal(0, np.sqrt(noise_avg_watts), (len(signal_watts)))
    return signal + y_noise


def add_noise_to_data(data, target_snr_db, noise_chance):
    for sample in range(data.shape[0]):
        for channel in range(data.shape[2]):
            if random.random() < noise_chance:
                data[sample, :, channel] = add_noise_to_signal(data[sample, :, channel], target_snr_db)
            else:
                data[sample, :, channel] = data[sample, :, channel]
    return data


########################################################################################################################


def transform2freq(x, idx):
    if x is None:
        return None
    out_x = np.zeros((x.shape[0], 128, 16, 1))

    for i in range(x.shape[0]):
        if np.sum(x[i, :, idx]):
            f, t, Zxx = signal.stft(x[i, :, idx], fs=64, padded=False)
            Zxx = np.squeeze(Zxx)
            Zxx = np.abs(Zxx)[:128, :16]
            out_x[i, :, :, 0] = ((Zxx - np.min(Zxx)) / (np.max(Zxx) - np.min(Zxx)))
        else:
            out_x[i, :, :, 0] = np.zeros((128, 16))
    return np.nan_to_num(out_x)


def resize(x, idx):
    if x is None:
        return None

    out_x = np.zeros((x.shape[0], 128, 16, len(idx)))
    for n, id in enumerate(idx):
        for i in range(x.shape[0]):
            out_x[i, :, :, n] = np.reshape(np.pad(x[i, :, id], [(0, 128)]), out_x.shape[1:3])
    return np.nan_to_num(out_x)


def generate_modalities(m_names):
    m_list = []
    modals = {
        "eog": Modality("eog", [0], (128, 16, 1), (16, 16, 1), need_freq=True),
        "eeg": Modality("eeg", [1], (128, 16, 1), (16, 16, 1), need_freq=True),
        "resp": Modality("resp", [3], (128, 16, 1), (16, 16, 1), need_freq=True),

        "spo2": Modality("spo2", [4], (128, 16, 1), (16, 16, 1), need_reshape=True),
        "co2": Modality("co2", [5], (128, 16, 1), (16, 16, 1), need_reshape=True),
        "ecg": Modality("ecg", [7, 8], (128, 16, 2), (16, 16, 1), need_reshape=True),

    }
    for m in m_names:
        m_list.append(modals[m])
    return m_list
