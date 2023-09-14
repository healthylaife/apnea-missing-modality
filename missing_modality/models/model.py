import keras
from keras import layers
import tensorflow as tf
from missing_modality.models.modality import get_inps, get_decs, get_clss, get_encs, get_q_s, get_f_enc_flats, get_f_a_s, \
    get_f_encs, generate_modalities
from missing_modality.models.model_2d import create_decoder_2d, create_encoder_2d


def create_fusion_network(m_list, HIDDEN_STATE_DIM=8):
    for m in m_list:
        m.f_enc = keras.Input(m.z_dim, name=m.name + '_f_z_inp')
        m.f_enc_flat = layers.Flatten(name=m.name + '_f_enc_flat')(m.f_enc)

        m.f_q = keras.Input(m.inp_dim, name=m.name + '_f_q_inp')
        m.f_q_c = layers.Conv1D(1, 128, strides=8, padding='same', name=m.name + '_f_q_conv')(m.f_q)
        m.f_q_f = layers.Flatten(name=m.name + '_f_q_f')(m.f_q_c)

        m.f_hd = layers.Dense(HIDDEN_STATE_DIM, activation=tf.nn.tanh, name=m.name + '_f_h')(m.f_enc_flat)
        m.f_h = layers.Dropout(rate=0.25)(m.f_hd)

    First = True
    z_stack = tf.concat(get_f_enc_flats(m_list), 1, name='z_stack')
    for m in m_list:
        m.f_zd = layers.Dense(HIDDEN_STATE_DIM, activation='sigmoid', name=m.name + "_z")(z_stack)
        m.f_z = layers.Dropout(rate=0.25)(m.f_zd)
        m.f_zh = layers.Multiply(name=m.name + "_zh")([m.f_z, m.f_h])

    for m in m_list:
        if First:
            h = m.f_zh
            First = False
        else:
            h = layers.Add(name=m.name + "_add")([h, m.f_zh])
    h_flat = layers.Flatten()(h)
    label = layers.Dense(1, activation='sigmoid')(h_flat)
    return keras.Model(get_f_encs(m_list) + get_f_a_s(m_list), label, name='fusion')


def create_classifier(modality_str, inp_dim):
    input = keras.Input(inp_dim)
    n = modality_str + '_cls'
    x = layers.Flatten()(input)
    x = layers.Dense(64, activation='relu', name=n + '_l1')(x)
    x = layers.Dense(16, activation='relu', name=n + '_l2')(x)
    label = layers.Dense(1, activation='sigmoid')(x)

    return keras.Model(input, label, name=n)


def create_unimodal_model(m_list):
    for m in m_list:
        m.inp = keras.Input(m.inp_dim, name=m.name + '_inp')
        m.enc = create_encoder_2d(m.name, m.inp_dim)(m.inp)
        m.dec = create_decoder_2d(m.name, m.z_dim, output_shape=m.inp_dim)(m.enc)
        m.cls = create_classifier(m.name, m.z_dim)(m.enc)

    return keras.Model(get_inps(m_list), get_decs(m_list) + get_clss(m_list))


def create_multimodal_model(m_list):
    for m in m_list:
        m.inp = keras.Input(m.inp_dim, name=m.name + '_inp')
        m.enc = create_encoder_2d(m.name, m.inp_dim)(m.inp)
        m.dec = create_decoder_2d(m.name, m.z_dim, output_shape=m.inp_dim)(m.enc)
        m.q = layers.Subtract()([m.inp, m.dec])

    ### FUSION NETWORK ###
    label = create_fusion_network(m_list)(get_encs(m_list) + get_q_s(m_list))
    return keras.Model(get_inps(m_list), label, name='multimodal_model')


if __name__ == "__main__":
    MODALS = ["eog", "eeg", "resp", "spo2", "ecg", "co2"]

    m_list = generate_modalities(MODALS)
    model = create_multimodal_model(m_list)
    model.summary()
    keras.utils.plot_model(model, 'models.png', show_shapes=True)

