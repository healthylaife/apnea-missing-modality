import numpy as np

from missing_modality.train_test import train_test

if __name__ == "__main__":
    config = {
        "STEP": "multimodal",  # unimodal, multimodal
        "EPOCHS": 100,
        "BATCH_SIZE": 256,
        "MODALS": ["eog", "eeg", "resp", "spo2", "ecg", "co2"],
        "NOISE_RATIO": 0.00,
        "NOISE_CHANCE": 0.0,

        "MISS_RATIO": 0.00,
        "MISS_INDEX": 0,

        "FOLDS": [0,1,2,3,4],
        "PHASE": "TEST",
        ########################################################
        "transformer_layers": 5,  # best 5
        "drop_out_rate": 0.25,  # best 0.25
        "num_patches": 30,  # best 30 TBD
        "transformer_units": 32,  # best 32
        "regularization_weight": 0.001,  # best 0.001
        "num_heads": 4,
        "epochs": 100,  # best 200
        "channels": [0, 3, 5, 6, 9, 10, 4],
    }

    for data_name in [('chat', "/home/hamed/dd/chat_b_30x64_")]: # ,('chat', "/home/hamedcan/dd/chat_b_30x64__",)
        config["DATA_NAME"] = data_name[0]
        config["DATA_PATH"] = data_name[1]
        for model_name in ['cnn', 'cnn-lstm', 'Transformer', 'qaf']:
            if model_name == "qaf":
                config["STEP"] = 'multimodal'
            else:
                config["STEP"] = 'unimodal'
            config["MODEL_NAME"] = model_name
            for miss_indexes in [[0],[1], [3],[4],[5],[7,8], [0,1]]:
                config["log_name"] = config["MODEL_NAME"] + "_" + config["DATA_NAME"] + "_missmodal_" + str(miss_indexes)
                config["MISS_INDEX"] = miss_indexes

                result = train_test(config)
                out_str = data_name[0] + ", " + model_name + ", " + str(miss_indexes) + ", "
                out_str += str("%.1f, %.1f" % (np.mean(result.auroc_list), np.std(result.auroc_list))) + " \n"
                f = open("./result/missing_modal.txt", "a")
                f.write(out_str)
                f.close()

