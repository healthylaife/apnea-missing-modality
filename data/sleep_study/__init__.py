from . import info
from . import data
from . import dataset


data_dir = r'D:\nchsdb'


def __init__():
    data.study_list = data.init_study_list()
    age_fn = data.init_age_file()
    info.SLEEP_STUDY = info.load_health_info(info.SLEEP_STUDY, False)
