"""Script used for extracting MACHO dataset features in streaming form"""

import numpy as np
import os.path
import logging
import sys

sys.path.append(os.path.abspath('../'))

from streaming_gmm import lightcurve
from streaming_gmm.features.lightcurve_features import LightCurveFeatures

PATH_MACHO = '../data/raw-macho'

LOGGING_FORMAT = '%(asctime)s|%(name)s|%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO,
                    format=LOGGING_FORMAT,
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def calculate_features_in_dir(lc_dir):
    lc_dir_path = '{}/{}'.format(PATH_MACHO, lc_dir)
    files_in_lc_dir = os.listdir(lc_dir_path)
    lc_files = filter(lambda f: f.startswith('lc_'), files_in_lc_dir)
    lc_id_to_file = {}
    for lc_file in lc_files:
        lc_id = lc_file[3:-5]
        if lc_id in lc_id_to_file:
            lc_id_to_file[lc_id].append(lc_file)
        else:
            lc_id_to_file[lc_id] = [lc_file]

    for lc_id, lc_files in lc_id_to_file.items():
        for lc_file in lc_files:
            lc_file_path = '{}/{}'.format(lc_dir_path, lc_file)
            lc_df = lightcurve.read_from_file(lc_file_path, skiprows=3)
            lc_df = lightcurve.remove_unreliable_observations(lc_df)
            lc_features = LightCurveFeatures()
            for time, mag, error in to_chunks(lc_df, chunk_size=20):
                observations = {}
                lc_features.update()
                pass


logger.info("Calculating features in %s", os.path.abspath(PATH_MACHO))
lc_dirs = filter(lambda d: not d.startswith('.') and d != 'non_variables',
                 os.listdir(PATH_MACHO))
for lc_dir in lc_dirs:
    logger.info("Calculating features in %s", lc_dir)
    calculate_features_in_dir(lc_dir)
