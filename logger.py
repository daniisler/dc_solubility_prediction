import logging
import os

PROJECT_ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(PROJECT_ROOT, 'input_data')
input_file = os.path.join(DATA_DIR, 'BigSolDB.csv')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(PROJECT_ROOT, 'logs', 'logging.log'))
os.makedirs(os.path.join(PROJECT_ROOT, 'logs'), exist_ok=True)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)