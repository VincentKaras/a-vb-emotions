import enum
from pathlib import Path

EMOTIONS = ["Awe", "Excitement", "Amusement", "Awkwardness", "Fear", "Horror", "Distress", "Triumph", "Sadness", "Surprise"]
CULTURES = ["China", "United States", "South Africa", "Venezuela"]
DIMENSIONS = ["Valence", "Arousal"]
VOCAL_TYPES =["Laugh", "Grunt", "Cry", "Pant", "Gasp", "Groan", "Scream", "Other"]

# cartesian prod of cultures and emotions (40 labels)
CULTURE_EMOTIONS = []
for c in CULTURES:
    for e in EMOTIONS:
        CULTURE_EMOTIONS.append("{}_{}".format(c, e))

# mapping of string types to integer class indices
MAP_VOCAL_TYPES = {t: i for i, t in enumerate(VOCAL_TYPES)}
INVERSE_MAP_VOCAL_TYPES = {i: t for i, t in enumerate(VOCAL_TYPES)}
MAP_CULTURES = {c: i for i, c in enumerate(CULTURES)}
INVERSE_MAP_CULTURES = {i: c for i, c in enumerate(CULTURES)}

# task naming convention of the organisers
MAP_OFFICIAL_TASK_NAMES = { 
    "voc_type": "type",
    "culture_emotion": "culture",
    "low": "two",
    "high": "high",
}

# permutation for classifier chain - descending performance from baseline
EMOTIONS_PERMUTED = ['Surprise', 'Amusement', 'Awe', 'Fear', 'Horror', 'Sadness', 'Distress', 'Excitement', 'Triumph', 'Awkwardness']
EMOTIONS_INDICES_PERMUTED = [9, 2, 0, 4, 5, 8, 6, 1, 7, 3]  # integer indices for rearranging label order
EMO_INDICES_RESTORED = [EMOTIONS_INDICES_PERMUTED.index(i) for i in range(len(EMOTIONS))]  # restores the orginal order

CULTURE_EMOTIONS_PERMUTED = ['United States_Amusement', 'United States_Awe', 'United States_Surprise', 'United States_Fear', 'United States_Triumph', 'United States_Sadness', 'United States_Horror', 'United States_Excitement', 'United States_Distress', 'United States_Awkwardness', 
'South Africa_Amusement', 'South Africa_Surprise', 'South Africa_Fear', 'South Africa_Excitement', 'South Africa_Sadness', 'South Africa_Horror', 'South Africa_Triumph', 'South Africa_Awe', 'South Africa_Distress', 'South Africa_Awkwardness', 
'Venezuela_Amusement', 'Venezuela_Awe', 'Venezuela_Sadness', 'Venezuela_Surprise', 'Venezuela_Horror', 'Venezuela_Fear', 'Venezuela_Triumph', 'Venezuela_Awkwardness', 'Venezuela_Distress', 'Venezuela_Excitement', 
'China_Horror', 'China_Surprise', 'China_Sadness', 'China_Distress', 'China_Fear', 'China_Excitement', 'China_Triumph', 'China_Amusement', 'China_Awkwardness', 'China_Awe']
CULTURE_EMOTIONS_INDICES_PERMUTED = [12, 10, 19, 14, 17, 18, 15, 11, 16, 13, 22, 29, 24, 21, 28, 25, 27, 20, 26, 23, 32, 30, 38, 39, 35, 34, 37, 33, 36, 31, 5, 9, 8, 6, 4, 1, 7, 2, 3, 0]
CULTURE_EMOTIONS_INDICES_RESTORED = [CULTURE_EMOTIONS_INDICES_PERMUTED.index(i) for i in range(len(CULTURE_EMOTIONS))]


CN_PERMUTED = ['China_Horror', 'China_Surprise', 'China_Sadness', 'China_Distress', 'China_Fear', 'China_Excitement', 'China_Triumph', 'China_Amusement', 'China_Awkwardness', 'China_Awe']
CN_INDICES_PERMUTED = [5, 9, 8, 6, 4, 1, 7, 2, 3, 0]
CN_INDICES_RESTORED = [CN_INDICES_PERMUTED.index(i) for i in range(len(CN_INDICES_PERMUTED))]
US_PERMUTED = ['United States_Amusement', 'United States_Awe', 'United States_Surprise', 'United States_Fear', 'United States_Triumph', 'United States_Sadness', 'United States_Horror', 'United States_Excitement', 'United States_Distress', 'United States_Awkwardness']
US_INDICES_PERMUTED = [2, 0, 9, 4, 7, 8, 5, 1, 6, 3]
US_INDICES_RESTORED = [US_INDICES_PERMUTED.index(i) for i in range(len(US_INDICES_PERMUTED))]
SA_PERMUTED = ['South Africa_Amusement', 'South Africa_Surprise', 'South Africa_Fear', 'South Africa_Excitement', 'South Africa_Sadness', 'South Africa_Horror', 'South Africa_Triumph', 'South Africa_Awe', 'South Africa_Distress', 'South Africa_Awkwardness']
SA_INDICES_PERMUTED = [2, 9, 4, 1, 8, 5, 7, 0, 6, 3]
SA_INDICES_RESTORED = [SA_INDICES_PERMUTED.index(i) for i in range(len(SA_INDICES_PERMUTED))]
VZ_PERMUTED = ['Venezuela_Amusement', 'Venezuela_Awe', 'Venezuela_Sadness', 'Venezuela_Surprise', 'Venezuela_Horror', 'Venezuela_Fear', 'Venezuela_Triumph', 'Venezuela_Awkwardness', 'Venezuela_Distress', 'Venezuela_Excitement']
VZ_INDICES_PERMUTED = [2, 0, 8, 9, 5, 4, 7, 3, 6, 1]
VZ_INDICES_RESTORED = [VZ_INDICES_PERMUTED.index(i) for i in range(len(VZ_INDICES_PERMUTED))]


DATA_DIR = Path("/data/eihw-gpu5/karasvin/databases/ExVo2022_ACII_A_VB/audio/wav")
LABEL_DIR = Path("/data/eihw-gpu5/karasvin/databases/ExVo2022_ACII_A_VB/labels")

TRAIN_FILE = LABEL_DIR / "train.csv"
VAL_FILE = LABEL_DIR / "val.csv"
TEST_FILE = LABEL_DIR / "test.csv"