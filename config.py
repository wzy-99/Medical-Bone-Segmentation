CLASS_NUMBER = 7 + 1

WEIGHT = [
    1.0,   # null
    10000.0,  # k
    10000.0,  # 1
    10000.0,  # 2
    10000.0,  # 3
    10000.0,  # 4
    10000.0,  # 5
    10000.0   # 6
]

LABEL2ID = {
    'null': 0,
    'K': 1,
    '1': 2,
    '2': 3,
    '3': 4,
    '4': 5,
    '5': 6,
    '6': 7,
}

ID2LABEL = {
    0: 'null',
    1: 'k',
    2: '1',
    3: '2',
    4: '3',
    5: '4',
    6: '5',
    7: '6',
}

INPUT_SIZE = 512
LABLE_SIZE = 512

MIN_CROP_RATIO = 0.6
MAX_CROP_RATIO = 1.0