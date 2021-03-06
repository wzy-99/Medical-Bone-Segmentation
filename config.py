
USE_ONE_LABEL = False

if USE_ONE_LABEL:

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

else:

    CLASS_NUMBER = 1 + 1

    WEIGHT = [
        1.0,   # null
        10000.0,  # k
    ]

    LABEL2ID = {
        'null': 0,
        'K': 1,
        '1': 1,
        '2': 1,
        '3': 1,
        '4': 1,
        '5': 1,
        '6': 1,
    }

    ID2LABEL = {
        0: 'null',
        1: 'bone',
    }



INPUT_SIZE = 512
LABLE_SIZE = 512

W_MIN_RATIO = 0.4
W_MAX_RATIO = 0.6
H_MIN_RATIO = 0.4
H_MAX_RATIO = 0.6

MARGIN = 100