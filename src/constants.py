"""Defines constants used throughout the model building pipeline, from preprocessing to training and evaluation."""

# Data preprocessing constants

SUBJECT_ID = "sub1"

if SUBJECT_ID == "sub1":
    CHANNELS_NUM = 62
elif SUBJECT_ID == "sub2":
    CHANNELS_NUM = 48
elif SUBJECT_ID == "sub3":
    CHANNELS_NUM = 64

"""Number of channels in ECoG data."""
WAVELET_NUM = 40
"""Number of wavelets in the frequency range."""
SAMPLE_RATE = 100
"""Final sampling rate for finger flexion and ECoG data."""
TIME_DELAY_SEC = 0.2
"""Time delay hyperparameter."""


# Data saving constants
ECOG_TRAIN_FILEPATH = "X_spectrogram_cropped.npy"
"""Filepath to the preprocessed ECoG training data."""
ECOG_VAL_FILEPATH = "X_spectrogram_cropped_val.npy"
"""Filepath to the preprocessed ECoG validation data."""
FLEXION_TRAIN_FILEPATH = "finger_flex_cropped.npy"
"""Filepath to the preprocessed finger flexion training data."""
FLEXION_VAL_FILEPATH = "finger_flex_cropped_val.npy"
"""Filepath to the preprocessed finger flexion validation data."""


# Model evaluation constants
FINGER_LABELS = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']


if __name__ == '__main__':
    pass
