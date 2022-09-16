from pathlib import Path
from end2you.utils import Params

"""
Model module
"""

WAV2VEC2_BASE_PATH = "/data/eihw-gpu5/karasvin/models/pretrained/facebook/wav2vec2-base/Model"

def get_feature_dim(params:Params)-> int:
    """
    Helper to calculate the number of input features from the feature extractor
    Currently only for wav2vec2-base
    """

    if "wav2vec2" in params.feature_extractor:

        if params.features == "attention":
            return 768
        elif params.features == "cnn":
            return 512
        elif params.features == "both":
            return 768 + 512
    
    else:
        raise ValueError



