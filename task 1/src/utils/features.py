from typing import Any, Dict, List

def word2features(sent: List[str], i: int) -> Dict[str, Any]:
    """Extracts features for a single word in a tokenized sentence."""
    word = sent[i]
    
    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
    }
    
    # Features for previous word
    if i > 0:
        word1 = sent[i - 1]
        features.update({
            "-1:word.lower()": word1.lower(),
            "-1:word.istitle()": word1.istitle(),
            "-1:word.isupper()": word1.isupper(),
        })
    else:
        features["BOS"] = True  # Beginning of sentence

    # Features for next word
    if i < len(sent) - 1:
        word1 = sent[i + 1]
        features.update({
            "+1:word.lower()": word1.lower(),
            "+1:word.istitle()": word1.istitle(),
            "+1:word.isupper()": word1.isupper(),
        })
    else:
        features["EOS"] = True  # End of sentence

    return features

def sent2features(sent: List[str]) -> List[Dict[str, Any]]:
    """Converts a tokenized sentence into a list of feature dictionaries."""
    return [word2features(sent, i) for i in range(len(sent))]