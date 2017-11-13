import numpy as np


class WordsWorth():
    text = None
    one_hot = None
    char_to_id = {}
    char_to_one_hot = {}
    id_to_char = {}
    id_to_one_hot = {}


    def __init__(s, text):
        'Generates a new WordsWorth Object'
        s.text = text
        unique = set(text)
        s.id_to_char = dict(zip(range(0, len(unique)), unique))
        s.char_to_id = dict(zip(s.id_to_char.values(), s.id_to_char.keys()))

        for key, char in s.id_to_char.items():
            one_hot = [0] * len(unique)
            one_hot[key] = 1
            s.id_to_one_hot[key] = one_hot
            s.char_to_one_hot[char] = one_hot

        s.one_hot = np.zeros((len(text), len(unique)))
        i = 0
        for char in text:
            s.one_hot[i, s.char_to_id[char]] = 1
            i = i + 1

    def get_sentences(s, letters, step = 1):

        X = []
        y = []
        for num in range(0, len(s.text) - letters, step):
            X.append(s.one_hot[num:num + letters] )
            y.append(s.one_hot[num + letters])

        return np.array(X), np.array(y)

    def __pred_to_text(s, result):
        return s.id_to_char[list(result).index(result.max())]

    def one_hot_to_text(s, result):
        if len(result.shape) ==1:
            return s.__pred_to_text( result)

        if len(result.shape) == 2:
            text = ''
            for item in result:
                text = text + s.__pred_to_text(item)
            return text

        if len(result.shape) >2:
            answer = []
            for thing in result:
                answer.append(s.one_hot_to_text(thing))
            return answer

