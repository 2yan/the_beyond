import numpy as np


class WordsWorth():
    text = None
    one_hot = None
    char_to_id = None
    char_to_one_hot = None
    id_to_char = None
    id_to_one_hot = None


    def __init__(s, text):
        'Generates a new WordsWorth Object'
        s.text = text
        s.id_to_one_hot = {}
        s.char_to_one_hot = {}

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

    def generate_seed(s, seed_text, letters):

        final = []

        if len(seed_text) > letters:
            seed_text = seed_text[-letters]
        if len(seed_text) < letters:
            difference = letters - len(seed_text)
            for num in range(0, difference):
                final.append(np.zeros(len(s.char_to_one_hot)))

        for letter in seed_text:
            final.append(s.char_to_one_hot[letter])
        final = [final]

        return np.array(final)

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

