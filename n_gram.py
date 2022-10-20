from dataclasses import dataclass
from typing import List, Literal, Tuple, Any

import numpy as np
import pandas as pd

import loader


def get_transitions(sequence: List[str], n=1) -> np.ndarray:
    transitions = np.array(
        [['_'.join(sequence[i - n:i]), sequence[i]] for i in range(n, len(sequence))])
    return transitions


def get_transition_matrix(transitions: np.ndarray) -> pd.DataFrame:
    contexts, targets = np.unique(transitions[:, 0]), np.unique(transitions[:, 1])
    transition_matrix = pd.DataFrame(0, columns=targets, index=contexts)
    for i, transition in enumerate(transitions):
        context, target = transition[0], transition[1]
        transition_matrix.loc[context, target] += 1
        # print(transition_matrix)
    return transition_matrix


@dataclass
class N_Gram:
    piece: loader.PieceInfo
    aspect: Literal['harmonies', 'measures', 'notes']
    key: str
    n: int

    def __post_init__(self):
        self.grams_seq = self.get_grams_seq()

    def get_grams_seq(self) -> List[str]:
        """
        produce a list of piece-wise grams as a list of tuples
        :return:
        """
        complete_seq = self.piece.get_aspect_df(aspect=self.aspect, selected_keys=[self.key]).values.flatten().tolist()
        grams_tuple = list(zip(*[complete_seq[i:] for i in range(self.n)]))
        grams = ['_'.join(grams_tuple[idx]) for idx, val in enumerate(grams_tuple)]
        return grams


if __name__ == '__main__':
    piece1 = loader.PieceInfo(parent_corpus_path='romantic_piano_corpus/debussy_suite_bergamasque/',
                              piece_name='l075-01_suite_prelude')
    piece2 = loader.PieceInfo(parent_corpus_path='romantic_piano_corpus/debussy_suite_bergamasque/',
                              piece_name='l075-02_suite_menuet')
    pieces = [piece1, piece2]
    corpera_transitions = []
    for piece in pieces:
        harmonies = piece.get_aspect_df('harmonies', selected_keys=['numeral']).values.flatten().tolist()
        transitions = get_transitions(sequence=harmonies, n=2)
        corpera_transitions.extend(transitions)
    # unigram = piece.get_aspect_df(aspect='harmonies', selected_keys=['numeral']).values.flatten().tolist()
    # n_gram = N_Gram(piece=piece, aspect='harmonies', key='numeral', n=2)
    # print(len(n_gram.get_grams_seq()))

    transition_matrix = get_transition_matrix(transitions)
    transition_probs = transition_matrix.divide(transition_matrix.sum(axis=1), axis=0)
    print(transition_matrix)
    print(transition_probs)
