import unittest

from musana.generics import Sequential, TransitionMatrix


class TestGenerics(unittest.TestCase):
    test_tuple_seq = Sequential.from_sequence(sequence=[('A', 'B'), ('B','C'), ('C', 'A'), ('A', 'B'), ('B', 'D'),
                      ('D', 'C'), ('C', 'D'), ('D', 'B'), ('B', 'A'), ('A', 'E')])
    def test_get_label_counts(self):
        answer = ...
        matrix = TransitionMatrix(n_grams=self.test_tuple_seq)
        lable_count = matrix.get_label_counts()

    # def test_sequential_get_n_grams():
    #     n: int = 2
    #     sequence = ['1', 'sdf', '333', '8', '9']
    #     # sequence = [1, [4], '333', lambda x:x+1, set()]
    #
    #     sequential = Sequential.from_sequence(sequence=sequence)
    #     n_gram = sequential.get_n_grams(n=n)
    #     expected_length = len(sequence) - n + 1
    #     assert len(n_gram) == expected_length
    #
    # def test_sequential_get_transition_matrix():
    #     n: int = 2
    #     sequence = [1, 2, 3, 4, 1, 2, 3, 4, 6, 5, 6, 5, 6, 5, 4, 3, 2, 1]
    #     sequence = list(map(str, sequence))
    #     # sequence = [1, [4], '333', lambda x:x+1, set()]
    #
    #     sequential = Sequential.from_sequence(sequence=sequence)
    #     matrix = sequential.get_n_grams(3).get_transition_matrix()
    #     print(matrix)


if __name__ == '__main__':
    unittest.main()
