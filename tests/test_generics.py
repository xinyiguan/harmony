import unittest

from harmony.generics import Sequential


class TestGenerics(unittest.TestCase):
    def test_sequential_get_n_grams():
        n: int = 2
        sequence = ['1', 'sdf', '333', '8', '9']
        # sequence = [1, [4], '333', lambda x:x+1, set()]

        sequential = Sequential.from_sequence(sequence=sequence)
        n_gram = sequential.get_n_grams(n=n)
        expected_length = len(sequence) - n + 1
        print(n_gram)
        assert len(n_gram) == expected_length

    def test_sequential_get_transition_matrix():
        n: int = 2
        sequence = [1, 2, 3, 4, 1, 2, 3, 4, 6, 5, 6, 5, 6, 5, 4, 3, 2, 1]
        sequence = list(map(str, sequence))
        # sequence = [1, [4], '333', lambda x:x+1, set()]

        sequential = Sequential.from_sequence(sequence=sequence)
        matrix = sequential.get_n_grams(3).get_transition_matrix()
        print(matrix)


if __name__ == '__main__':
    unittest.main()
