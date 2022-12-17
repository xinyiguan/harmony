import unittest


class TestChordBigramsAnalysis(unittest.TestCase):
    test_seq = ['A', 'B', 'C', 'A', 'B', 'D', 'C', 'D', 'B', 'A', 'E']
    test_tuple_seq = [('A', 'B'), ('B','C'), ('C', 'A'), ('A', 'B'), ('B', 'D'),
                      ('D', 'C'), ('C', 'D'), ('D', 'B'), ('B', 'A'), ('A', 'E')]
    def test_symmetry_measure_conditional_prob_version(self):
        answer =


if __name__ == '__main__':
    unittest.main()
