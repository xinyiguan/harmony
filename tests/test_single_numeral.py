import unittest

from musana.harmony_types import SingleNumeral, Key
from pitchtypes import SpelledPitchClass


class TestSingleNumeral(unittest.TestCase):
    sn1 = SingleNumeral.parse(key_str="C", numeral_str="I+(+4)")
    sn2 = SingleNumeral.parse(key_str="C", numeral_str="V64(#6b5)")
    sn3 = SingleNumeral.parse(key_str="C", numeral_str="V7(#9)")
    sn4 = SingleNumeral.parse(key_str="C", numeral_str="#viio65(4)")
    sn5 = SingleNumeral.parse(key_str="C", numeral_str="ii%43")
    sn6 = SingleNumeral.parse(key_str="C", numeral_str="bbIII")
    sn7 = SingleNumeral.parse(key_str="C", numeral_str="IV7(+6+2)")


    def test_sn_regex(self):
        # _sn_regex = re.compile(
        #     "^(?P<modifiers>(b*)|(#*))(?P<roman_numeral>(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))
        #     (?P<form>%|o|\+|M|\+M)?(?P<figbass>7|65|43|42|2|64|6)?(?:\((?P<changes>(?:[\+-\^v]?[b#]*\d)+)\))?$")

        sn1_match = SingleNumeral._sn_regex.match("I+(+4)")
        sn2_match = SingleNumeral._sn_regex.match("V64(#6b5)")
        sn3_match = SingleNumeral._sn_regex.match("V7(#9)")
        sn4_match = SingleNumeral._sn_regex.match("#viio65(4)")
        sn5_match = SingleNumeral._sn_regex.match("ii%43")
        sn6_match = SingleNumeral._sn_regex.match("#viio2")
        sn7_match = SingleNumeral._sn_regex.match("IV7(+6+2)")
        sn8_match = SingleNumeral._sn_regex.match("ii(+4+#2)")

        self.assertEqual(sn1_match["modifiers"], "")
        self.assertEqual(sn4_match["modifiers"], "#")
        self.assertEqual(sn6_match["modifiers"], "#")

        self.assertEqual(sn1_match["roman_numeral"], "I")
        self.assertEqual(sn2_match["roman_numeral"], "V")
        self.assertEqual(sn3_match["roman_numeral"], "V")
        self.assertEqual(sn4_match["roman_numeral"], "vii")
        self.assertEqual(sn5_match["roman_numeral"], "ii")
        self.assertEqual(sn6_match["roman_numeral"], "vii")

        self.assertEqual(sn1_match["form"], "+")
        self.assertEqual(sn2_match["form"], None)
        self.assertEqual(sn3_match["form"], None)
        self.assertEqual(sn4_match["form"], "o")
        self.assertEqual(sn5_match["form"], "%")
        self.assertEqual(sn6_match["form"], "o")

        self.assertEqual(sn1_match["figbass"], None)
        self.assertEqual(sn2_match["figbass"], "64")
        self.assertEqual(sn3_match["figbass"], "7")
        self.assertEqual(sn4_match["figbass"], "65")
        self.assertEqual(sn5_match["figbass"], "43")
        self.assertEqual(sn6_match["figbass"], "2")

        self.assertEqual(sn1_match["changes"], "+4")
        self.assertEqual(sn2_match["changes"], "#6b5")
        self.assertEqual(sn3_match["changes"], "#9")
        self.assertEqual(sn4_match["changes"], "4")
        self.assertEqual(sn5_match["changes"], None)
        self.assertEqual(sn6_match["changes"], None)
        self.assertEqual(sn7_match["changes"], "+6+2")


        self.assertRaises(ValueError,
                          lambda: SingleNumeral.parse(key_str="C", numeral_str="xyz"))  # not meaningful at all
        self.assertRaises(ValueError,
                          lambda: SingleNumeral.parse(key_str="C", numeral_str="ix"))  # there is no ix chord
        self.assertRaises(ValueError,
                          lambda: SingleNumeral.parse(key_str="C", numeral_str="b#V7"))  # there is no mixed accidentals
        self.assertRaises(ValueError,
                          lambda: SingleNumeral.parse(key_str="C", numeral_str="Io5"))  # there is no figbass with 5

    def test_parsing(self):
        # test key:
        for item in [self.sn1, self.sn2, self.sn3, self.sn4, self.sn5]:
            self.assertEqual(item.key, Key.parse(key_str='C'))
        # test degree:
        self.assertEqual(self.sn1.degree, 1)
        self.assertEqual(self.sn2.degree, 5)
        self.assertEqual(self.sn3.degree, 5)
        self.assertEqual(self.sn4.degree, 7)
        self.assertEqual(self.sn5.degree, 2)
        # test alteration:
        self.assertEqual(self.sn1.alteration, 0)
        self.assertEqual(self.sn2.alteration, 0)
        self.assertEqual(self.sn3.alteration, 0)
        self.assertEqual(self.sn4.alteration, 1)
        self.assertEqual(self.sn5.alteration, 0)
        self.assertEqual(self.sn6.alteration, -2)
        # test quality:
        self.assertEqual(self.sn1.quality, "M")
        self.assertEqual(self.sn2.quality, "M")
        self.assertEqual(self.sn3.quality, "M")
        self.assertEqual(self.sn4.quality, "m")
        self.assertEqual(self.sn5.quality, "m")
        self.assertEqual(self.sn6.quality, "M")

    def test_root(self):
        self.assertEqual(self.sn1.root(), SpelledPitchClass("C"))
        self.assertEqual(self.sn2.root(), SpelledPitchClass("G"))
        self.assertEqual(self.sn3.root(), SpelledPitchClass("G"))
        self.assertEqual(self.sn4.root(), SpelledPitchClass("B#"))
        self.assertEqual(self.sn5.root(), SpelledPitchClass("D"))
        self.assertEqual(self.sn6.root(), SpelledPitchClass("Ebb"))
        self.assertEqual(self.sn7.root(), SpelledPitchClass("F"))

    def test_key_if_tonicized(self):
        self.assertEqual(self.sn1.key_if_tonicized(), Key(root=SpelledPitchClass("C"), mode=self.sn1.quality))
        self.assertEqual(self.sn2.key_if_tonicized(), Key(root=SpelledPitchClass("G"), mode=self.sn2.quality))
        self.assertEqual(self.sn3.key_if_tonicized(), Key(root=SpelledPitchClass("G"), mode=self.sn3.quality))
        self.assertEqual(self.sn4.key_if_tonicized(), Key(root=SpelledPitchClass("B#"), mode=self.sn4.quality))
        self.assertEqual(self.sn5.key_if_tonicized(), Key(root=SpelledPitchClass("D"), mode=self.sn5.quality))
        self.assertEqual(self.sn6.key_if_tonicized(), Key(root=SpelledPitchClass("Ebb"), mode=self.sn6.quality))



if __name__ == '__main__':
    unittest.main()
