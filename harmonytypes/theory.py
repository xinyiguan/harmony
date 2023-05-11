import re

import numpy as np
from pitchtypes import asic

intervals_in_key_dict = {'major': asic(things=np.array(['P1', 'M2', 'M3', 'P4', 'P5', 'M6', 'M7'])),
                         'natural_minor': asic(things=np.array(['P1', 'M2', 'm3', 'P4', 'P5', 'm6', 'm7'])),
                         'harmonic_minor': asic(things=np.array(['P1', 'M2', 'm3', 'P4', 'P5', 'm6', 'M7']))}

chordtype_intervalclass_dict = {'M': asic(['M3', 'm3']),
                                'm': asic(['m3', 'M3']),
                                'o': asic(['m3', 'm3']),
                                '+': asic(['M3', 'M3']),
                                'o7': asic(['m3', 'm3', 'm3']),
                                '%7': asic(['m3', 'm3', 'M3']),
                                'mm7': asic(['m3', 'M3', 'm3']),
                                'mM7': asic(['m3', 'M3', 'M3']),
                                'Mm7': asic(['M3', 'm3', 'm3']),
                                'MM7': asic(['M3', 'm3', 'M3']),
                                '+7': asic(['M3', 'M3', 'd3']),
                                '+M7': asic(['M3', 'M3', 'm3']),
                                'Ger': 'Ger',
                                'Fr': 'Fr',
                                'It': 'It'}

DCML_numeral_regex = re.compile("^(?P<numeral>(b*|#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))"
                                "(?P<form>(%|o|\+|M|\+M))?"
                                "(?P<figbass>(7|65|43|42|2|64|6))?"
                                "(\((?P<changes>((\+|-|\^|v)?(b*|#*)\d)+)\))?"
                                "(/(?P<relativeroot>((b*|#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)*))?$")

augmented_6th_chords = {'It': 'viio6(b3)/V',
                        'Ger': 'viio65(b3)/V',
                        'Fr': 'V43(b5)/V'}

roman_numeral_scale_degree_dict = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7,
                                   "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7}

diatonic_chords_in_major = {1: "I", 2: "ii", 3: "iii", 4: "IV", 5: "V", 6: "VI", 7: "VII"}
diatonic_chords_in_natural_minor = {1: "i", 2: "iio", 3: "III", 4: "iv", 5: "v", 6: "VI", 7: "viio"}
diatonic_chords_in_harmonic_minor = ...
diatonic_chords_in_melodic_minor = ...