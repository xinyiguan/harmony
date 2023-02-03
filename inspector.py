import typing
from dataclasses import dataclass

import pandas as pd

from musana import harmony_types, loader


@dataclass
class HarmonyInspector:
    globalkey: str
    localkey: str
    chord: str
    numeral: str
    form: str
    figbass: str
    changes: str
    relativeroot: str
    chord_type: str
    chord_tones:typing.List[int]
    added_tones: typing.List[int]
    root: int
    bass_note: int

    @staticmethod
    def from_directory(corpus_path: str):
        pass

    @classmethod
    def from_tsv(cls, tsv_path: str) -> typing.Self:
        df = pd.read_csv(tsv_path + 'metadata.tsv', sep='\t')
        globalkey = ...
        localkey = ...
        chord = ...
        numeral = ...
        form = ...
        figbass = ...
        changes = ...
        relativeroot = ...
        chord_type = ...
        chord_tones = ...
        added_tones = ...
        root = ...
        bass_note = ...

        instance = cls(chord=chord, globalkey=globalkey, localkey=localkey,
                       numeral=numeral, form = form, figbass=figbass, changes=changes, relativeroot=relativeroot,
                       chord_type=chord_type, chord_tones=chord_tones, added_tones=added_tones, root = root, bass_note=bass_note )
        return instance

    def mark_discrepencies(self, kw_list= typing.List[str]):
        newly_parsed = harmony_types.TonalHarmony.parse_dcml(globalkey_str=self.globalkey, localkey_numeral_str=self.localkey,
                                                             chord_str = self.chord)