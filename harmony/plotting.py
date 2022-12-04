#  By Xinyi Guan on 03 December 2022.
from typing import List

from harmony.loader import MetaCorporaInfo, PieceInfo

meta_corpus_info: MetaCorporaInfo
piece_infos: List[PieceInfo]
for piece_info in piece_infos:
    year = piece_info.meta_info.composed_end
    f_name = piece_info.meta_info.piece_name
    local_key_labels = piece_info.harmony_info.get_aspect('localkey').remove_repeated_labels_occurrences()
