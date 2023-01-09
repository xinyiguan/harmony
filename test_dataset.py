import re

from musana import loader as loader



if __name__=='__main__':
    metacorpora_path = '/Users/xinyiguan/Codes/data/romantic_piano_corpus_updated/'

    metacorpora = loader.MetaCorporaInfo.from_directory(metacorpora_path=metacorpora_path)

    aspect = metacorpora.harmony_info.get_aspect(key='')

