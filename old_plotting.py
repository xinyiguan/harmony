import loader



if __name__ == '__main__':
    metacorpora_path = 'romantic_piano_corpus/'
    metacorpora = loader.MetaCorporaInfo(metacorpora_path)
    metadata = metacorpora.metadata
    print(metadata)
