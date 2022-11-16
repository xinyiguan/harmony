# Created by Xinyi Guan in 2022.
import os
import shutil


def delete_files():

    folder_path = 'dcml_corpora/'

    for corpus in os.listdir(folder_path):
        corpus_path = folder_path+corpus+'/'
        print(corpus_path)
        try:
            # remove .github/
            shutil.rmtree(corpus_path+'.github/')
        except:
            print('No such file/folder.')

        # remove .git/
        try:
            shutil.rmtree(corpus_path+'.git/')
        except:
            print('No such file/folder.')

        # remove .git
        try:
            shutil.rmtree(corpus_path+'.git')
        except:
            print('No such file/folder.')

        # remove .gitignore
        try:
            shutil.rmtree(corpus_path + '.gitignore')
        except:
            print('No such file/folder.')



if __name__=='__main__':
    delete_files()