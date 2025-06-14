"""


  ____                                      ___ __  __    _    ____ _____ _   _
 |  _ \  ___  ___  ___ __ _ _ __ __ _  __ _|_ _|  \/  |  / \  / ___| ____| \ | |
 | | | |/ _ \/ __|/ __/ _` | '__/ _` |/ _` || || |\/| | / _ \| |  _|  _| |  \| |
 | |_| |  __/\__ \ (_| (_| | | | (_| | (_| || || |  | |/ ___ \ |_| | |___| |\  |
 |____/ \___||___/\___\__,_|_|  \__, |\__,_|___|_|  |_/_/   \_\____|_____|_| \_|
                                |___/

"""


import requests
import numpy as np
import time
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import pandas as pd
import sys
import os
import hashlib
import time
import random
import mimetypes

def download_url(url,out):
    #hashlib.sha1().update(str(time.time()).encode("utf-8"))
    try:
        bashcom = f'python -m gallery_dl --cookies "C:/Users/pumgu/OneDrive/Escritorio/UG/LIDIA_6to_Semestr/LIDIA_6to_Semestre/verano/cookies.txt"  -D "{out}" "{url}"'
        #print(f'\n{bashcom}\n')
        os.system(bashcom)
        #with open(out+f'/{hashlib.md5(os.urandom(32)).hexdigest()}'+f'{extension}', 'wb') as f:
    except Exception as e:
        print('Exception in download_url():', e)

def list_files_in_directory(directory, extension):
    try:
        # Get a list of all files and directories in the specified directory
        files = os.listdir(directory)
        # Filter out directories and keep only files with the specified extension
        file_list = [f for f in files if os.path.isfile(os.path.join(directory, f)) and f.endswith(extension)]
        return file_list

    except Exception as e:
        print(f"An error occurred: {e}")
        return []



path   = sys.argv[1]
files  = list_files_in_directory(path,'csv')

for f in files:
    auxPath = path+'/'+f.replace('.csv','')
    print(f"\nAnalizando: {auxPath}\n")
    isExist = os.path.exists(auxPath)
    if not isExist:
        os.makedirs(auxPath)
        print(f'\n{path+"/"+f}\n')
        urlsx  = pd.read_csv(path+'/'+f, header=0, encoding='latin-1')
        urls    = urlsx['postUrl'].tolist()
        for l in urls:
            time.sleep(random.randint(1,10))
            download_url(l,auxPath)


print(f"\nTERMINADO\n")
