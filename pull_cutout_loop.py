# RUN IN NW TERMINAL

import sys 
import os
import glob
from pull_cutout_func import pull_cutout

num_files = 10000

vos_path = 'vos:OSSOS/measure3/2015A-P/15AP+2-1/'
local_path = '/arc/projects/uvickbos/ML-MOD/OSSOS_datapull/2015A-P/15AP+2-1/'

# loop through all dirs in 2015A-P
contents = os.popen('vls vos:OSSOS/measure3/2015A-P/15AP+2-1/').read().split('\n')

count = 0
for file in contents: 
    print('checking file',file)
    count +=1 
    if count > num_files:
        break

    elif file.endswith(".cands.astrom"):
        print('this is a .cans.astrom file')
        file_path = os.path.join(vos_path, file)
        #print(file_path)
        real_file = file.replace('.cands.astrom', '.reals.astrom')
        print('searching for .reals.astrom, found')
        print(real_file)
        real_exists = 0
        if real_file in contents:
            print('reals file found')
            filesize = os.path.getsize(local_path + real_file)
            #filesize = os.popen('stat '+path+real_file)
            #print(filesize)
            if filesize != 0:
                print('real_exists = 1')
            real_exists = 1
        pull_cutout(file_path, file, real_exists)