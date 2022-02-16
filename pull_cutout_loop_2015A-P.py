# RUN IN NW TERMINAL

import sys 
import os
import glob
from pull_cutout_func import pull_cutout

num_files = 100000

vos_path = 'vos:OSSOS/measure3/2015A-P/'

local_path = '/arc/projects/uvickbos/ML-MOD/OSSOS_datapull/2015-AP/'

str1 = 'ls ' + local_path
AP_15_dirs = os.popen(str1).read().split('\n')

for dir in AP_15_dirs:
    # loop through all dirs in 2015A-P
    print(dir)
    str2 = 'vls vos:OSSOS/measure3/2015A-P/'+ dir
    contents = os.popen(str2).read().split('\n')

    count = 0
    for file in contents: 
        print('checking file',file)
        count +=1 
        if count > num_files:
            break

        elif file.endswith(".cands.astrom"):
            print('this is a .cans.astrom file')
            file_path = os.path.join(vos_path  + '/' + dir + '/', file)
            #print(file_path)
            real_file = file.replace('.cands.astrom', '.reals.astrom')
            print('searching for .reals.astrom, found')
            print(real_file)
            real_exists = 0
            if real_file in contents:
                print('reals file found')
                filesize = os.path.getsize(local_path + '/' + dir + '/' + real_file)
                #filesize = os.popen('stat '+path+real_file)
                #print(filesize)
                if filesize != 0:
                    print('real_exists = 1')
                    real_exists = 1
            pull_cutout(local_path + '/' + dir + '/', file, real_exists)