# RUN IN NW TERMINAL

import sys 
import os
import glob
from pull_cutout_func import pull_cutout

num_files = 1000

path = 'vos:OSSOS/measure3/2015A-P/15AP+2-1/'

contents = os.popen('vls vos:OSSOS/measure3/2015A-P/15AP+2-1/').read().split('\n')
print(contents)
print(len(contents))

count = 0
for file in contents: 
    print(file)
    count +=1 
    if count > num_files:
        break

    elif file.endswith(".cands.astrom"):
        #print(file)
        file_path = os.path.join(path, file)
        #print(file_path)
        real_file = file.replace('.cands.astrom', '.reals.astrom')
        print(file)
        print(real_file)
        if real_file in contents:
            real_exists = 1
            print('reals file found')
        else:
            real_exists = 0
        pull_cutout(file_path, file, real_exists)