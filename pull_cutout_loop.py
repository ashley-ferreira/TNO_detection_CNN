# RUN IN NW TERMINAL

import sys 
import os
import glob

num_files = 100

# path = '/arc/home/jkavelaras/'
path = 'vos:OSSOS/measure3/2015AP/' # 15AP+2-1_p36.measure3.cands.astrom'
# vos:OSSOS/measure3/${BLOCK} - use this to run through diff blocks

real_files = glob.glob(path + "/*.reals.astrom")
print('Number of .reals:', len(real_files))
# print(real_files) 

# files = os.listdir(path)

for file in os.listdir(path):

    count +=1 
    if count > num_files:
        break

    elif file.endswith(".cands.astrom"):
        print(file)
        file_path = os.path.join(path, file)
        print(file_path)
        if file_path in real_files:
            real_exists = 1
        else:
            real_exists = 0
        pull_cutout(file_path, file, real_exists)

#.measure3.cands.astrom

