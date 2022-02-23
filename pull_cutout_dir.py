# RUN IN NW TERMINAL

import sys 
import os
import glob
from pull_cutout_func import pull_cutout

AP15_dirs = ['15AP+0+1/','15AP+0-1/','15AP+0-2/','15AP+1+0/','15AP+1+1/','15AP+1-1/','15AP+1-2/','15AP+2+0/', \
        '15AP+2+1/','15AP+2-1/','15AP+2-2/','15AP-1+0/','15AP-1+1/','15AP-1-1/','15AP-1-2/','15AP-2+0/','15AP-2+1/', \
        '15AP-2-1/','15AP-2-/','15AP-2-2/']

def pull_dir_loop(cutout_dir = '2015A-P/', num_cutouts = 100000, main_dir=AP15_dirs):

    for d in main_dir:
        try: 
            vos_path = 'vos:OSSOS/measure3/' + cutout_dir + d
            local_path = '/arc/projects/uvickbos/ML-MOD/OSSOS_datapull/' + cutout_dir + d

            str1 = 'ls ' + local_path
            dirs = os.popen(str1).read().split('\n')

            for dir in dirs:
                print(dir)
                str2 = 'vls vos:OSSOS/measure3/' + cutout_dir + dir
                contents = os.popen(str2).read().split('\n')

                count = 0
                for file in contents: 
                    print('checking file',file)
                    
                    if count > num_cutouts//2:
                        break

                    elif file[:2] == 'fk': 
                        print('fk file, excluding for now')

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
                                count +=1
                                print('real_exists = 1')
                                real_exists = 1
                        pull_cutout(local_path + '/' + dir + '/', file, real_exists)

        except Exception as e:
            print('ERROR with dir', d)
            print(e)