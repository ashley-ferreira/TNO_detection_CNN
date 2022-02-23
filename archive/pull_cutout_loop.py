# RUN IN NW TERMINAL

import sys 
import os
import glob
from pull_cutout_func import pull_cutout

num_files = 10000000

AP_dirs = ['15AP+0+1/','15AP+0-1/','15AP+0-2/','15AP+1+0/','15AP+1+1/','15AP+1-1/','15AP+1-2/','15AP+2+0/','15AP+2+1/','15AP+2-1/','15AP+2-2/','15AP-1+0/','15AP-1+1/','15AP-1-1/','15AP-1-2/','15AP-2+0/','15AP-2+1/','15AP-2-1/','15AP-2-/','15AP-2-2/']

for ap in AP_dirs:
    try: 
        vos_path = 'vos:OSSOS/measure3/2015A-P/' + ap
        local_path = '/arc/projects/uvickbos/ML-MOD/OSSOS_datapull/2015A-P_automatic/' + ap

        # loop through all dirs in 2015A-P later
        vls_str = 'vls vos:OSSOS/measure3/2015A-P/' +  ap #15AP+0+0/
        contents = os.popen(vls_str).read().split('\n')

        count = 0
        for file in contents: 
            print('checking file',file)
            count +=1 
            if count > num_files:
                break

            elif file[:2] == 'fk': # TAKING THEM OUT FOR NOW, move into fo loop if not
                print('fk file, excluding for now')

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

    except Exception as e:
        print('ERROR with dir', ap)
        print(e)