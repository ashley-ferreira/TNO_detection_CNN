# RUN IN NW TERMINAL

import sys 
import os
import glob
from pull_cutout import pull_cutout



def pull_dir_loop(cutout_dir = '2015A-P/', num_cutouts = 10000000):
    year = cutout_dir[2:4]
    print(year)
    sem = cutout_dir[4:5]
    print(sem)
    block = cutout_dir[6:7]
    print(block)

    
    og_name = cutout_dir.replace('_automatic/', '/')
    vos_path = 'vos:OSSOS/measure3/' + og_name + '/'
    local_path = '/arc/projects/uvickbos/ML-MOD/OSSOS_datapull/' + cutout_dir
   
    main_dirs = [directory for directory in os.listdir(local_path) if os.path.isdir(local_path+directory)]
    print(main_dirs)

    for d in main_dirs:
        try: 
            vos_path_d = vos_path + d
            local_path_d = local_path +  d

            str1 = 'ls ' + local_path_d
            dirs = os.popen(str1).read().split('\n')

            count = 0
            for dir in dirs:
                file = dir

                file_cut = file.rsplit('/', 1)[-1]
                print('checking file',file_cut)
                
                if count > num_cutouts//2:
                    print('MAX CUTOUTS SAVED, exciting program...')
                    break

                elif file.endswith(".cands.astrom") and file_cut[9] == 'p': 
                    file_path = os.path.join(str(vos_path_d)  + '/' + str(dir)) 
                    real_file = file.replace('.cands.astrom', '.reals.astrom') 
                    real_file_cut = real_file.rsplit('/', 1)[-1]
                    #print('searching for .reals.astrom, found')
                    print(real_file)
                    real_exists = 0
                        
                    filesize = os.path.getsize(local_path_d + '/' + real_file_cut)
                    print(filesize)

                    if filesize != 0:
                        count +=1
                        print('real_exists = 1')
                        real_exists = 1
                       
                    pull_cutout(str(file_path), str(file), real_exists)


        except Exception as e:
            print('ERROR with dir', d)
            print(e)

# list_x = ['2015B-D_automatic/']
# list_x = ['2013A-E_automatic/', '2013B-L_redo_automatic/', '2014B-H_fix_automatic/', '2015B-H_automatic/', '2015B-S_automatic/', '2015B-D_automatic/']
list_x = ['2015A-M_automatic/', '2013A-O_automatic/', '2015A-P_automatic/'] #'2015B-D_automatic/'
for x in list_x:
    pull_dir_loop(x)