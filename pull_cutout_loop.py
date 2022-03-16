# RUN IN NH TERMINAL

import os
from pull_cutout_together import pull_cutout
import numpy as np 


def pull_dir_loop(cutout_dir = '2015A-P/', num_cutouts = 10000):
    #year = cutout_dir[2:4]
    #print(year)
    #sem = cutout_dir[4:5]
    #print(sem)
    #block = cutout_dir[6:7]
    #print(block)

    # find what name will be in VOS
    og_name = cutout_dir.replace('_automatic/', '/')
    vos_path = 'vos:OSSOS/measure3/' + og_name + '/'

    # define local path
    local_path = '/arc/projects/uvickbos/ML-MOD/OSSOS_datapull/' + cutout_dir
   
    # make a list of all dirs in cutout_dir, doesnt actually need to be sorted
    main_dirs = [directory for directory in os.listdir(local_path) if os.path.isdir(local_path+directory)]
    #print(main_dirs)

    # loop through all these directories
    for d in main_dirs:
        try: 
            vos_path_d = vos_path + d
            local_path_d = local_path +  d

            # loop through all files in d directory (ones that actually contain cands)
            str1 = 'ls ' + local_path_d
            files = os.popen(str1).read().split('\n') # glob way: ls *.fits,files = glob.glob(‘*.fits’)

            count = 0
            for file in files:

                # get just filename (not whole path)
                file_cut = file.rsplit('/', 1)[-1]
                print('checking file',file_cut)
                
                # stop early if max (label=1) cutouts have been saved
                if count > num_cutouts//2:
                    print('MAX CUTOUTS SAVED, exiting program...')
                    break

                # loop through all p .cands.astrom files
                # alternate way: glob.glob(‘????????p*.cands.astrom')
                elif file.endswith(".cands.astrom") and file_cut[9] == 'p': 
                    file_path = os.path.join(str(vos_path_d)  + '/' + str(file)) 

                    # define real_file name and see if there are any contents
                    real_file = file.replace('.cands.astrom', '.reals.astrom') 
                    real_file_cut = real_file.rsplit('/', 1)[-1]
                    #print('searching for .reals.astrom, found')
                    #print(real_file)
                    real_exists = 0
                    filesize = os.path.getsize(local_path_d + '/' + real_file_cut)
                    #print(filesize)
                    if filesize != 0:
                        count +=1
                        print('real_exists = 1')
                        real_exists = 1
                       
                    # call the pull_cutout function that will do the rest
                    print(file_path, file)
                    pull_cutout(str(file_path), str(file), real_exists)


        except Exception as e:
            print('ERROR with dir', d)
            print(e)

#list_x = ['2013A-E_automatic/', '2013B-L_redo_automatic/', '2014B-H_fix_automatic/', '2015B-H_automatic/', '2015B-S_automatic/', '2015B-D_automatic/','2015A-M_automatic/', '2013A-O_automatic/']
list_x = ['2015A-P_automatic/'] 
for x in list_x:
    pull_dir_loop(x)