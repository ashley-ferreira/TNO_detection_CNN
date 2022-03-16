from ossos import storage
from ossos import astrom
from ossos.downloads.cutouts import downloader

from astropy import units

import sys 
import os

from ossos.gui import logger

from reals_contents import pull_real_decs

storing_directory = '/arc/projects/uvickbos/ML-MOD/new_cutouts_mar16/'
pos_label = 1

def pull_cutout(full_filename='vos:OSSOS/measure3/2015A-P/15AP+0+0/15AP+0+0_p14.measure3.cands.astrom', 
        filename='15AP+0+0_p14.measure3.cands.astrom', real_exists=0):
    '''
    
    '''
    storage.DBIMAGES = 'vos:OSSOS/dbimages'
    storage.MEASURE3 = 'vos:OSSOS/measure3'

    parser = astrom.AstromParser()
    dlm = downloader.ImageCutoutDownloader()
    sources = parser.parse(full_filename)

    if real_exists:
        real_cands = pull_real_decs(full_filename)
    else:
        real_cands = []

        cand=0
        for source in sources.get_sources(): 
            #print(source)

            file_dir = filename.replace('.measure3.cands.astrom', '')  + '_cand=' + str(cand) + '/'
            sub_dir = storing_directory + file_dir
            os.mkdir(sub_dir)
            for i,reading in enumerate(source.get_readings()):
                #print(reading)

                if reading.dec in real_cands: # check up one earlier in source? always all 3 only
                    label=1
                    pos_label += 1
                else:
                    label=0 #do this earlier?

                cutout = dlm.download_cutout(reading, needs_apcor=True)
                sub_filename = filename.replace('.measure3.cands.astrom', '')  + '_cand=' + str(cand) + '_triplet='+ str(i) + '_label=' + str(label) + '.fits'
                cutout.hdulist.writeto(sub_dir + sub_filename, overwrite=True)
                print(sub_dir+sub_filename)
            cand+=1

        print('num pos labels',pos_label)