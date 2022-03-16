from ossos import storage
from ossos import astrom
from ossos.downloads.cutouts import downloader

from astropy import units
import os

from reals_contents import pull_real_decs

storing_directory = '/arc/projects/uvickbos/ML-MOD/new_cutouts_mar16/'

def pull_cutout(full_filename='vos:OSSOS/measure3/2015A-P/15AP+0+0/15AP+0+0_p14.measure3.cands.astrom', 
        filename='15AP+0+0_p14.measure3.cands.astrom', real_exists=0):
    '''
    
    '''
    # use previous code to get cutout function initally setup
    storage.DBIMAGES = 'vos:OSSOS/dbimages'
    storage.MEASURE3 = 'vos:OSSOS/measure3'
    parser = astrom.AstromParser()
    dlm = downloader.ImageCutoutDownloader()
    sources = parser.parse(full_filename)

    # make a list with the decs of the real cands
    if real_exists:
        real_cands = pull_real_decs(full_filename)
    else:
        real_cands = []

    cand=0
    pos_label = 0

    # loop through all candiates 
    for source in sources.get_sources(): 

        # make a directory for each candidate
        file_dir = filename.replace('.measure3.cands.astrom', '')  + '_cand=' + str(cand) + '/'
        sub_dir = storing_directory + file_dir
        os.mkdir(sub_dir)

        # loop through each of the three readings for the above candiate
        for i,reading in enumerate(source.get_readings()):

            # check if the dec of this reading is exact same as real cands dec
            # if dec matches then label=1, if not label=0
            if reading.dec in real_cands: 
                label=1
                pos_label += 1
            else:
                label=0 

            # save cutout with information stored in filename
            cutout = dlm.download_cutout(reading, needs_apcor=True)
            sub_filename = filename.replace('.measure3.cands.astrom', '')  + '_cand=' + str(cand) + '_triplet='+ str(i) + '_label=' + str(label) + '.fits'
            cutout.hdulist.writeto(sub_dir + sub_filename, overwrite=True)
            print(sub_dir+sub_filename)

        cand+=1

        # use this to double check num decs and pos labels are equal and %3=0
        print('num pos labels',pos_label)