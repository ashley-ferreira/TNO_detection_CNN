from ossos import storage
from ossos import astrom
from ossos.downloads.cutouts import downloader
import sys 

def pull_cutout(full_filename='vos:OSSOS/measure3/2015AP/15AP+2-1/15AP+2-1_p36.measure3.cands.astrom', 
        filename='15AP+2-1_p36.measure3.cands.astrom', real_exists=0):
    '''
    
    '''
    storage.DBIMAGES = 'vos:OSSOS/dbimages'
    storage.MEASURE3 = 'vos:OSSOS/measure3'

    parser = astrom.AstromParser()

    dlm = downloader.ImageCutoutDownloader(slice_rows=200, slice_cols=200)

    sources = parser.parse(full_filename)

    for source in sources.get_sources()[1:2]:
        for i,reading in enumerate(source.get_readings()):
            print(reading)
            cutout = dlm.download_cutout(reading, needs_apcor=True)
            print(cutout.hdulist)
            cutout.hdulist.writeto('/arc/projects/uvickbos/ML-MOD/'+ filename + \
                + str(i) + '_label=' + str(real_exists) + '.fits', overwrite=True)
            mjd_obs = float(cutout.fits_header.get('MJD-OBS'))
            exptime = float(cutout.fits_header.get('EXPTIME'))
            print(f'Exposure taken at: {mjd_obs} with exposure time {exptime}')


    return 1

pull_cutout() # temporary
# right cutout size
# loop through *.cands.astrom in block
# save filename 

# move to CNN code
# search for .reals, assign label1, others 0 (can put in fname?)
# remove background for indiv img, then regularize 