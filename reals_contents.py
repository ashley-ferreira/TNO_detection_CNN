from ossos import storage
from ossos import astrom


def pull_real_decs(full_filename='vos:OSSOS/measure3/2015A-P/15AP+0+0/15AP+0+0_p14.measure3.cands.astrom'):
    '''
    
    '''
    # use previous code to get cutout function initally setup
    storage.DBIMAGES = 'vos:OSSOS/dbimages'
    storage.MEASURE3 = 'vos:OSSOS/measure3'
    parser = astrom.AstromParser()
    real_full_filename = full_filename.replace('.cands.astrom', '.reals.astrom') 
    real_sources = parser.parse(real_full_filename)

    # make a list of decs for all real sources 
    real_cands_decs = []
    for real_source in real_sources.get_sources(): 
        #print(real_source)

        for i,real_reading in enumerate(real_source.get_readings()):
            print(real_reading.dec)
            real_cands_decs.append(real_reading.dec)

    print('num reals dec',len(real_cands_decs))

    return real_cands_decs