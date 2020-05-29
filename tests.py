import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ismrmrd
import ismrmrd.xsd
from tqdm import tqdm
import pickle


file_list = sorted([f for f in os.listdir('/media/george/storge/DeepMRI_data/') if f.endswith('h5')])

for f in file_list:
    filename = os.path.join('/media/george/storge/DeepMRI_data/', f)
    if not os.path.isfile(filename):
        print("%s is not a valid file" % filename)
        raise SystemExit
    dset = ismrmrd.Dataset(filename, 'dataset', create_if_needed=False)

    header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
    enc = header.encoding[0]

    # Matrix size
    eNx = enc.encodedSpace.matrixSize.x
    eNy = enc.encodedSpace.matrixSize.y
    eNz = enc.encodedSpace.matrixSize.z

    # Field of View
    eFOVx = enc.encodedSpace.fieldOfView_mm.x
    eFOVy = enc.encodedSpace.fieldOfView_mm.y
    eFOVz = enc.encodedSpace.fieldOfView_mm.z


    # Number of Slices, Reps, Contrasts, etc.
    ncoils = header.acquisitionSystemInformation.receiverChannels
    if enc.encodingLimits.slice != None:
        nslices = enc.encodingLimits.slice.maximum + 1
    else:
        nslices = 1

    if enc.encodingLimits.repetition != None:
        nreps = enc.encodingLimits.repetition.maximum + 1
    else:
        nreps = 1

    if enc.encodingLimits.contrast != None:
        ncontrasts = enc.encodingLimits.contrast.maximum + 1
    else:
        ncontrasts = 1

    # Initialiaze a storage array
    all_data = np.zeros((nreps, ncontrasts, nslices, ncoils, eNz, eNy, eNx), dtype=np.complex64)


    # Loop through the rest of the acquisitions and stuff
    for acqnum in tqdm(range(0, dset.number_of_acquisitions())):
        acq = dset.read_acquisition(acqnum)
        rep = acq.idx.repetition
        contrast = acq.idx.contrast
        slice = acq.idx.slice
        y = acq.idx.kspace_encode_step_1
        z = acq.idx.kspace_encode_step_2
        all_data[rep, contrast, slice, :, z, y, :] = acq.data

    with open(f'{filename.split(".")[0]}.pickle', 'wb') as handle:
        pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print(213)
