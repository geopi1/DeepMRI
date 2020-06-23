# DeepMRI

Pytorch implementation of RAKI paper with mild changes and optimizations [1]

## Getting Started

Clone the Repo:
git clone https://github.com/geopi1/DeepMRI.git

Download the Datasets:
[link to mridata](http://mridata.org/list)
In this site select any of the available MRI scans and download to a folder

Run save_raw_data_to_pickle.py to save the .h5 files from mridata.org as a pickle with np matrices
python save_raw_data_to_pickle.py -p [path_to_wanted_folder]
or
python save_raw_data_to_pickle.py --data_path [path_to_wanted_folder]



### Prerequisites

[add a requirements.txt file]

### Datasets
[add the download links for mridata.org]

## Running the tests
main [add the flags]

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
[1] Akçakaya, Mehmet et al. “Scan-specific robust artificial-neural-networks for k-space interpolation (RAKI) reconstruction: Database-free deep learning for fast imaging.” Magnetic resonance in medicine vol. 81,1 (2019): 439-453. doi:10.1002/mrm.27420
[add citations]
