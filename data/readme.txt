------------------------------------------
#### Directory for store all datasets ####
------------------------------------------

Here, there are three directories, all of them contain the same number of datasets, but they are in different stages.

____________________________
Inside 'downloaded_datasets', you will find all the datasets as they were downloaded, without unzipping.
    
    There is a folder for each dataset. All these directories contain a text file with the used URLs and some of them (eyepacs and aptos_2019) have also a Python
    script used to download all their files.

    Every downloaded file is stored inside 'downloads' directory.

    It is important to note that original eyepacs partitioned zip files must be fixed using a00_fix_eyepacs_zip_files.sh script in Retinopathy directory.

    All these directories have been createda and filled manually.

__________________________
Inside 'original_datasets', you will find all the unzipped datasets, consisting of all the images and all the labels files (these files contains the needed 
information about the images: DR level, DME presence and gradability).

    There is a directory for each dataset. Inside each folder, there is an 'images' directory where all the original or pre-processed images are stored (eyepacs 
    dataset have its images distributed in 'images/train' and 'images/test').

    All the labels files are also stored in every dataset folder. And, in addition to that, there is a CSV file, named as its dataset, which contains all the 
    relevant information of the current dataset. This CSV file was created by the current dataset's a01_...py script, located in Retinopathy directory.

    This CSV's structure can be accessed in 'readme.txt' file in Retinopathy directory.

___________________________
Inside 'processed_datasets', you will find every dataset once they have been processed by the python script a02_redistribute_datasets.py in Retinopathy directory.

    There is a folder for each directory, inside this folder, there can be:
    - A directory for each DR level present in the dataset. For example: eyepacs has 5 folders (0 to 4), and messidor_2_abramoff has only 2 (0 and 1).
    - A directory for all 'ungradable' images. In case of not having gradability information, this folder will not be created and a text file named
      'This dataset does not have gradability labels.txt' will replace it.
    - A directory for those images whose FOV could not be detected.
    - A CSV file which contains the main information about the images. This CSV file was created by a02_redistribute_datasets.py script. Its structure can be accessed
      in 'readme.txt' file in Retinopathy directory.
    
    There is also a '0_csvs/' folder, created by that script, which contains a copy of all datasets' CSV files
