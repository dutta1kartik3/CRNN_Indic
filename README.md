To build the project, first install the latest versions of Torch7 and LMDBOn Ubuntu, lmdb can be installed by apt-get install liblmdb-dev.
The following project also needs to be installed: https://github.com/qassemoquab/stnbhwd


Instructions for Testing
1. Create a text file containig path to list of images for which textual recognition is required
2. Run th src/test.lua <img-file-list> (If you want to specifiy parent path, you can set the addn variable)
3. If you are unable to run the code, you have to install fblualib and then go to src/ and execute sh build_cpp.sh to build the C++ code. If successful, a file named libcrnn.so should be produced in the src/ directory.
4. If you wish to do lexicon based decoding, you need to run create_lexicon_dataset.py to create the lexion file and that needs to be present in the src/ folder (You can change the name of the lexicon file in the test.lua script)
5. For viewing the output in unicode you can use the converter.py script under tools/ (python3 tools/converter.py). It needs location of the lookup file (<lookup-file>) and the file where the annotations outputted by the test.lua script are stored. It prints the unicode characters on screen.

Instructions for Training -- Following the stucture in IIIT-HW-Dev dataset, one would have train/test/val files with image-path and label pairs
1. Check using tool/maxLengthWords.py if any word exceeds the max. word length the architecture can decode, if nothing is printed on the terminal, there's no issue.
2. From the train/test/val files, seperate out all the labels and create a file containing all the unique labels. Run tools/create_lookup_file.py and create the lookup file ( say <lookup-file> ).
3. Depending on whether you want to create a dataset with/without augmented images, run tools/create_dataset or tools/create_dataset_aug respectively. Either class requires outputPath to where the lmdb file is to be written, imagePathfileLilst which is just the train/test/val .txt file from the dataset, parentDirofImages, which mentions the directory where the HindiSeg folder is present and the lookupFile which was created in step 2.
4. Go to the models/ folder and either create new folder or copy an existing folder. The various properties present that need to be changed in config.lua file are mentioend below.
5. Run the training code as: src/th main_train.lua <Path to config.lua file> <Path to snapshot / saved model> ( <Path to snapshot / saved model> is optional.)


PS: The code is based upon the work done here: https://github.com/bgshih/crnn (Search in that repos forum in case of doubts)
IIIT-HW-Dev dataset link: http://preon.iiit.ac.in/~kartik/IIIT-HW-Dev.zip


A basisc description of the various files and folders
Model Folder -->
config.lua-
Contains all the various arch. defn's
like number of classes and the layers.
snapshotInterval: After how many iter.
a model is saved to disk
nClasses: no of unique char's in the dataset.
maxT: See from the model defn
Model cannot recognize words having more characters than
maxT.
savePath: Path where the various snapshots are saved
testInterval: After how many iterations you get test
results
trainSetPath and valSetPath need to be set where 
the training and validation lmdb files are present

In src folder:
DatasetLmdb.lua --> only need to change local imgW, imgH
everywhere
test.lua --> use for predicting results on test set, change
snapshot path and wether to use lexicon free or lexicon based
decoding.
inference.lua --> lexicon size is fixed to 30 characters.
utilities.lua -->
    ascii2label:
    ranking always starts from 1
    ends at nClasses in config.lua
    label2ascii:
    opposite of ascii2label function
    and be careful of label ==0 case
    loadAndResizeImage:
    Needs to be the set to the same values in DatasetLmdb.lua 
