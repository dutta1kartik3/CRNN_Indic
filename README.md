To build the project, first install the latest versions of Torch7 and LMDBOn Ubuntu, lmdb can be installed by apt-get install liblmdb-dev.
The following project also needs to be installed: https://github.com/qassemoquab/stnbhwd


Instructions for Testing
1. Create a text file containig path to list of images for which textual recognition is required
2. Run th src/test.lua <img-file-list> (If you want to specifiy parent path, you can set the addn variable)
3. If you are unable to run the code, you have to install fblualib and then go to src/ and execute sh build_cpp.sh to build the C++ code. If successful, a file named libcrnn.so should be produced in the src/ directory.
4. If you wish to do lexicon based decoding, you need to run create_lexicon_dataset.py to create the lexion file and that needs to be present in the src/ folder (You can change the name of the lexicon file in the test.lua script)
5. For viewing the output in unicode you can use the converter.py script under tools/ (python3 tools/converter.py). It needs location of the lookup file and the file where the annotations outputted by the test.lua script are stored. It prints the unicode characters on screen.

Instructions for Training -- Following the stucture in IIIT-HW-Dev dataset, one would have train/test/val files with image-path and label pairs
1. Check using tool/maxLengthWords.py if any word exceeds the max. word length the architecture can decode, if nothing is printed on the terminal, there's no issue.
2. 





PS: The code is based upon the work done here: https://github.com/bgshih/crnn
IIIT-HW-Dev dataset link: http://preon.iiit.ac.in/~kartik/IIIT-HW-Dev.zip

