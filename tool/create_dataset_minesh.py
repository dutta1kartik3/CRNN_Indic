import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)

#imagepathlist contains list of images  and their gt in space separated format

def createDataset(outputPath, imagePathListFile, parentDirofImages, lookupFile, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath        : LMDB output path
	parentDirofImages : the path which would be prefixed with the images path
        imagePathListFile     : list of image path
        lookupFile        : a list of unique unicode points ( as their decimal values) possible
	lexiconList       : (optional) list of lexicon lists
        checkValid        : if true, check the validity of every image
    """
    with open(imagePathListFile) as f:
    	imagePathList = f.read().splitlines()
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    with open(lookupFile) as f:
        lookupTableList = f.read().splitlines()
    for i in xrange(nSamples):
        imagePath = parentDirofImages+imagePathList[i].split(' ')[0]
        labelString = imagePathList[i].split(' ')[1]
	label=""
	#the string is spliited into unicode code points
	#now each of the unicode points decimal value is found and
	#labels is a concatenated sequence of such values seprated by space
	#then the index of this value in the lookupfile is found so finally this can be used as the
	#class label for the network ( index begin from zero)

	for uc in unicode(labelString,'utf-8'):
		decimal_uc=str(ord(uc))
		label_index=str(lookupTableList.index(decimal_uc))
		label+=label_index + ' '
	label=label.strip()

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'r') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)
#createDataset(outputPath='./IM-hin-train/', imagePathListFile='hindi_synth_data.txt', parentDirofImages='/OCRData3/Data/minesh_render/rendering/', lookupFile='hindi_ml.txt')
createDataset(outputPath='./IH-test-lmdb/', imagePathListFile='test.txt', parentDirofImages='', lookupFile='hindi_final_lookup.txt')
