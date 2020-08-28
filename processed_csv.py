from __future__ import print_function, division  

import os, sys
import codecs
import gzip
import numpy as np
import shutil
import time

import torch


# If all files exist then return 'True' else 'False'
def isExistAllFiles(path, files, debug=False):
    isExistList = [os.path.exists(os.path.join(path, f)) for f in files]
    if debug: print(isExistList)
    return all(isExistList)

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def open_maybe_compressed_file(path):
    """Return a file object that possibly decompresses 'path' on the fly.
       Decompression occurs when argument `path` is a string and ends with '.gz' or '.xz'.
    """
    if not isinstance(path, torch._six.string_classes):
        return path
    if path.endswith('.gz'):
        import gzip
        return gzip.open(path, 'rb')
    if path.endswith('.xz'):
        import lzma
        return lzma.open(path, 'rb')
    return open(path, 'rb')

def read_sn3_pascalvincent_tensor(path, strict=True):
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # typemap
    if not hasattr(read_sn3_pascalvincent_tensor, 'typemap'):
        read_sn3_pascalvincent_tensor.typemap = {
            8: (torch.uint8, np.uint8, np.uint8),
            9: (torch.int8, np.int8, np.int8),
            11: (torch.int16, np.dtype('>i2'), 'i2'),
            12: (torch.int32, np.dtype('>i4'), 'i4'),
            13: (torch.float32, np.dtype('>f4'), 'f4'),
            14: (torch.float64, np.dtype('>f8'), 'f8')
        }
        
    # read
    with open_maybe_compressed_file(path) as f:
        data = f.read()

    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert nd >= 1 and nd <= 3
    assert ty >= 8 and ty <= 14
    m = read_sn3_pascalvincent_tensor.typemap[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)

def read_image_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 3)
    return x

def read_label_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()

def createProcessedDataFiles(inPath, inFiles, outPath, outFiles):
    training_set = (
        read_image_file(os.path.join(inPath, inFiles[0])),
        read_label_file(os.path.join(inPath, inFiles[1]))
    )
    test_set = (
        read_image_file(os.path.join(inPath, inFiles[2])),
        read_label_file(os.path.join(inPath, inFiles[3]))
    )
    with open(os.path.join(outPath, outFiles[0]), 'wb') as f:
        torch.save(training_set, f)
    with open(os.path.join(outPath, outFiles[1]), 'wb') as f:
        torch.save(test_set, f)
    return

def decompressDataFiles(path_raw, datafiles):
    for f in datafiles:
        inFile  = os.path.join(path_raw, f)
        outFile = os.path.join(path_raw, f.replace('.gz', ''))
        with gzip.open(inFile, 'r') as f_in, open(outFile, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)      
    return

def main():
    RawPath = 'D:\\GitWork\\mnist\\raw\\'
    ProcessedPath = 'D:\\GitWork\\mnist\\processed\\'

    if not os.path.exists(ProcessedPath):
        os.makedirs(ProcessedPath)

    Resources = [
        "train-images-idx3-ubyte.gz", 
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]

    dataFiles  = [f.replace('.gz', '') for f in Resources]
    
    ptFiles = [ 'training.pt', 'test.pt' ]

    isExistPtFiles = isExistAllFiles(ProcessedPath, ptFiles)
    print('Is *.pt exist? ', isExistPtFiles)
    
    if not isExistPtFiles:
        isExistDataFiles = isExistAllFiles(RawPath, dataFiles)
        if not isExistDataFiles:
            isExistResources = isExistAllFiles(RawPath, Resources)
            if not isExistResources:
                print('Resources not exist, system exit...')
                sys.exit(0)
            
            print('Data files not exist, decompressing...')
            decompressDataFiles(RawPath, Resources)

            isExistDataFiles = isExistAllFiles(RawPath, dataFiles)
            if not isExistDataFiles:
                print('Data files not exist, system exit...')
                sys.exit(0)
            
            print('Data files decompressed.')
        
        createProcessedDataFiles(RawPath, dataFiles, ProcessedPath, ptFiles)

    for f in ptFiles:
        isExist = os.path.exists(os.path.join(ProcessedPath, f))
        print('{} exist: {}'.format(f, isExist))
    
    return (0)

        
if __name__=='__main__':
    main()