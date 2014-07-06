import sys

def create_phnmap(mapFname,invert):
    phnMap = {}
    mapfh = open(mapFname)
    for line in mapfh:
        (phn,phnClass) = line.strip().split('\t')
        if(!invert):
            phnMap[phn] = phnClass
        else:
            phnMap[phnClass] = phn
    mapfh.close()
    return phnMap

def find_phoneme(startFrameNo,endFrameNo,preds,phnMap):
    phns = {}
    for lineNo in range(startFrameNo,endFrameNo+1):
        phnClass = lineNo.strip()
        if(phnClass in phns.keys()):
            phns[phnClass] += 1
        else:
            phns[phnClass] = 1
    max = 0
    for phnClass in phns.keys():
        if(max<phns[phnClass]):
            maxPhnClass = phnClass
            max = phns[phnClass]
    return [maxPhnClass,phns[phnClass]/float((endFrameNo+1-startFrameNo))]
            
if(len(sys.argv)!=5):
    print("ERROR:argument count mismatch.Run it as follows:")
    print("python timit_accuracy.py <file list> <timit phoneme map file> <predictions file> <phoneme save file>")
    exit()
fileList = open(sys.argv[1])
preds = open(sys.argv[3])

phnPreds = open(sys.argv[4])
phnMap = create_phnmap(sys.argv[2],True)
for line in fileList:
    (startFrameNo,endFrameNo,phn) = line.strip().split()
    (phnClass,phnPercent) = find_phoneme(startFrameNo,endFrameNo,preds,phnMap)
    if(phnPercent > 0.7):
        phnPreds.write(phnMap[phnClass]+"\n")
    else:
        phnPreds.write(phnMap[phnClass]+"\t *** \n")
fileList.close()
preds.close()
phnPreds.close()
