import sys

def gen_center_frames(phnFrmBdry,inpFrms,outFrms,inpCntrFrms,outCntrFrms):
    for line in phnFrmBdry:
        (startFrameNo,endFrameNo,phn) = line.strip().split()
        startFrameNo = int(startFrameNo)
        endFrameNo = int(endFrameNo)
        if(startFrameNo <= endFrameNo):
            midPos = (endFrameNo + 1 - startFrameNo)/2
            for frameNo in range(endFrameNo + 1 - startFrameNo):
                inpFrame = inpFrms.readline()
                outFrame = outFrms.readline()
                if(frameNo == midPos):
                    inpCntrFrms.write(inpFrame)
                    outCntrFrms.write(outFrame)

if(len(sys.argv)!=2):
    print("ERROR:Argument count mismatch.Run the program as follows:")
    print("python timit_gen_centerframes.py <file list>")
    exit()
    

fileList = open(sys.argv[1])
for line in fileList:
    fname = line.strip()
    phnFrmBdry = open(fname+".phn_fr_bndry")
    inpFrms = open(fname+"_8.htk.mfcc")
    outFrms = open(fname+".ann_out")
    inpCntrFrms = open(fname+"_8_cntr.htk.mfcc",'w')
    outCntrFrms = open(fname+"_cntr.ann_out",'w')
    gen_center_frames(phnFrmBdry,inpFrms,outFrms,inpCntrFrms,outCntrFrms)
    inpFrms.close()
    outFrms.close()
    inpCntrFrms.close()
    outCntrFrms.close()

fileList.close()    
