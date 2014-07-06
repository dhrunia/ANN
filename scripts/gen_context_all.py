import sys


def gen_context(inpFname,outFname,nFrames):
    in_file = open(inpFname,'r')
    out_file = open(outFname, 'w')

    left=[]
    right=[]
    new=[]
    temp=[]

    content = in_file.read().splitlines()
    lc = len(content)
    last = lc-nFrames

    for i in range(lc):

            if(i < nFrames):
                    middle = [content[i]] * nFrames
                    middle += content[i:i+nFrames+1]

            elif(i >= last):
                    middle = content[i-nFrames:i]
                    middle += [content[i]] * (nFrames+1)

            else:
                    middle = content[i-nFrames : i+nFrames+1]
            for n in middle:
                    out_file.write(n + " ")
                    #print n + " ",
            out_file.write("\n")
            #print ""
    out_file.close()
    in_file.close()

if(len(sys.argv)!=2):
    print("ERROR:argument count mismatch.Run the program as follows")
    print("python gen_context_all.py <file list>")
    exit()
            
fileList = open(sys.argv[1])

for line in fileList:
    fname = line.strip()
    gen_context(fname+"_8.htk.mfcc",fname+"_8_context.htk.mfcc",2)
    
fileList.close()
    
    
    
    

