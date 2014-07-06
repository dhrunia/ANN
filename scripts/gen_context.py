import sys

in_file = open(sys.argv[1],'r')
out_file = open(sys.argv[2], 'w')
frames = int(sys.argv[3])

left=[]
right=[]
new=[]
temp=[]
#with open(sys.argv[1],'r') as in_file

content = in_file.read().splitlines()
lc = len(content)
last = lc-frames

for i in range(lc):

	if(i < frames):
		middle = [content[i]] * frames
		middle += content[i:i+frames+1]

	elif(i >= last):
		middle = content[i-frames:i]
		middle += [content[i]] * (frames+1)

	else:
		middle = content[i-frames : i+frames+1]
	for n in middle:
		out_file.write(n + " ")
		#print n + " ",
	out_file.write("\n")
	#print ""