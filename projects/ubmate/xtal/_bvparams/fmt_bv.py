file = open("bvparm2006.cif")
lines = file.readlines()
file.close()
header = ''
while True:
    x = lines.pop(0)
    if x == "## Start\n": break
    header = header + x
#print(header)

file = open("bvparm.py",'w')
file.write('"""\n')
file.write(header)
file.write('"""\n')
file.write("bvparm = {\n")
for line in lines:
    x = line.split()
    xx = '"%s%s%s%s":[%s,%s],\n' % (x[0],x[1],x[2],x[3],x[4],x[5])
    file.write(xx)
file.write("}")

