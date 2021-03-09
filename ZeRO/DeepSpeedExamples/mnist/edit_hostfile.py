f = open('hostfile.txt', 'r')
lines = f.readlines()
f.close()
result = {}
for line in lines:
    temp = line.strip()
    if temp in result:
        result[temp] += 1
    else:
        result[temp] = 1
w_r = open('hostfile.txt','w')
for key in result:
    w_r.write(key+' '+f'slots={str(result[key])}\n')
