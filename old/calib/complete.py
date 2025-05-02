data = []
with open('left2.calib', 'r') as f:
    for line in f:
        line = line.strip('\n')
        row = line.split(' ')
        for each in row:
            each = int(each)
        data.append(row)
        

print(data)

