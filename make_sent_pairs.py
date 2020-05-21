from itertools import permutations

data = open('ChatbotData.csv', 'r', encoding='utf-8')

output = open('ChatbotData_pair.txt', 'w', encoding='utf-8')

lines = data.readlines()

data.close()

for n in range(len(lines)):
    line = lines[n]
    if n == 0:
        continue
    line = line.replace('\n', '') 
    output.write(line.split(',')[0] + '\t' + line.split(',')[1])
    output.write('\n')
        
output.close()
