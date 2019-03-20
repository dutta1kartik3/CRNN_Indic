##Used to Convert The numbers output by test code into unicode characters
##Requires as input: The Lookup file and the text file containing Predicted output

f = open("hindi_final_lookup.txt", 'r')
answer = {}
count = 1
for line in f:
    k= int(line.strip())
    answer[count] = k
    count+=1

f.close()

infile = "input.txt"

words = []
with open(infile, 'r') as f:
   for line in f:
        string = ""
        line = line.split()
        for ch in line[2:]:
            if ch!='0':
                #print(chr(int(ch)), ch)
                string += chr(answer[int(ch)])
        words.append(string)

for word in words:
        print(word)
