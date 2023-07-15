import os

ops = {}
for _,_,files in os.walk("faultLogs"):
    for f in files:
        with open("faultLogs/" + f, "r") as fread:
            lines = fread.readlines()
            for line in lines:
                if "Count" in line:
                    idx = line.index("Count : ")
                    count = int(line[idx + 8:])
                if "OpName" in line:
                    idx = line.index("Ops.")
                    ops[line[idx + 4:]] = ops.get(line[idx + 4:], 0) + count
                    
for op, count in ops.items():
    print(op, count)

'''    
('IDENTITY', 493)
('CONV2D', 87)
('MAXPOOL', 58)
('ARGMAX', 49)
('RELU', 101)
('BIASADD', 111)
'''
