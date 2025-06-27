import sys
import json

read_from = sys.argv[1]
wmap_file = sys.argv[2]
check_with = sys.argv[3]


rmap = {
    "I_JUMP" : "jump",
    "I_WALK" : "walk",
    "I_LOOK" : "look",
    "I_RUN" : "run",
    "I_TURN_RIGHT" : "right",
    "I_TURN_LEFT" : "left"
}
rmap_r = {y : x for x, y in rmap.items()}

wmaps = json.loads(open(wmap_file).read())

total = 0
correct = 0
with open(read_from) as f, open(check_with) as cw:
    
    cwlines = cw.readlines()
    for li, line in enumerate(f):
        total+=1
        line = line.strip().split()
        #print(line)
        
        wmap = wmaps[li]
        rwmap = {y : x for x, y in wmap.items()}
        #print(rwmap)
        new_line = []
        for w in line:
            if w.startswith("W_") and (w in rwmap):
                t = rwmap[w]
                if t in rmap_r:
                    new_line.append(rmap_r[t])
            else:
                new_line.append(w)
                
        #print(new_line)
        
        #break
        
        new_line = " ".join(new_line)
        
        print(new_line)
        cwline = cwlines[li].strip()
        print(cwline)
        
        if new_line == cwline:
            correct +=1 
        
        print()
        # wf.write(new_line+"\n")

print(correct, total, correct * 100/ total)