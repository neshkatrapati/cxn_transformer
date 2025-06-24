import sys

src_file = sys.argv[1]

vmap = {
    "jump" : "I_JUMP",
    "look" : "I_LOOK",
    "walk" : "I_WALK",
    "run" : "I_RUN"
}

with open(src_file) as s:
    slines = s.readlines()
   
    for sline in slines:
        sline = sline.strip().split()
        words = set(sline)
        sel_words = ", ".join([ f"{x} -> {y}" for x,y in vmap.items() if x in words ])
        
        print(f"{' '.join(sline)} || {sel_words}")