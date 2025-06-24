import sys

orig_file = sys.argv[1]
pred_file = sys.argv[2]

with open(orig_file) as o, open(pred_file) as p:
    olines = o.readlines()
    plines = p.readlines()
   
    total = 0
    matched = 0
   
    for oline, pline in zip(olines, plines):
        oline, pline = oline.strip(), pline.strip()
        
        if oline != pline:
            print(f"MISMATCH : ORIGINAL : {oline} \t || \t PRED : {pline}")
        else:
            matched += 1
        
        total += 1
        
    print(f"TOTAL: {total}, MATCHED : {matched}, ACCURACY: {(matched * 100) / total}")
    