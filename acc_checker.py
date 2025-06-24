import sys

src_file = sys.argv[1]
orig_file = sys.argv[2]
pred_file = sys.argv[3]

with open(src_file) as s, open(orig_file) as o, open(pred_file) as p:
    slines = s.readlines()
    olines = o.readlines()
    plines = p.readlines()
   
    total = 0
    matched = 0
   
    for sline, oline, pline in zip(slines, olines, plines):
        oline, pline = oline.strip(), pline.strip()
        
        if oline != pline:
            print(f"MISMATCH : SOURCE: {sline} \t || \t ORIGINAL : {oline} \t || \t PRED : {pline}")
        else:
            matched += 1
        
        total += 1
        
    print(f"TOTAL: {total}, MATCHED : {matched}, ACCURACY: {(matched * 100) / total}")
    