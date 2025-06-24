import sys
import re

inp_file = sys.argv[1]
src_file = sys.argv[2]
gen_file = sys.argv[3]
mode = sys.argv[4]

tups = re.compile("\((.*?)\)+")

with open(inp_file) as ip, open(src_file) as s, open(gen_file) as g:
    slines = s.readlines()
    glines = g.readlines()
    ilines = ip.readlines()
    for sid, sline in enumerate(slines):
        sline = sline.strip().split(" ")
        ipline = ilines[sid].strip()
        gline = glines[sid].strip()
        #print(sline)
        #print(gline)
        #print(tups.findall(gline))
        templates = [x.strip().split() for x in tups.findall(gline)]
        #print(templates)
        
        current_tpl = templates[0]
        ctpl_idx = 0
        tidx = 0
        mask_map = {}
        for widx in range(len(sline)):
            w = sline[widx]
            if tidx > len(current_tpl) - 1:
                tidx = 0
                current_tpl = templates[ctpl_idx + 1]
                ctpl_idx += 1
            
            tw = current_tpl[tidx]
            if tw.startswith("W_"):
                if tw not in mask_map:
                    mask_map[tw] = None
                mask_map[tw] = w
                
            tidx += 1
                
        for k, v in mask_map.items():
            if mode == "all":
                print(f"IN: {ipline} || OUT: {gline} || W: {k} || V: {v}")
            elif mode == "input":
                print(f"IN: {ipline} || OUT: {gline} || W: {k}")
            elif mode == "output":
                print(f"V: {v}")
        
        
        
        