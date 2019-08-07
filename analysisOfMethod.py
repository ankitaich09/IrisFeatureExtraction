# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 12:56:46 2019

@author: cisguest
"""

"""
take 8 vectors and make 4 pairs
maxAyushX , maxAnkitX
minAyushX , minAnkitX
maxAyushY , maxAnkitY
minAyushY , minAnkitY

once this is done set a threshold value (0.2)

for every pair, based on the differences classify as 

MaC - Max Values are close
MiC - Min values are Close
BoC - Both are close
BD - Both Different

"""
def classify(maxTup, minTup):
    #takes in 2 tuples and classifies
    ankMax = maxTup[1]
    aayMax = maxTup[0]
    ankMin = minTup[1]
    aayMin = minTup[0]
    
    if abs(ankMax - aayMax) <= 0.2 and abs(ankMin - aayMin) > 0.2:
        cla = "MaC"
    elif abs(ankMax - aayMax) <= 0.2 and abs(ankMin - aayMin) <= 0.2:
        cla = "BoC"
    elif abs(ankMax - aayMax) > 0.2 and abs(ankMin - aayMin) <= 0.2:
        cla = "MiC"
    else:
        cla = "BD"
    return cla


maxAyushX = [0.29,0.38,0.40,0.22,0.39,0.33,0.28,0.65,0.16]
maxAnkitX = [0.57,0.38,0.60,0.47,0.45,0.49,0.68,0.36,0.39]

minAyushX = [-0.47,-0.29,-0.22,-0.06,-0.19,-0.31,-0.55,-0.09,-0.40]
minAnkitX = [-0.45,-0.35,-0.33,-0.35,-0.37,-0.38,-0.40,-0.43,-0.40]

maxAyushY = [0.14,0.04,0.12,0.50,0.05,0.68,0.20,0.27,0.23]
