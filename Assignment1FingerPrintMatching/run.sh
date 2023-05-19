#!/bin/bash
pip install fingerprint-feature-extractor
pip install fingerprint_enhancer
argument1=$1
argument2=$2
python check.py $1 $2
python hough.py $1 $2
python genetic.py $1 $2


        
            
    


