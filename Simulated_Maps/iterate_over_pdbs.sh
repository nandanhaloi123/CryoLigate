#!/bin/bash
# module load chimerax
# Define the input file
INFILE=names.txt

# Read the input file line by line
while read -r LINE
do
    cp create_sim_map.cxc temp.cxc
    sed -i "s/TEMP/$LINE/g" temp.cxc
    chimerax temp.cxc
    # printf '%s\n' "$LINE"
    rm temp.cxc
done < "$INFILE"