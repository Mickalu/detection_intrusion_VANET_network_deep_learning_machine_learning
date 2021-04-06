#!/bin/sh

# script assumes install was successful

# install dir of veins
cd veins-maat

# these scripts submit jobs:


for x in {1,2,4,8,16}; do
  if ! ./jobscript.sh AttackerType$x simulations/securecomm2018/omnetpp.ini
  then
    echo "Attacker Type $x threw an error"
    exit 1
  fi  
done

