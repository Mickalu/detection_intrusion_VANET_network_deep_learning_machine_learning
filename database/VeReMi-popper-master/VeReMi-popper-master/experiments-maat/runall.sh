#!/bin/bash

cd ~/maat
module load devel/java_jdk
chmod +x jobscript.sh
./jobscript.sh ../veins-maat-output
