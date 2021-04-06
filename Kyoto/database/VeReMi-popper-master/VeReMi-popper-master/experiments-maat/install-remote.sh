#!/bin/sh

#run this on the machine where maat is stored in the dir specified below

MAATDIR="${HOME}/research/maat"
REMOTE="bwunicluster.scc.kit.edu:maat/"

OLD="`pwd`"

cd ${MAATDIR}
gradle jar
scp bin/libs/Maat-1.0.jar "${REMOTE}"Maat.jar
scp testdata/JSONlog-0-short.json "${REMOTE}"JSONlog-0-short.json
scp detectors.xml "${REMOTE}"detectors.xml
scp job.moab "${REMOTE}"job.moab
scp jobscript.sh "${REMOTE}"jobscript.sh

cd ${OLD}

scp runall.sh "${REMOTE}/../maat-runall.sh"
scp pre-validate.sh "${REMOTE}pre-validate.sh"

echo "Install complete, please SSH to ${REMOTE} and execute ./maat-runall.sh to execute experiments"
