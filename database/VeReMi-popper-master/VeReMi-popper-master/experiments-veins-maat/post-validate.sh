#!/bin/sh

NUMSIMS=`opp_run -q numruns veins-maat/simulations/securecomm2018/omnetpp.ini -c AttackerType1 | grep 'Number of runs: ' | cut -d' ' -f4`

# attacker types: 1,2,4,8,16 --> 5 types
NUMSIMS=`echo "${NUMSIMS} * 5"|bc -l`

if [ "`ls veins-maat-output | grep tgz | wc -l`" = "$NUMSIMS"]
then
  echo "Simulations were all successful"
else
  echo "veins-maat-output does not contain the expected amount of tgz files.."
fi
