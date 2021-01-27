#!/bin/sh


# test OMNET
if ! ${HOME}/omnetpp/omnetpp-5.1.1/bin/opp_run -v;
then
  echo 'opp_run -v failed'
  exit 1
fi

# test SUMO
if ! ${HOME}/sumo/sumo-0.30.0/bin/sumo --version;
then
  echo 'SUMO failed'
  exit 1
elif [ -z "`${HOME}/sumo/sumo-0.30.0/bin/sumo --version | grep 'TRACI PROJ GDAL'`" ];
then
  echo 'Your SUMO version lacks one or more of: TRACI, PROJ, GDAL'
  exit 1
fi

# test VEINS
cd veins-maat/simulations/securecomm2018
export PATH="${PATH}:${HOME}/sumo/sumo-0.30.0/bin"
export SUMO_HOME="${HOME}/sumo/sumo-0.30.0"
if ! ./validate.sh;
then
  echo 'validation script in your project failed, please look at the JSON output and remove it aftwards'
  exit 1
else
  echo 'removing validation output...'
  find . -name '*.json' | xargs rm
  rm -rf results
  rm dua.static.log
  rm dua.static.summary.xml
  rm dua.static.tripinfo.xml
  rm e1_OffRamp_output.xml
  rm e1_OnRamp_output.xml
  rm e1_begin_output.xml
  rm e1_end_output.xml
  rm e1_link_output.xml
  rm .cmdenv-log
  rmdir 1 2 4 8 16 32 33 64
fi

cd ~
