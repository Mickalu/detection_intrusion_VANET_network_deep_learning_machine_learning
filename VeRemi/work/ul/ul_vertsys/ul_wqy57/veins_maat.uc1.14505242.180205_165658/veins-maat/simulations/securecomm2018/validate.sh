#!/bin/sh

for x in {1,2,4,8,16,32,33,64}; do
  mkdir $x
  if ! ./run -u Cmdenv -c AttackerType$x -r0 validate.ini
  then
    echo "Attacker Type $x threw an error"

    exit 1
  else
    mv *.json $x/
    #TODO test output validity?
  fi

done
