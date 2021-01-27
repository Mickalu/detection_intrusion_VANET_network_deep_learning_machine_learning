MIN3=999999999
MIN5=999999999
MIN7=999999999
MAX3=0
MAX5=0
MAX7=0
TOT3=0
TOT5=0
TOT7=0
counts="`pwd`/counts"
for item in `ls *tgz`
do
  tar xf $item
  FOLDERNAME=`ls work/ul/ul_vertsys/ul_wqy57`
  cd "work/ul/ul_vertsys/ul_wqy57/$FOLDERNAME/veins-maat/simulations/securecomm2018/results"

  for log in `ls JSONlog*.json`
  do
    if ls *sca | grep 'start=3' #this is 0 iff a .sca file with start=3 exists
    then
      COUNT=`grep 'type":3' $log | wc -l`
      echo $COUNT >> $counts
      TOT3=$((TOT3 + COUNT))
      if test $COUNT -lt $MIN3
      then
        MIN3=$COUNT
      fi
      if test $COUNT -gt $MAX3
      then
        MAX3=$COUNT
      fi
    elif ls *sca | grep 'start=5' #this is 0 iff a .sca file with start=5 exists
    then
      COUNT=`grep 'type":3' $log | wc -l`
      echo $COUNT >> $counts
      TOT5=$((TOT5 + COUNT))
      if test $COUNT -lt $MIN5
      then
        MIN5=$COUNT
      fi
      if test $COUNT -gt $MAX5
      then
        MAX5=$COUNT
      fi
    elif ls *sca | grep 'start=7' #this is 0 iff a .sca file with start=7 exists
    then
      COUNT=`grep 'type":3' $log | wc -l`
      echo $COUNT >> $counts
      TOT7=$((TOT7 + COUNT))
      if test $COUNT -lt $MIN7
      then
        MIN7=$COUNT
      fi
      if test $COUNT -gt $MAX7
      then
        MAX7=$COUNT
      fi
    else
      echo "error counting: $item -- $log"
    fi
  done

  cd ../../../../../../../../..
  rm -rf work
done
echo "low density: ${MIN3} ${MAX3} (total ${TOT3})"
echo "medium density: ${MIN5} ${MAX5} (total ${TOT5})"
echo "high density: ${MIN7} ${MAX7} (total ${TOT7})"
