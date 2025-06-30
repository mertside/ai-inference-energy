#!/bin/bash

numRuns=2
mode="dvfs"
arch="GA100" #GV100
oldfile="changeme"
#m_freq=877 #V100
m_freq=1215 #A100
sleep_interval=1
t=0

declare -a progs=("lstm") #executable name
declare -a apps=("LSTM") #any name
declare -a appParams=(" > results/LSTM_RUN_OUT") 

declare -a freqs=(1410 1395 1380 1365 1350 1335 1320 1305 1290 1275 1260 1245 1230 1215 1200 1185 1170 1155 1140 1125 1110 1095 1080 1065 1050 1035 1020 1005 990 975 960 945 930 915 900 885 870 855 840 825 810 795 780 765 750 735 720 705 690 675 660 645 630 615 600 585 570 555 540 525 510)

#declare -a freqs=(1410)

for c_freq in "${freqs[@]}"
do
	./control $m_freq $c_freq
	nRuns=$((numRuns-1))
	for i in $(seq 0 $nRuns)
	do
		appID=0
		for app in "${apps[@]}"
    	do	
			t=$((t+1))
			# GET RESULTS WITH NEW FREQUENCY
			echo "### $app - $c_freq (MHz) - Iteration: $i ###"
			file="results/$arch-$mode-$app-$c_freq-$i"
			
			#pushd apps
			#appPath=$(pwd)/
			#popd
					
			#appCmd="$appPath${progs[$appID]}${appParams[$appID]}"
			appCmd="python ${progs[$appID]}.py${appParams[$appID]}"

      START=$(date +%s)
      echo $appCmd
			./profile $appCmd
      END=$(date +%s)

			cp $oldfile $file
			rm -f $oldfile
			sleep $sleep_interval
      
      DIFF=$(( $END - $START ))
      #echo "...Execution of application script took $DIFF seconds."
      # save execution time
      #wall_T="${DIFF//[$'\t\r\n ']}"
      #echo $wall_T
      #printf '%s\n' $c_freq $wall_T | paste -sd ',' >> results/$arch-dvfs-lstm-perf.csv

			# SAVE time for LSTM
			if [[ $app == "LSTM" ]]
			then
				v=$(sed -n '20p' results/LSTM_RUN_OUT)
				tokens=( $v )
				exec_time=${tokens[0]}
				echo $exec_time
				exec_time="${exec_time//[$'\t\r\n ']}"
				printf '%s\n' $c_freq $exec_time | paste -sd ',' >> results/$arch-dvfs-lstm-perf.csv
			fi
			
      appID=$((appID+1))
		done #apps
   	done #runs
done #freqs

echo "DONE!!!"

# revert the core frequency
#c_freq=1380 #V100
c_freq=1410 #A100
./control $m_freq $c_freq
echo $t
