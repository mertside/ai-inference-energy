#!/bin/bash 
# args=$@
echo "***Starting Control***"
# echo "Args:"$args

# #Parsing arguments
# pl=$1
# freq=$2
# basePath=$3

# cmd=${args[@]/$pl}
# cmds=${cmd[@]/$freq}
# appcmd=${cmds[@]/$basePath}
# # prof=utility/profile
# prof=profile
# profilecmd=$basePath$prof
# echo $profilecmd
# mem_freq=877
#enforcing GPU control parameters
m_freq=$1
c_freq=$2
echo $m_freq $c_freq
dcgmi config --set -a $m_freq,$c_freq
sleep 2

# #Enable PM
# sudo nvidia-smi -pm 1
# # ***set power limit ***
# if [ $pl -ne 250 ]
# then 
# sudo nvidia-smi -i 0 -pl $pl
# fi
# # *** set GPU frequency ***
# if [ "$freq" != "P0" ]
# then
# #sudo nvidia-smi -i 0 -lgc $freq
# sudo nvidia-smi -i 0 -ac 877,$freq
# fi

# *** launching application and start collecting metrics***
# $profilecmd $appcmd

#***Changing GPU control parameter to default ***

# #***setting default pl (i.e.) TDP***
# if [ $pl -ne 250 ]
# then
# sudo nvidia-smi -i 0 -pl 250
# fi

# #***Resets the GPU clocks to the default values***
# if [ "$freq" != "P0" ]
# then
# #sudo nvidia-smi -i 0 -rgc
# sudo nvidia-smi -i 0 -rac
# fi

# #*** disable PM ***
# sudo nvidia-smi -pm 0

echo "***Exiting Control***"
