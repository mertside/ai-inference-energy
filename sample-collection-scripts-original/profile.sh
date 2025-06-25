#! /usr/bin/env python
# Python power utility to measure power consumed by any process
#
# Reports NVidia SMI data
#
# No root privileges are necessary
#
# It is recommended that the running time of a process is at least 'X' seconds
# to get acceptable power consumption results

import os
import time
import signal
import subprocess
import sys
import random

#r = str(random.random())
# outputFileDirectory='/home/ghali/output/'
# outputFileStartPattern = 'dpgpu_mets_'
# outputFile = outputFileDirectory+outputFileStartPattern+str(size)
outputFile = os.path.join(sys.path[0], "changeme")

outputFormat = "csv"
interval = "50"    # millisecond
initTime = 0

#nvidiaSmiCmd = "nvidia-smi --query-gpu=timestamp,index,power.draw,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,clocks.sm,clocks.mem,clocks.gr --format="+outputFormat+" --loop-ms="+interval+" -f "+ outputFile

nvidiaSmiCmd = "dcgmi dmon -i 0 -e 1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,203,204,210,211,155,156,110 -d "+str(interval)+" > changeme"

#nvidiaSmiCmd = "dcgmi dmon -e 110,155,156,1005,1006,1007 -d 1 > changeme"

def get_power_info(cmd):
    global initTime
    global pid
    initTime = time.time()
    print ("profile cmd:",cmd)
    pid = os.fork()
    if pid == 0:
        code = os.system(nvidiaSmiCmd)
    else:
        os.system(cmd)
        os.system("killall -9 dcgmi")
        print("Interval: "+str(round(time.time() - initTime, 2))+" s")
    return

if __name__ == '__main__':
    if len(sys.argv) < 2:
        get_power_info('')
    else:
        get_power_info(' '.join(str(x) for x in sys.argv[1:]))
