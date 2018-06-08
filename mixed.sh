#!/bin/bash
# Run all on all configurations for this platform

# Whether to run shmoo on the big or LITTLE cores
USE_BIG=1

# Whether to use external power meter (0 or 1)
USE_POWERMON=0

# Must run script with root privileges
#if [ "$(id -u)" -ne 0 ]
#then
#  echo "Please run with root privileges"
#  exit 1
#fi
#
## Get the app to run (must match directory structure)
#APP=$1
#if [ -z "$APP" ]
#then
#  echo "Usage:"
#  echo "  $0 <application>"
#  exit 1
#fi
#
#PRERUN="apps/$APP/power-control/pre-run.sh"
#if [ ! -e "$PRERUN" ]
#then
#  echo "pre-run script not found: $PRERUN"
#  exit 1
#fi
#source "$PRERUN"
#
#export POET_DISABLE_CONTROL=1
#export HEARTBEAT_ENABLED_DIR=heartenabled/
#rm -Rf ${HEARTBEAT_ENABLED_DIR}
#mkdir -p ${HEARTBEAT_ENABLED_DIR}

RESULTS_FILE="${PREFIX}.results"
#POWER_MON=./powerQoS/pyWattsup-hank.py

# System properties
#BIG_FREQUENCIES=(2000000 1900000 1800000 1700000 1600000 1500000 1400000 1300000 1200000 1100000 1000000 900000 800000 700000 600000 500000 400000 300000 200000)
#BIG_FREQUENCIES=(2000000 1900000 1800000)
#NUM_BIG_FREQUENCIES=${#BIG_FREQUENCIES[*]}
##BIG_CORES_START=4
#BIG_CORES_START=4
#BIG_CORES_END=7
#LITTLE_FREQUENCIES=(1400000 1300000 1200000 1100000 1000000 900000 800000 700000 600000 500000 400000 300000 200000)
#NUM_LITTLE_FREQUENCIES=${#LITTLE_FREQUENCIES[*]}
#LITTLE_CORES_START=0
#LITTLE_CORES_END=3
# Set loop parameters
#if [ $USE_BIG -eq 1 ]
#then
#  FREQUENCIES=(${BIG_FREQUENCIES[@]})
#  CORES_START=$BIG_CORES_START
#  CORES_END=$BIG_CORES_END
#  NUM_FREQUENCIES=$NUM_BIG_FREQUENCIES
#else
#  FREQUENCIES=(${LITTLE_FREQUENCIES[@]})
#  CORES_START=$LITTLE_CORES_START
#  CORES_END=$LITTLE_CORES_END
#  NUM_FREQUENCIES=$NUM_LITTLE_FREQUENCIES
#fi

#Creating parameters for fixed cores max
CORES_START=0
CORES_END=7
#FREQUENCIES=(2000000 1400000 1300000)
FREQUENCIES=(20 19)
NUM_FREQUENCIES=2

#hex_list=(0x88 0xff 0x99 0xcc 0xaa)
hex_list=(0x99 0xcc 0xaa 0x88 0xff)
hex_length=5

for ((i=0; i<hex_length; i++))
do
    mask=${hex_list[$i]}
    echo $mask
  for (( j=0; j<NUM_FREQUENCIES; j++))
  do
    #for (( k=CORES_START; k<=CORES_END; k++ ))

      #if [ $USE_BIG -eq 1 ]; then
        # Not using this core - reduce speed to reduce power
        #freq=${FREQUENCIES[$NUM_FREQUENCIES - 1]}
      #else
      freq=${FREQUENCIES[$j]}
      #fi
      echo "Setting speed $freq on cpu$k"
      echo "$freq" #> /sys/devices/system/cpu/cpu$k/cpufreq/scaling_max_freq
    # Configure the CPU speeds
    # No DVFS on XU3, always uses performance governor.
    # Just set the max frequency to our target

      #echo "about to sleep"
      sleep 1

      #freq=${FREQUENCIES[$j]}
      #echo "this is the frequency after sleep"
      #echo $freq
  done
done
#    hr=''
#    power=''
#    joules=''
#    c=1
#    while [[ $hr = '' ]]||[[ $power = '' ]]||[[ $joules = '' ]]||[[ $c -le 0 ]]
#    do
#      if [ $USE_POWERMON -gt 0 ]
#      then
#        $POWER_MON start
#      fi
#
#      CMD=(taskset $mask "${BINARY}" "${ARGS[@]}")
#      echo "${CMD[@]}"
#      "${CMD[@]}"
#
#      if [ $USE_POWERMON -gt 0 ]
#      then
#        $POWER_MON stop > power.txt
#        power2=$(awk '/Pavg/ {print $2}' power.txt)
#        joules2=$(awk '/Joules/ {print $2}' power.txt)
#        cp power.txt "power_$mask-$freq.txt"
#      fi
#
#      hr=$(tail -n 1 heartbeat.log | awk '// {print $4}')
#      power=$(tail -n 1 heartbeat.log | awk '// {print $10}')
#      joules=$(echo "scale=4; $NUMBER / $hr * $power" | bc)
#      c=$(echo "$power > 0" | bc)
#
#      source hb_cleanup.sh
#    done
#
#    if [ ! -f "$RESULTS_FILE" ]
#    then
#      echo "cores freq Rate Power Energy WU_PWR_AVG WU_ENERGY" > "$RESULTS_FILE"
#    fi
#    echo "$mask $freq $hr $power $joules $power2 $joules2" >> "$RESULTS_FILE"
#
#    cp heartbeat.log "heartbeat_$mask-$freq.log"
#
#    sleep 20

#  done
#done
