#!/bin/bash
# Run all on all configurations for this platform

# Whether to run shmoo on the big or LITTLE cores
USE_BIG=1

# Whether to use external power meter (0 or 1)
USE_POWERMON=0

# Must run script with root privileges
if [ "$(id -u)" -ne 0 ]
then
  echo "Please run with root privileges"
  exit 1
fi

# Get the app to run (must match directory structure)
APP=$1
if [ -z "$APP" ]
then
  echo "Usage:"
  echo "  $0 <application>"
  exit 1
fi

PRERUN="apps/$APP/power-control/pre-run.sh"
if [ ! -e "$PRERUN" ]
then
  echo "pre-run script not found: $PRERUN"
  exit 1
fi
source "$PRERUN"

export POET_DISABLE_CONTROL=1
export HEARTBEAT_ENABLED_DIR=heartenabled/
rm -Rf ${HEARTBEAT_ENABLED_DIR}
mkdir -p ${HEARTBEAT_ENABLED_DIR}

RESULTS_FILE="${PREFIX}.results"
FULL_RESULTS="${PREFIX}_full.results"
POWER_MON=./powerQoS/pyWattsup-hank.py

# System properties
#BIG_FREQUENCIES=(2000000 1900000 1800000 1700000 1600000 1500000 1400000 1300000 1200000 1100000 1000000 900000 800000 700000 600000 500000 400000 300000 200000)
BIG_FREQUENCIES=(2000000 1900000 1800000)
NUM_BIG_FREQUENCIES=${#BIG_FREQUENCIES[*]}
#BIG_CORES_START=4
BIG_CORES_START=4
BIG_CORES_END=7
LITTLE_FREQUENCIES=(1400000 1300000 1200000 1100000 1000000 900000 800000 700000 600000 500000 400000 300000 200000)
NUM_LITTLE_FREQUENCIES=${#LITTLE_FREQUENCIES[*]}
LITTLE_CORES_START=0
LITTLE_CORES_END=3

#Creating parameters for fixed cores max
CORES_START=0
CORES_END=7
FREQUENCIES=(2000000 1400000)
#FREQUENCIES=(2000000 1700000 1600000 1500000 1400000 1300000 1200000)
#FREQUENCIES=(100000 200000 400000 600000 700000 800000 1000000 1200000 1400000)
NUM_FREQUENCIES=2
#NUM_FREQUENCIES=3
hex_length=2
hex_list=(0x88 0xff)
#hex_list=(0x88 0xff 0x99 0xcc 0xaa)
#hex_list=(0x99 0xcc 0xaa 0x88 0xff 0x8f 0xaf 0xf1 0xf5 0x55)
#hex_list=(0xd8 0x17 0x14 0x24 0x99 0x47 0xd9)
for ((i=0; i<hex_length; i++))
do
    mask=${hex_list[$i]}
    echo $mask
  for (( j=0; j<NUM_FREQUENCIES; j++ ))
  do
      freq=${FREQUENCIES[$j]}
      echo "Setting speed $freq on cpu$k"
      echo "$freq" > /sys/devices/system/cpu/cpu$k/cpufreq/scaling_max_freq
    sleep 1

    freq=${FREQUENCIES[$j]}
    hr=''
    power=''
    joules=''
    c=1
    while [[ $hr = '' ]]||[[ $power = '' ]]||[[ $joules = '' ]]||[[ $c -le 0 ]]
    do
      if [ $USE_POWERMON -gt 0 ]
      then
        $POWER_MON start
      fi

      CMD=(taskset $mask "${BINARY}" "${ARGS[@]}")
      echo "${CMD[@]}"
      "${CMD[@]}"

      if [ $USE_POWERMON -gt 0 ]
      then
        $POWER_MON stop > power.txt
        power2=$(awk '/Pavg/ {print $2}' power.txt)
        joules2=$(awk '/Joules/ {print $2}' power.txt)
        cp power.txt "power_$mask-$freq.txt"
      fi
    
      hr=$(tail -n 1 heartbeat.log | awk '// {print $4}')
      power=$(tail -n 1 heartbeat.log | awk '// {print $10}')
      joules=$(echo "scale=4; $NUMBER / $hr * $power" | bc)
      c=$(echo "$power > 0" | bc)


      source hb_cleanup.sh
    done


    #ALL CHANGES
    custom_tab='    '
    #all data gather
    while read h; do
      hr_all=$h| awk '// {print $4}'
      power_all=$h | awk '// {print $10}'
      joules_all=$(echo "scale=4; $NUMBER / $hr_all * $power_all" | bc)
      c=$(echo "$power_all > 0" | bc)
      if [ ! -f "$FULL_RESULTS" ]
      then
          echo "cores ${custom_tab} Freq ${custom_tab}   Rate ${custom_tab}   Power ${custom_tab}   Energy" > "$FULL_RESULTS"
      fi
      echo "$mask ${custom_tab} $freq ${custom_tab} $hr_all ${custom_tab} $power_all ${custom_tab} $joules_all" >> "$FULL_RESULTS"
     done <heartbeat.log
     #########



    if [ ! -f "$RESULTS_FILE" ]
    then
      echo "cores freq Rate Power Energy WU_PWR_AVG WU_ENERGY" > "$RESULTS_FILE"
    fi
    echo "$mask $freq $hr $power $joules $power2 $joules2" >> "$RESULTS_FILE"
    
    cp heartbeat.log "heartbeat_$mask-$freq.log"

    sleep 20

  done
done
