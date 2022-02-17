#!/bin/sh

DIR=$(cd `dirname $0`; pwd)
FILENAME=${0##*/}
ALGNAME=${FILENAME%.*}
CONFIGFILENAME="${DIR}/config.ini"
TASKNAME=${DIR##*/}

source $CONFIGFILENAME

nohup sh scripts/comparison/comparison_general.sh $ALGNAME $MODEL $CLASSNUM $RATE $TOP1 $TOP5 $PARAM $FLOP > ./logs/1log_${TASKNAME}_${ALGNAME}.log &
