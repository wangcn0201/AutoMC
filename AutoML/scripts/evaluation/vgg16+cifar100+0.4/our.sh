#!/bin/sh

DIR=$(cd `dirname $0`; pwd)
FILENAME=${0##*/}
ALGNAME=${FILENAME%.*}
CONFIGFILENAME="${DIR}/config.ini"
TASKNAME=${DIR##*/}

source $CONFIGFILENAME

NUMDEC=0.4466685564173143
SCHEMECODE="[['prune_C4', {'HP11': 3, 'HP10': 1.0, 'HP2': 1.0}], ['prune_C2', {'HP9': 'l2_bn', 'HP8': 0.01, 'HP7': 0.4, 'HP6': 0.7, 'HP2': 0.5, 'HP1': 0.6}]]"

nohup sh scripts/evaluation/evaluation_general.sh $NUMDEC $TASKNAME $ALGNAME "$SCHEMECODE" $1 $REALTASKNAME > ./logs/log_evaluation_${TASKNAME}_${ALGNAME}_${1}.log &
