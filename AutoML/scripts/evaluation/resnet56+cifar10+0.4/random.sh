#!/bin/sh

DIR=$(cd `dirname $0`; pwd)
FILENAME=${0##*/}
ALGNAME=${FILENAME%.*}
CONFIGFILENAME="${DIR}/config.ini"
TASKNAME=${DIR##*/}

source $CONFIGFILENAME

NUMDEC=0.394025280817796
SCHEMECODE="[['prune_C3', {'HP1': 0.2, 'HP2': 0.5, 'HP6': 0.7}], ['prune_C3', {'HP1': 0.4, 'HP2': 1.0, 'HP6': 0.7}], ['prune_C4', {'HP2': 1.0, 'HP10': 1.5, 'HP11': 1}], ['prune_C4', {'HP2': 0.3, 'HP10': 1.5, 'HP11': 3}]]"

nohup sh scripts/evaluation/evaluation_general.sh $NUMDEC $TASKNAME $ALGNAME "$SCHEMECODE" $1 $REALTASKNAME > ./logs/log_evaluation_${TASKNAME}_${ALGNAME}_${1}.log &
