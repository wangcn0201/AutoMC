#!/bin/sh

DIR=$(cd `dirname $0`; pwd)
FILENAME=${0##*/}
ALGNAME=${FILENAME%.*}
CONFIGFILENAME="${DIR}/config.ini"
TASKNAME=${DIR##*/}

source $CONFIGFILENAME

NUMDEC=0.804403383
SCHEMECODE="[['prune_C1', {'HP5': 0.5, 'HP4': 10, 'HP3': 6, 'HP2': 1.0, 'HP1': 0.8}], ['prune_C4', {'HP11': 5, 'HP10': 0.8, 'HP2': 1.0}], ['prune_C3', {'HP6': 0.7, 'HP2': 1.0, 'HP1': 1.5}], ['prune_C4', {'HP11': 1, 'HP10': 0.8, 'HP2': 0.5}]]"

nohup sh scripts/evaluation/evaluation_general.sh $NUMDEC $TASKNAME $ALGNAME "$SCHEMECODE" $1 $REALTASKNAME > ./logs/log_evaluation_${TASKNAME}_${ALGNAME}_${1}.log &
