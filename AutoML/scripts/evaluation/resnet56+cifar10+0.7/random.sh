#!/bin/sh

DIR=$(cd `dirname $0`; pwd)
FILENAME=${0##*/}
ALGNAME=${FILENAME%.*}
CONFIGFILENAME="${DIR}/config.ini"
TASKNAME=${DIR##*/}

source $CONFIGFILENAME

NUMDEC=0.814989975
SCHEMECODE="[['prune_C2', {'HP1': 0.2, 'HP2': 1.0, 'HP6': 0.9, 'HP7': 0.6, 'HP8': 0.01, 'HP9': 'l2_weight'}], ['prune_C4', {'HP2': 0.1, 'HP10': 0.6, 'HP11': 3}], ['prune_C4', {'HP2': 0.3, 'HP10': 0.8, 'HP11': 3}], ['prune_C3', {'HP1': 0.6, 'HP2': 0.3, 'HP6': 0.9}]]"

nohup sh scripts/evaluation/evaluation_general.sh $NUMDEC $TASKNAME $ALGNAME "$SCHEMECODE" $1 $REALTASKNAME > ./logs/log_evaluation_${TASKNAME}_${ALGNAME}_${1}.log &
