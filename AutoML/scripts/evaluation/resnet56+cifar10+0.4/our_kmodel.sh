#!/bin/sh

DIR=$(cd `dirname $0`; pwd)
FILENAME=${0##*/}
ALGNAME=${FILENAME%.*}
CONFIGFILENAME="${DIR}/config.ini"
TASKNAME=${DIR##*/}

source $CONFIGFILENAME

NUMDEC=0.313644448
SCHEMECODE="[['prune_C1', {'HP5': 0.05, 'HP4': 3, 'HP3': 10, 'HP2': 0.9, 'HP1': 1.5}], ['prune_C2', {'HP9': 'l1_weight', 'HP8': 0.01, 'HP7': 0.7, 'HP6': 0.9, 'HP2': 0.7, 'HP1': 1.5}], ['prune_C4', {'HP11': 3, 'HP10': 0.6, 'HP2': 0.1}], ['prune_C2', {'HP9': 'l2_bn_param', 'HP8': 0.01, 'HP7': 0.4, 'HP6': 0.7, 'HP2': 0.1, 'HP1': 0.2}]]"

nohup sh scripts/evaluation/evaluation_general.sh $NUMDEC $TASKNAME $ALGNAME "$SCHEMECODE" $1 $REALTASKNAME > ./logs/log_evaluation_${TASKNAME}_${ALGNAME}_${1}.log &
