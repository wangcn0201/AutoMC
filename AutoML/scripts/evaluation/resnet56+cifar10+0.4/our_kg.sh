#!/bin/sh

DIR=$(cd `dirname $0`; pwd)
FILENAME=${0##*/}
ALGNAME=${FILENAME%.*}
CONFIGFILENAME="${DIR}/config.ini"
TASKNAME=${DIR##*/}

source $CONFIGFILENAME

NUMDEC=0.696251618
SCHEMECODE="[['prune_C1', {'HP5': 0.5, 'HP4': 10, 'HP3': 6, 'HP2': 1.0, 'HP1': 0.8}], ['prune_C4', {'HP11': 5, 'HP10': 0.8, 'HP2': 1.0}], ['prune_C2', {'HP9': 'l2_bn_param', 'HP8': 0.01, 'HP7': 0.7, 'HP6': 0.7, 'HP2': 0.1, 'HP1': 1.5}], ['prune_C4', {'HP11': 3, 'HP10': 0.6, 'HP2': 0.1}], ['prune_C5', {'HP15': 1, 'HP14': 0.6, 'HP13': 'l1norm', 'HP12': 'P2', 'HP2': 0.5, 'HP1': 0.2}]]"

nohup sh scripts/evaluation/evaluation_general.sh $NUMDEC $TASKNAME $ALGNAME "$SCHEMECODE" $1 $REALTASKNAME > ./logs/log_evaluation_${TASKNAME}_${ALGNAME}_${1}.log &
