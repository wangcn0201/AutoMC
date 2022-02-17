#!/bin/sh

DIR=$(cd `dirname $0`; pwd)
FILENAME=${0##*/}
ALGNAME=${FILENAME%.*}
CONFIGFILENAME="${DIR}/config.ini"
TASKNAME=${DIR##*/}

source $CONFIGFILENAME

NUMDEC=0.659399113441623
SCHEMECODE="[['prune_C2', {'HP9': 'l2_bn_param', 'HP8': 0.01, 'HP7': 0.7, 'HP6': 0.9, 'HP2': 0.9, 'HP1': 1.0}], ['prune_C4', {'HP11': 1, 'HP10': 0.4, 'HP2': 0.9}], ['prune_C1', {'HP5': 0.3, 'HP4': 3, 'HP3': 10, 'HP2': 0.3, 'HP1': 0.2}], ['prune_C5', {'HP15': 1, 'HP14': 1.5, 'HP13': 'skew_kur', 'HP12': 'P3', 'HP2': 0.1, 'HP1': 1.0}]]"

nohup sh scripts/evaluation/evaluation_general.sh $NUMDEC $TASKNAME $ALGNAME "$SCHEMECODE" $1 $REALTASKNAME > ./logs/log_evaluation_${TASKNAME}_${ALGNAME}_${1}.log &
