#!/bin/sh

DIR=$(cd `dirname $0`; pwd)
FILENAME=${0##*/}
ALGNAME=${FILENAME%.*}
CONFIGFILENAME="${DIR}/config.ini"
TASKNAME=${DIR##*/}

source $CONFIGFILENAME

NUMDEC=0.400876147
SCHEMECODE="[['prune_C1', {'HP5': 0.5, 'HP4': 6, 'HP3': 10, 'HP2': 0.7, 'HP1': 1.0}], ['prune_C2', {'HP9': 'l1_weight', 'HP8': 0.01, 'HP7': 0.5, 'HP6': 0.7, 'HP2': 0.7, 'HP1': 0.2}], ['prune_C5', {'HP15': 1, 'HP14': 0.4, 'HP13': 'l1norm', 'HP12': 'P3', 'HP2': 0.9, 'HP1': 1.5}]]"

nohup sh scripts/evaluation/evaluation_general.sh $NUMDEC $TASKNAME $ALGNAME "$SCHEMECODE" $1 $REALTASKNAME > ./logs/log_evaluation_${TASKNAME}_${ALGNAME}_${1}.log &
