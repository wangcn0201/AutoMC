#!/bin/sh

DIR=$(cd `dirname $0`; pwd)
FILENAME=${0##*/}
ALGNAME=${FILENAME%.*}
CONFIGFILENAME="${DIR}/config.ini"
TASKNAME=${DIR##*/}

source $CONFIGFILENAME

NUMDEC=0.42084022699749757
SCHEMECODE="[['prune_C2', {'HP9': 'l1_weight', 'HP8': 0.01, 'HP7': 0.7, 'HP6': 0.7, 'HP2': 0.5, 'HP1': 1.0}], ['prune_C5', {'HP15': 1, 'HP14': 1.5, 'HP13': 'skew_kur', 'HP12': 'P2', 'HP2': 0.9, 'HP1': 0.4}]]"

nohup sh scripts/evaluation/evaluation_general.sh $NUMDEC $TASKNAME $ALGNAME "$SCHEMECODE" $1 $REALTASKNAME > ./logs/log_evaluation_${TASKNAME}_${ALGNAME}_${1}.log &
