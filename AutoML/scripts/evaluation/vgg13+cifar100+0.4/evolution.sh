#!/bin/sh

DIR=$(cd `dirname $0`; pwd)
FILENAME=${0##*/}
ALGNAME=${FILENAME%.*}
CONFIGFILENAME="${DIR}/config.ini"
TASKNAME=${DIR##*/}

source $CONFIGFILENAME

NUMDEC=0.4517093579748154
SCHEMECODE="[['prune_C2', {'HP9': 'l2_bn_param', 'HP8': 0.01, 'HP7': 0.4, 'HP6': 0.7, 'HP2': 0.5, 'HP1': 0.8}], ['prune_C5', {'HP15': 3, 'HP14': 1.5, 'HP13': 'k34', 'HP12': 'P1', 'HP2': 1.0, 'HP1': 1.0}]]"

nohup sh scripts/evaluation/evaluation_general.sh $NUMDEC $TASKNAME $ALGNAME "$SCHEMECODE" $1 $REALTASKNAME > ./logs/log_evaluation_${TASKNAME}_${ALGNAME}_${1}.log &
