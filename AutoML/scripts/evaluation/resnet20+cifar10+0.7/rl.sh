#!/bin/sh

DIR=$(cd `dirname $0`; pwd)
FILENAME=${0##*/}
ALGNAME=${FILENAME%.*}
CONFIGFILENAME="${DIR}/config.ini"
TASKNAME=${DIR##*/}

source $CONFIGFILENAME

NUMDEC=0.7125456
SCHEMECODE="[['prune_C3', {'HP6': 0.7, 'HP2': 1.0, 'HP1': 0.2}], ['prune_C5', {'HP15': 1, 'HP14': 0.6, 'HP13': 'l1norm', 'HP12': 'P3', 'HP2': 0.7, 'HP1': 1.5}]]"

nohup sh scripts/evaluation/evaluation_general.sh $NUMDEC $TASKNAME $ALGNAME "$SCHEMECODE" $1 $REALTASKNAME > ./logs/log_evaluation_${TASKNAME}_${ALGNAME}_${1}.log &
