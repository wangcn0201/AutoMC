#!/bin/sh

DIR=$(cd `dirname $0`; pwd)
FILENAME=${0##*/}
ALGNAME=${FILENAME%.*}
CONFIGFILENAME="${DIR}/config.ini"
TASKNAME=${DIR##*/}

source $CONFIGFILENAME

NUMDEC=0.321584392
SCHEMECODE="[['prune_C2', {'HP9': 'l2_bn', 'HP8': 0.01, 'HP7': 0.6, 'HP6': 0.7, 'HP2': 0.3, 'HP1': 0.4}], ['prune_C1', {'HP5': 0.3, 'HP4': 10, 'HP3': 6, 'HP2': 0.7, 'HP1': 1.0}], ['prune_C1', {'HP5': 0.3, 'HP4': 10, 'HP3': 10, 'HP2': 1.0, 'HP1': 0.6}], ['prune_C5', {'HP15': 1, 'HP14': 1.5, 'HP13': 'l1norm', 'HP12': 'P3', 'HP2': 0.5, 'HP1': 0.6}]]"

nohup sh scripts/evaluation/evaluation_general.sh $NUMDEC $TASKNAME $ALGNAME "$SCHEMECODE" $1 $REALTASKNAME > ./logs/log_evaluation_${TASKNAME}_${ALGNAME}_${1}.log &
