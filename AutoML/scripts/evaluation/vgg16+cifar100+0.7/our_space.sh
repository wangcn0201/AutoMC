#!/bin/sh

DIR=$(cd `dirname $0`; pwd)
FILENAME=${0##*/}
ALGNAME=${FILENAME%.*}
CONFIGFILENAME="${DIR}/config.ini"
TASKNAME=${DIR##*/}

source $CONFIGFILENAME

NUMDEC=0.7502720916047151
SCHEMECODE="[['prune_C2', {'HP9': 'l2_bn', 'HP8': 0.01, 'HP7': 0.6, 'HP6': 0.7, 'HP2': 1.0, 'HP1': 1.0}], ['prune_C2', {'HP9': 'l2_weight', 'HP8': 0.01, 'HP7': 0.6, 'HP6': 0.7, 'HP2': 0.7, 'HP1': 0.4}], ['prune_C2', {'HP9': 'l2_bn', 'HP8': 0.01, 'HP7': 0.5, 'HP6': 0.9, 'HP2': 0.1, 'HP1': 0.8}], ['prune_C2', {'HP9': 'l1_weight', 'HP8': 0.01, 'HP7': 0.7, 'HP6': 0.9, 'HP2': 0.7, 'HP1': 0.6}]]"

nohup sh scripts/evaluation/evaluation_general.sh $NUMDEC $TASKNAME $ALGNAME "$SCHEMECODE" $1 $REALTASKNAME > ./logs/log_evaluation_${TASKNAME}_${ALGNAME}_${1}.log &
