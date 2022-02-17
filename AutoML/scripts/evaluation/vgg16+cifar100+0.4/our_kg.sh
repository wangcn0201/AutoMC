#!/bin/sh

DIR=$(cd `dirname $0`; pwd)
FILENAME=${0##*/}
ALGNAME=${FILENAME%.*}
CONFIGFILENAME="${DIR}/config.ini"
TASKNAME=${DIR##*/}

source $CONFIGFILENAME

NUMDEC=0.39016142477452276
SCHEMECODE="[['prune_C2', {'HP9': 'l2_bn_param', 'HP8': 0.01, 'HP7': 0.5, 'HP6': 0.9, 'HP2': 0.1, 'HP1': 0.2}], ['prune_C2', {'HP9': 'l2_bn_param', 'HP8': 0.01, 'HP7': 0.5, 'HP6': 0.9, 'HP2': 1.0, 'HP1': 1.0}], ['prune_C2', {'HP9': 'l2_bn', 'HP8': 0.01, 'HP7': 0.4, 'HP6': 0.9, 'HP2': 0.1, 'HP1': 0.6}], ['prune_C2', {'HP9': 'l1_weight', 'HP8': 0.01, 'HP7': 0.5, 'HP6': 0.9, 'HP2': 0.1, 'HP1': 1.0}]]"

nohup sh scripts/evaluation/evaluation_general.sh $NUMDEC $TASKNAME $ALGNAME "$SCHEMECODE" $1 $REALTASKNAME > ./logs/log_evaluation_${TASKNAME}_${ALGNAME}_${1}.log &
