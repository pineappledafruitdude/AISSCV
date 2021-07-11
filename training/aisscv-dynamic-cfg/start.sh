#!/bin/bash
CURRENT_PATH="/"
COLOR=false
OCCLUDE=false
BATCH_SIZE=3000
FOLDS=1
NAME=$(date +%Y%m%d_%H%M%S)
NBR_AUGMENTATIONS=10
DARKNET="/darknet"
TRANSFORM=1
INCL_NO_LABEL=false

for arg in "$@"
do
    case $arg in
        -path=*|--path=*)
        CURRENT_PATH="${arg#*=}"
        shift # Remove argument value from processing
        ;;
        -darknet=*|--darknet_path=*)
        DARKNET="${arg#*=}"
        shift # Remove argument value from processing
        ;;
        -c|--color)
        COLOR=true
        shift # Remove --initialize from processing
        ;;
        -occl|--occlude)
        OCCLUDE=true
        shift # Remove --initialize from processing
        ;;
        -no_label|--incl_no_label)
        INCL_NO_LABEL=true
        shift # Remove --initialize from processing
        ;;
        -b=*|--batch_size=*)
        BATCH_SIZE="${arg#*=}"
        shift # Remove argument value from processing
        ;;
        -f=*|--folds=*)
        FOLDS="${arg#*=}"
        shift # Remove argument name from processing
        ;;
        -t=*|--transform=*)
        TRANSFORM="${arg#*=}"
        shift # Remove argument name from processing
        ;;
        -n=*|--name=*)      
        NAME="${arg#*=}"
        shift # Remove argument name from processing
        ;;
        -a=*|--augmentations=*)
        NBR_AUGMENTATIONS="${arg#*=}"
        shift # Remove argument name from processing
        ;;
    esac
done
COMMANDS="-n=$NAME -o /aisscv/model -f=$FOLDS -batch_size=$BATCH_SIZE -nbr_augment=$NBR_AUGMENTATIONS -darknet=$DARKNET -t=$TRANSFORM"

#(cd "$CURRENT_PATH" && git clone https://github.com/pineappledafruitdude/AISSCV.git aisscv)

if $COLOR
then
  COMMANDS="$COMMANDS -c"
fi

if $OCCLUDE
then
  COMMANDS="$COMMANDS -occl"
fi

if $INCL_NO_LABEL
then
  COMMANDS="$COMMANDS -incl_no_label"
fi

echo "Running script with: $COMMANDS"
(cd "$CURRENT_PATH"aisscv/preprocessing/ && python3.8 run_train.py $COMMANDS)

