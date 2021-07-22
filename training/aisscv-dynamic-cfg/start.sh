#!/bin/bash
CURRENT_PATH="/"
COLOR=false
OCCLUDE=false
MAX_BATCHES=3000
FOLDS=1
NAME=$(date +%Y%m%d_%H%M%S)
NBR_AUGMENTATIONS=10
DARKNET="/darknet"
TRANSFORM=1
INCL_NO_LABEL=false
IS_FINAL=false

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
        -final|--is_final)
        IS_FINAL=true
        shift # Remove --initialize from processing
        ;;
        -b=*|--max_batches=*)
        MAX_BATCHES="${arg#*=}"
        shift # Remove argument from processing
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
COMMANDS="-n=$NAME -o /aisscv/model -f=$FOLDS -max_batches=$MAX_BATCHES -nbr_augment=$NBR_AUGMENTATIONS -darknet=$DARKNET -t=$TRANSFORM"

(cd "$CURRENT_PATH" && git clone https://github.com/pineappledafruitdude/AISSCV.git aisscv)

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

if $IS_FINAL
then
  COMMANDS="$COMMANDS -is_final"
fi

echo "Running script with: $COMMANDS"
(cd "$CURRENT_PATH"aisscv/preprocessing/ && python3.8 run_train.py $COMMANDS)

