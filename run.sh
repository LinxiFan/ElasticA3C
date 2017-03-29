#!/bin/bash
case $1 in 
    p) game=Pong ;;
    b) game=Breakout ;;
    be) game=BeamRider ;;
    se) game=Seaquest ;;
    sp) game=SpaceInvaders ;;
    q) game=Qbert ;;
    *) echo 'Bad arg, select game from: p, b, be, se, sp, q'
       exit 1 ;;
esac
shift    

if [ -z $1 ]; then
    arg_suffix=''
    dir_suffix=''
else
    arg_suffix="-s $1"
    dir_suffix="-$1"
fi

if [ $# -le 1 ]; then
    procs=32
else
    procs=$2
fi

a3c -e $game $arg_suffix -n $procs #--cuda-devices 5
echo "$1 source code copied"
