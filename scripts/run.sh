#!/bin/bash
while true; do
	stat1=$(gpustat | awk '{print $11}' | sed -n '2p')
	stat2=$(gpustat | awk '{print $11}' | sed -n '3p')
	stat3=$(gpustat | awk '{print $11}' | sed -n '4p')
	stat4=$(gpustat | awk '{print $11}' | sed -n '5p')
	stat5=$(gpustat | awk '{print $11}' | sed -n '6p')
	stat6=$(gpustat | awk '{print $11}' | sed -n '7p')
	stat7=$(gpustat | awk '{print $11}' | sed -n '8p')
	stat8=$(gpustat | awk '{print $11}' | sed -n '9p')
	echo 'GPU显存占用情况:' $stat1 $stat2 $stat3 $stat4 $stat5 $stat6 $stat7 $stat8
	stat_arr=($stat1 $stat2 $stat3 $stat4 $stat5 $stat6 $stat7 $stat8)
	gpu_available=0
	gpu_available_index_arr=()
	# 得到空闲GPU的数量和对应的序号
	for i in ${!stat_arr[@]}; do
		# 如果显存占用小于100M，继续
		if [ "${stat_arr[$i]}" -lt 100 ]; then
			gpu_available=$((gpu_available + 1))
			gpu_available_index_arr[${#gpu_available_index_arr[@]}]=$i
		fi
	done
	echo '-可用GPU数:'$gpu_available', 第'${gpu_available_index_arr[@]}'块GPU可用'
	sleep 10
	if [ $stat1 -lt 100 ]; then
		echo 'start running my code...'
		bash scripts/train_gpus.sh
		break
	fi
	sleep 30
done
