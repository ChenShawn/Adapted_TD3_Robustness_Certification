#!/bin/bash
set -x

rm train/* -rf
rm logs/*

for i in `seq 0 9`
do
	echo "Round $i -----------------------------------------"

	python main.py \
	--policy "td3" \
	--env "HalfCheetah-v2" \
	--seed $i \
	--start-timesteps 10000 &

	python main.py \
	--policy "td3" \
	--env "Hopper-v2" \
	--seed $i \
	--start-timesteps 1000 &

	python main.py \
	--policy "td3" \
	--env "Walker2d-v3" \
	--seed $i \
	--start-timesteps 1000 &

	python main.py \
	--policy "td3" \
	--env "Ant-v2" \
	--seed $i \
	--start-timesteps 10000 &

	wait
	python test.py -e "Hopper-v2" -n 100 --train-seed $i --test-seed $i >> ./logs/evaluate.log &
	python test.py -e "HalfCheetah-v2" -n 100 --train-seed $i --test-seed $i >> ./logs/evaluate.log &
	python test.py -e "Walker2d-v3" -n 100 --train-seed $i --test-seed $i >> ./logs/evaluate.log &
	python test.py -e "Ant-v2" -n 100 --train-seed $i --test-seed $i >> ./logs/evaluate.log &

	python main.py \
	--policy "td3" \
	--env "Humanoid-v2" \
	--seed $i \
	--max-timesteps 3000000 \
	--batch-size 512 \
	--start-timesteps 10000 &

	python main.py \
	--policy "td3" \
	--env "LunarLanderContinuous-v2" \
	--seed $i \
	--start-timesteps 1000 &

	python main.py \
	--policy "td3" \
	--env "BipedalWalker-v2" \
	--batch-size 1024 \
	--seed $i \
	--start-timesteps 1000 &

	python main.py \
	--policy "td3" \
	--env "BipedalWalkerHardcore-v2" \
	--batch-size 1024 \
	--seed $i \
	--start-timesteps 1000 &

	wait
	python test.py -e "LunarLanderContinuous-v2" -n 100 --train-seed $i --test-seed $i >> ./logs/evaluate.log &
	python test.py -e "Humanoid-v2" -n 100 --train-seed $i --test-seed $i >> ./logs/evaluate.log &
	python test.py -e "BipedalWalker-v2" -n 100 --train-seed $i --test-seed $i >> ./logs/evaluate.log &
	python test.py -e "BipedalWalkerHardcore-v2" -n 100 --train-seed $i --test-seed $i >> ./logs/evaluate.log &
done