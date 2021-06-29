[comment]: <> (<div align="center">)

[comment]: <> (  <h3> python3 diffmpm.py </h3>)

[comment]: <> (  <img  width="400px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/diffmpm80.gif">)

[comment]: <> (</div>        )


# Train


# Test
`python3 main.py --env-name "PongNoFrameskip-v4" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 1000 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --save-interval 50 --save-dir ./trained_models --log-dir ../RL_results --validate 10`

