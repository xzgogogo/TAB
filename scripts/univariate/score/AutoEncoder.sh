python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_config.json" --model-name "merlion.AutoEncoder" --model-hyper-params '{"hidden_size": 5, "lr": 5e-05, "num_epochs": 10}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "score/AutoEncoder"

