python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_config.json" --model-name "self_impl.DCdetector" --model-hyper-params '{"anomaly_ratio": 20.0, "batch_size": 64, "num_epochs": 3, "patch_size": [3, 5], "win_size": 90}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "score/DCdetector"

