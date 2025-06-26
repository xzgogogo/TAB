python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_config.json" --model-name "merlion.ArimaDetector" --model-hyper-params '{"enable_threshold": 0, "max_forecast_steps": 24}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "score/ArimaDetector"

