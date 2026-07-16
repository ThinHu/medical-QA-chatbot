$env:PYTHONIOENCODING="utf-8"
$OutputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "Bắt đầu huấn luyện cục bộ trên GPU..."
python src/main.py `
    --model_type vimq_model `
    --model_dir ViMQ_Model `
    --data_dir data `
    --seed 100 `
    --do_train `
    --do_eval `
    --train_batch_size 16 `
    --save_steps 50 `
    --logging_steps 50 `
    --num_train_epochs 3 `
    --num_iteration 2 `
    --tuning_metric f1_score `
    --gpu_id 0 `
    --iternoise 1 `
    --omega 0 `
    --threshold_iou 0.9 `
    --lamda 3

Write-Host "Hoàn tất!"
