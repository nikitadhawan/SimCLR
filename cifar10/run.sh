timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=../ nohup python train_classifier.py
>> log_train_classifier_${timestamp}.txt 2>&1 &
echo log_train_classifier_${timestamp}.txt


CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=../ python train_classifier.py >> log_train_classifier_${timestamp}.txt 2>&1 &

