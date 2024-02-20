python mod_main.py --masking-rate 0.035 --tag mr_0.035_00 --device 0
python mod_main.py --masking-rate 0.035 --tag mr_0.035_01 --device 0

python mod_main.py --masking-rate 0.035 --tag mr_0.035_02 --device 1
python mod_main.py --masking-rate 0.035 --tag mr_0.035_03 --device 1

python mod_main.py --masking-rate 0.112 --tag mr_0.112_00 --device 2
python mod_main.py --masking-rate 0.112 --tag mr_0.112_01 --device 2

python mod_main.py --masking-rate 0.112 --tag mr_0.112_02 --device 3
python mod_main.py --masking-rate 0.112 --tag mr_0.112_03 --device 3

python mod_main.py --masking-rate 0.206 --tag mr_0.206_00 --device 4
python mod_main.py --masking-rate 0.206 --tag mr_0.206_01 --device 4

python mod_main.py --masking-rate 0.206 --tag mr_0.206_02 --device 5
python mod_main.py --masking-rate 0.206 --tag mr_0.206_03 --device 5

python mod_main.py --masking-rate 0.331 --tag mr_0.331_00 --device 6
python mod_main.py --masking-rate 0.331 --tag mr_0.331_01 --device 6
python mod_main.py --masking-mode random --tag mr_random_00 --device 6
python mod_main.py --masking-mode random --tag mr_random_01 --device 6

python mod_main.py --masking-rate 0.331 --tag mr_0.331_02 --device 7
python mod_main.py --masking-rate 0.331 --tag mr_0.331_03 --device 7
python mod_main.py --masking-mode random --tag mr_random_02 --device 7
python mod_main.py --masking-mode random --tag mr_random_03 --device 7

python mod_main.py --masking-rate 0.206 --tag mr_0.206_dyn_00 --device 0
python mod_main.py --masking-rate 0.206 --tag mr_0.206_dyn_01 --device 1
python mod_main.py --masking-rate 0.206 --tag mr_0.206_dyn_02 --device 2
python mod_main.py --masking-rate 0.206 --tag mr_0.206_dyn_03 --device 3

python mod_main.py --masking-rate 0.206 --tag ex --device 4

python mod_main.py --masking-mode random --masking-rate 0.035 --tag mr_random_0.035_00 --device 4
python mod_main.py --masking-mode random --masking-rate 0.035 --tag mr_random_0.035_01 --device 5


퇴근 전에 올려두고 가는 것들
python mod_main.py --masking-mode random --masking-rate 0.035 --tag mr_random_0.035_02 --device 6
python mod_main.py --masking-mode random --masking-rate 0.035 --tag mr_random_0.035_03 --device 6

python mod_main.py  --masking-rate 0.206 --tag tps_mr_0.206_dyn_00 --device 7
python mod_main.py  --masking-rate 0.206 --tag tps_mr_0.206_dyn_01 --device 7

python mod_main.py  --masking-rate 0.206 --tag tps_mr_0.206_dyn_02 --device 3
python mod_main.py  --masking-rate 0.206 --tag tps_mr_0.206_dyn_03 --device 3


240216 다음주에 확인해야할 것들...
python mod_main.py --masking-mode random --masking-rate 0.331 --tag mr_random_0.331_00 --device 7
python mod_main.py --masking-mode random --masking-rate 0.331 --tag mr_random_0.331_01 --device 7
python mod_main.py --masking-mode random --masking-rate 0.331 --tag mr_random_0.331_02 --device 7
python mod_main.py --masking-mode random --masking-rate 0.331 --tag mr_random_0.331_03 --device 7

# longterm training
python mod_main.py --masking-mode random --masking-rate 0.035 --epochs 90 --check-val 5 --lr 1e-5 --tag lt_random_0.035_00 --device 1
python mod_main.py --masking-mode random --masking-rate 0.112 --epochs 90 --check-val 5 --lr 1e-5 --tag lt_random_0.112_00 --device 2
python mod_main.py --masking-rate 0.035 --epochs 90 --check-val 5 --lr 1e-5 --tag lt_0.035_00 --device 3
python mod_main.py --masking-rate 0.112 --epochs 90 --check-val 5 --lr 1e-5 --tag lt_0.112_00 --device 4

# unbalanced validation
python mod_main.py --masking-rate 0.331 --val-masking-rate 0.206 --epochs 10 --check-val 1 --lr 6e-6 --tag unbal_mr_0.331_val_0.206_00 --device 5
python mod_main.py --masking-rate 0.331 --val-masking-rate 0.206 --epochs 10 --check-val 1 --lr 6e-6 --tag unbal_mr_0.331_val_0.206_01 --device 5
python mod_main.py --masking-rate 0.331 --val-masking-rate 0.206 --epochs 10 --check-val 1 --lr 6e-6 --tag unbal_mr_0.331_val_0.206_02 --device 5
python mod_main.py --masking-rate 0.331 --val-masking-rate 0.206 --epochs 10 --check-val 1 --lr 6e-6 --tag unbal_mr_0.331_val_0.206_03 --device 5

python mod_main.py --masking-rate 0.331 --val-masking-rate 0.206 --epochs 10 --check-val 1 --lr 1e-5 --tag unbal_mr_0.331_val_0.206_lr_1e-5_00 --device 6
python mod_main.py --masking-rate 0.331 --val-masking-rate 0.206 --epochs 10 --check-val 1 --lr 1e-5 --tag unbal_mr_0.331_val_0.206_lr_1e-5_01 --device 6
python mod_main.py --masking-rate 0.331 --val-masking-rate 0.206 --epochs 10 --check-val 1 --lr 1e-5 --tag unbal_mr_0.331_val_0.206_lr_1e-5_02 --device 6
python mod_main.py --masking-rate 0.331 --val-masking-rate 0.206 --epochs 10 --check-val 1 --lr 1e-5 --tag unbal_mr_0.331_val_0.206_lr_1e-5_03 --device 6

python mod_main.py --masking-mode random --masking-rate 0.331 --val-masking-rate 0.206 --epochs 10 --check-val 1 --lr 6e-6 --tag unbal_random_0.331_val_0.206_00 --device 0
python mod_main.py --masking-mode random --masking-rate 0.331 --val-masking-rate 0.206 --epochs 10 --check-val 1 --lr 6e-6 --tag unbal_random_0.331_val_0.206_01 --device 0
python mod_main.py --masking-mode random --masking-rate 0.331 --val-masking-rate 0.206 --epochs 10 --check-val 1 --lr 6e-6 --tag unbal_random_0.331_val_0.206_02 --device 0
python mod_main.py --masking-mode random --masking-rate 0.331 --val-masking-rate 0.206 --epochs 10 --check-val 1 --lr 6e-6 --tag unbal_random_0.331_val_0.206_03 --device 0


20240219
# 주석 실험 폐기
# python mod_main.py --masking-mode random --masking-rate 0.331 --val-masking-rate 0.206 --epochs 10 --check-val 1 --lr 1e-5 --tag unbal_random_0.331_val_0.206_lr_1e-5_00 --device 5
# python mod_main.py --masking-mode random --masking-rate 0.331 --val-masking-rate 0.206 --epochs 10 --check-val 1 --lr 1e-5 --tag unbal_random_0.331_val_0.206_lr_1e-5_01 --device 5
# python mod_main.py --masking-mode random --masking-rate 0.331 --val-masking-rate 0.206 --epochs 10 --check-val 1 --lr 1e-5 --tag unbal_random_0.331_val_0.206_lr_1e-5_02 --device 6
# python mod_main.py --masking-mode random --masking-rate 0.331 --val-masking-rate 0.206 --epochs 10 --check-val 1 --lr 1e-5 --tag unbal_random_0.331_val_0.206_lr_1e-5_03 --device 6

# python mod_main.py --masking-rate 0.206 --epochs 10 --check-val 1 --lr 1e-5 --tag re_0.206_lr_1e-5_00 --device 1
# python mod_main.py --masking-rate 0.206 --epochs 10 --check-val 1 --lr 1e-5 --tag re_0.206_lr_1e-5_01 --device 1
# python mod_main.py --masking-rate 0.206 --epochs 10 --check-val 1 --lr 1e-5 --tag re_0.206_lr_1e-5_02 --device 1
# python mod_main.py --masking-rate 0.206 --epochs 10 --check-val 1 --lr 1e-5 --tag re_0.206_lr_1e-5_03 --device 1

# python mod_main.py --masking-rate 0.206 --epochs 10 --check-val 1 --lr 6e-6 --tag re_0.206_lr_6e-6_00 --device 2
# python mod_main.py --masking-rate 0.206 --epochs 10 --check-val 1 --lr 6e-6 --tag re_0.206_lr_6e-6_01 --device 2
# python mod_main.py --masking-rate 0.206 --epochs 10 --check-val 1 --lr 6e-6 --tag re_0.206_lr_6e-6_02 --device 2
# python mod_main.py --masking-rate 0.206 --epochs 10 --check-val 1 --lr 6e-6 --tag re_0.206_lr_6e-6_03 --device 2

# python mod_main.py --masking-rate 0.112 --epochs 10 --check-val 1 --lr 1e-5 --tag re_0.112_lr_1e-5_00 --device 7
# python mod_main.py --masking-rate 0.112 --epochs 10 --check-val 1 --lr 1e-5 --tag re_0.112_lr_1e-5_01 --device 7
# python mod_main.py --masking-rate 0.112 --epochs 10 --check-val 1 --lr 1e-5 --tag re_0.112_lr_1e-5_02 --device 7
# python mod_main.py --masking-rate 0.112 --epochs 10 --check-val 1 --lr 1e-5 --tag re_0.112_lr_1e-5_03 --device 7

# 큰일 나버렸다,,
# python mod_main.py --masking-rate 0.035 --epochs 90 --check-val 5 --lr 1e-5 --tag re_lt_0.035_00 --device 3
# python mod_main.py --masking-rate 0.112 --epochs 90 --check-val 5 --lr 1e-5 --tag re_lt_0.112_00 --device 4

실험 recover.
python mod_main.py --masking-rate 0.206 --epochs 10 --check-val 1 --lr 6e-6 --tag re_0.206_lr_6e-6_00 --device 0
python mod_main.py --masking-rate 0.206 --epochs 10 --check-val 1 --lr 6e-6 --tag re_0.206_lr_6e-6_01 --device 0
python mod_main.py --masking-rate 0.206 --epochs 10 --check-val 1 --lr 6e-6 --tag re_0.206_lr_6e-6_02 --device 0
python mod_main.py --masking-rate 0.206 --epochs 10 --check-val 1 --lr 6e-6 --tag re_0.206_lr_6e-6_03 --device 0

python mod_main.py --masking-rate 0.331 --epochs 10 --check-val 1 --lr 6e-6 --tag re_0.331_lr_6e-6_00 --device 1
python mod_main.py --masking-rate 0.331 --epochs 10 --check-val 1 --lr 6e-6 --tag re_0.331_lr_6e-6_01 --device 1
python mod_main.py --masking-rate 0.331 --epochs 10 --check-val 1 --lr 6e-6 --tag re_0.331_lr_6e-6_02 --device 1
python mod_main.py --masking-rate 0.331 --epochs 10 --check-val 1 --lr 6e-6 --tag re_0.331_lr_6e-6_03 --device 1

python mod_main.py --masking-rate 0.112 --epochs 10 --check-val 1 --lr 6e-6 --tag re_0.112_lr_6e-6_00 --device 2
python mod_main.py --masking-rate 0.112 --epochs 10 --check-val 1 --lr 6e-6 --tag re_0.112_lr_6e-6_01 --device 2
python mod_main.py --masking-rate 0.112 --epochs 10 --check-val 1 --lr 6e-6 --tag re_0.112_lr_6e-6_02 --device 2
python mod_main.py --masking-rate 0.112 --epochs 10 --check-val 1 --lr 6e-6 --tag re_0.112_lr_6e-6_03 --device 2

python mod_main.py --masking-rate 0.035 --epochs 10 --check-val 1 --lr 6e-6 --tag re_0.035_lr_6e-6_00 --device 3
python mod_main.py --masking-rate 0.035 --epochs 10 --check-val 1 --lr 6e-6 --tag re_0.035_lr_6e-6_01 --device 3
python mod_main.py --masking-rate 0.035 --epochs 10 --check-val 1 --lr 6e-6 --tag re_0.035_lr_6e-6_02 --device 3
python mod_main.py --masking-rate 0.035 --epochs 10 --check-val 1 --lr 6e-6 --tag re_0.035_lr_6e-6_03 --device 3

python mod_main.py --masking-mode random --masking-rate 0.206 --epochs 10 --check-val 1 --lr 1e-5 --tag mr_random_0.206_lr_1e-5_00 --device 4
python mod_main.py --masking-mode random --masking-rate 0.206 --epochs 10 --check-val 1 --lr 1e-5 --tag mr_random_0.206_lr_1e-5_01 --device 4
python mod_main.py --masking-mode random --masking-rate 0.206 --epochs 10 --check-val 1 --lr 1e-5 --tag mr_random_0.206_lr_1e-5_02 --device 4
python mod_main.py --masking-mode random --masking-rate 0.206 --epochs 10 --check-val 1 --lr 1e-5 --tag mr_random_0.206_lr_1e-5_03 --device 4

python mod_main.py --masking-mode random --masking-rate 0.331 --epochs 10 --check-val 1 --lr 1e-5 --tag mr_random_0.331_lr_1e-5_00 --device 5
python mod_main.py --masking-mode random --masking-rate 0.331 --epochs 10 --check-val 1 --lr 1e-5 --tag mr_random_0.331_lr_1e-5_01 --device 5
python mod_main.py --masking-mode random --masking-rate 0.331 --epochs 10 --check-val 1 --lr 1e-5 --tag mr_random_0.331_lr_1e-5_02 --device 5
python mod_main.py --masking-mode random --masking-rate 0.331 --epochs 10 --check-val 1 --lr 1e-5 --tag mr_random_0.331_lr_1e-5_03 --device 5

python mod_main.py --masking-mode random --masking-rate 0.112 --epochs 10 --check-val 1 --lr 1e-5 --tag mr_random_0.112_lr_1e-5_00 --device 6
python mod_main.py --masking-mode random --masking-rate 0.112 --epochs 10 --check-val 1 --lr 1e-5 --tag mr_random_0.112_lr_1e-5_01 --device 6
python mod_main.py --masking-mode random --masking-rate 0.112 --epochs 10 --check-val 1 --lr 1e-5 --tag mr_random_0.112_lr_1e-5_02 --device 6
python mod_main.py --masking-mode random --masking-rate 0.112 --epochs 10 --check-val 1 --lr 1e-5 --tag mr_random_0.112_lr_1e-5_03 --device 6

python mod_main.py --masking-mode random --masking-rate 0.035 --epochs 10 --check-val 1 --lr 1e-5 --tag mr_random_0.035_lr_1e-5_00 --device 7
python mod_main.py --masking-mode random --masking-rate 0.035 --epochs 10 --check-val 1 --lr 1e-5 --tag mr_random_0.035_lr_1e-5_01 --device 7
python mod_main.py --masking-mode random --masking-rate 0.035 --epochs 10 --check-val 1 --lr 1e-5 --tag mr_random_0.035_lr_1e-5_02 --device 7
python mod_main.py --masking-mode random --masking-rate 0.035 --epochs 10 --check-val 1 --lr 1e-5 --tag mr_random_0.035_lr_1e-5_03 --device 7



