#!/bin/bash -x


export backdoor_dir='/home/iis519409/github/test/backdoor'
export reprogram_dir='/home/iis519409/github/test/Reprogramming'


export num_sample=50000
export num_shadow=1
export poison_rate=0.200
export dataset=cifar10
export attack=blend
export model=target_1616_test
export exp_name="${model}_${dataset}_${num_sample}_${attack}"



create_poisoned_set() {
  local poison_type=$1
  local poison_rate=$2
  python create_poisoned_set_16.py -dataset="$dataset" -poison_type="$poison_type" -poison_rate="$poison_rate" -exp_name="$exp_name" -num_sample="$num_sample"
}


train_on_poisoned_set() {
  local poison_type=$1
  local poison_rate=$2
  for i in $(seq 1 $num_shadow); do
    python train_on_poisoned_set.py -dataset="$dataset" -poison_type="$poison_type" -poison_rate="$poison_rate" -exp_name="$exp_name"
  done
}

# 定?一?函?用于复制和重命名文件
copy_and_rename() {
  local source_base=$1
  local destination_base=$2
  local prefix=$3
  mkdir -p "$destination_base"
  local counter=0
  for source_file in ${source_base}/full_base_aug_seed=2333.pt*; do
    new_name=$(printf "${prefix}_%04d.pt" "$counter")
    cp "$source_file" "${destination_base}${new_name}"
    ((counter++))
  done
  echo "Files copied and renamed successfully."
}


# ?建?据集
cd "${backdoor_dir}"
#create_poisoned_set none 0.0
create_poisoned_set "${attack}" "${poison_rate}"

# # ??模型
train_on_poisoned_set "${attack}" "${poison_rate}"
#train_on_poisoned_set none 0.0

# 复制和重命名 attack 文件
source_base=$(find "${backdoor_dir}/poisoned_train_set/${dataset}/" -type d -regex ".*/${attack}_.*_seed=0_exp=${exp_name}")
destination_base="${reprogram_dir}/models/${exp_name}/"
copy_and_rename "$source_base" "$destination_base" "backdoor"

# 复制和重命名 clean 文件
#source_base="${backdoor_dir}/poisoned_train_set/${dataset}/none_0.000_poison_seed=0_exp=${exp_name}/"
#destination_base="${reprogram_dir}/models/${exp_name}/"
#copy_and_rename "$source_base" "$destination_base" "clean"

# ???本的?用
cd "${reprogram_dir}"

#python maint.py -m train -g 0 -s "$num_shadow" -e "$exp_name" -n clean
python maint.py -m train -g 0 -s "$num_shadow" -e "$exp_name" -n backdoor 
