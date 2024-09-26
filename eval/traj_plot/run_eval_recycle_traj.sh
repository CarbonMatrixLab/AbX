data_dir=~/data/antibody/split
name_idx=./small_test.idx
output_dir=/home/zhutian/Git_repo/AbFold2/eval_data/traj_relax
pred_data_dir=/home/zhutian/Git_repo/AbFold2/eval_data/traj
source /home/zhutian/anaconda3/etc/profile.d/conda.sh
conda activate carbonfold

python traj_evaluate.py -m ${data_dir} -n ${name_idx} -o ${output_dir} -p ${pred_data_dir} -c 50
