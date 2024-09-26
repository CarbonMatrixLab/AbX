data_dir=~/data/antibody/split
name_idx=~/data/antibody/all_antibody.idx
output_dir=~/data/antibody/
source /home/zhutian/anaconda3/etc/profile.d/conda.sh
conda activate carbonfold

python cacl_energy.py -m ${data_dir} -n ${name_idx} -o ${output_dir} -c 30
