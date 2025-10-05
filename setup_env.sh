#!/bin/bash
mkdir -p sql-gen/
cd sql-gen/

python -m venv .venv
source .venv/bin/activate
pip install vllm==0.8.5.post1 func_timeout pandas pandas-stubs wandb gdown tqdm

mkdir data/
cd data/

wget https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip
unzip minidev.zip 
mv minidev/MINIDEV .
rm -r minidev
rm minidev.zip 
mv MINIDEV/ bird_minidev

gdown --fuzzy https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view?usp=sharing
unzip spider_data.zip 
rm -r __MACOSX/
rm spider_data.zip

cd ..