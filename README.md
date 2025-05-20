# DDG
IJCAI-2025
For the new samples obtained after the distillation process, they first need to be converted from a .json file to a .parquet file, and we give examples of the relevant .py scripts. For subsequent experiments with the ICL process, the relevant runtime environment configuration can be found in: https://github.com/TIGER-AI-Lab/LongICLBench for the relevant steps.
For example, if we experiment with the BANKING77 dataset and set the round to 5, the model used to be qwen, and the number of samples in the test set used as a query to be 500, then we provide the following instructions for reference:
python banking77_infer.py --round 5 -m qwen --test_number 500
