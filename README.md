# DDG
IJCAI-2025:In-context Learning Demonstration Generation with Text Distillation

For the overall data distillation process of the DDG method, it is recommended to first refer to https://github.com/arumaekawa/DiLM Configure the corresponding PyTorch environment. It is worth noting that all of our running steps are single machine multi card settings. Secondly, we conduct pre training on the generative model combined with language modeling loss, etc (Taking the sst2 dataset as an example and setting it to run on 4 cards, master_port=37621 is a necessary parameter for single machine multi card operation. At the same time, the trainer can of course set the parameters of the lm part in the config according to their own needs) :

torchrun --nproc_per_node=4 --master_port=37621 src/train.py --config-name=lm data.task_name=sst2

For the subsequent iterative loop algorithm based on gradient descent, we have designed the following command to run:

torchrun --nproc_per_node=4 --master_port=37621 src/train.py --config-name=dc data.task_name=sst2 +generator=pretrained_sst2

Note that this command is also based on the single machine multi card operation setting. Trainers can refer to the parameter settings in the experimental section of the paper, or update the parameter settings according to their own experimental needs. Due to the rigor of the overall experimental setup, we have also set up optional evaluation instructions based on similar work, as follows:

torchrun --nproc_per_node=4 --master_port=37621 src/test.py --config-name=dc data.task_name=sst2 generator.pretrained_model_dir=path/to/pretrained_model_dir

For the new samples obtained after the distillation process, they first need to be converted from a .json file to a .parquet file, and we give examples of the relevant example1.py scripts. For subsequent experiments with the ICL process, the relevant runtime environment configuration can be found in: https://github.com/TIGER-AI-Lab/LongICLBench for the relevant steps.

For example, if we experiment with the BANKING77 dataset and set the round to 5, the model used to be qwen, and the number of samples in the test set used as a query to be 500, then we provide the following instructions for reference:

python banking77_infer.py --round 5 -m qwen --test_number 500

The prompts in the .py scripts are for reference only, and the trainer is free to change and design them to his or her liking. 
