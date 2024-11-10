set -e
set -o xtrace

# mkdir -p $DATA_ROOT/datasets/tinystories/tinystories_v1
mkdir -p $DATA_ROOT/datasets/tinystories/tinystories_v2

# wget -P $DATA_ROOT/datasets/tinystories/tinystories_v1 https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt
# wget -P $DATA_ROOT/datasets/tinystories/tinystories_v1 https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt
wget -P $DATA_ROOT/datasets/tinystories/tinystories_v2 https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget -P $DATA_ROOT/datasets/tinystories/tinystories_v2 https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt