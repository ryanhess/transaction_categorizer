## Prerequisites
1. uv
2. macos: `brew install libomp`

## Training Data
private--not part of this repo. Instead, git points to the data as a submodule, which is its own repo. For setup,
request access to the private training data repo. Once you have access, run:
```bash
git submodule init
git submodule update
```

For changes to the training data, they can be made in place. In the training_data directory:
1. Run the normal sequence of add-commit-push. 
2. To deploy with the updated training data, first run from project root:
```bash
git add src/inference/cat/training_data
git commit -m "update training data"
```