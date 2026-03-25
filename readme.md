## Prerequisites
1. uv
2. macos: `brew install libomp`

## Setup:
1. Clone the repo
2. `uv sync`

## Training Data
Training data is a separate private repo. The main repo contains a submodule, pointing to a selected commit in the training data repo.
This can be updated--see below.

### Setup: 
1. First request access to the private repo. 
2. Then, back in this repo:
    
    ```bash
    git submodule update --init
    git config submodule.recurse true
    ```

The first line configures the submodule, then pulls down the commit of it pointed to in the main repo.
The second line confugres auto updating of the training data working tree based on branch.
For example, if you switched branches and the new branch points to a different commit,
that commit will be automatically checked out while in that branch on the main repo.

### For changes to training data
With terminal in the training_data directory `src/inference/cat/training_data`:
1. Run the normal sequence of add-commit-push. This will be committing to the sub-repo that holds the data.
2. The main repo will track the updated commit if you commit changes on the submodule's directory.
    Note: this only updates the pointer to the commit in submodule and doesnt track the contents in the main repo.

    ```bash
    git add src/inference/cat/training_data
    ```
Or just add everything. The submodule pointer will be added too:
    
    ```bash
    git add .
    ```

Followed by commit.

## To Train the model:
In the project root run:
`python -c "from src.inference.cat.train import train; print(train())"`
