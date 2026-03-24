## Prerequisites
1. uv
2. macos: `brew install libomp`

## Setup:
1. Clone the repo
2. Setup the training data submodule:
    


2. run the following command:

    `git config submodule.recurse true`

    This configures automatic updating of the working tree state of the training data submodule.

## Training Data
private--not part of this repo. Instead, git points to the data as a submodule, which is its own repo. 

### Training Data Setup: 
Training data is a separate private repo which is pointed to as a submodule in the main repo
1. First request access to the private repo. 
2. Then, back in this repo:
    
    ```bash
    git submodule update --init
    git config submodule.recurse true
    ```

The first line configures the submodule, then pulls down the commit of it pointed to in the main repo.
The second line confugres automatic updating of the correct commit of the submodule based on the pointer store

### For changes to training data
In the training_data directory:
1. Run the normal sequence of add-commit-push. This will be committing to the sub-repo that holds the data, and updating the local working tree with new data.
2. The main repo will track the updated commit if you add/commit the directory with the training data:

    ```bash
    git add src/inference/cat/training_data
    git commit -m "update training data"
    ```
It will also be tracked with:
    
    ```bash
    git add .
    ```

Followed by commit.