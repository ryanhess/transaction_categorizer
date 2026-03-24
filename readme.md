## Prerequisites
1. uv
2. macos: `brew install libomp`

## Setup:
1. Clone the repo
2. Setup the training data submodule:
    
    First request access to the private repo. Then, back in this repo:
    
    ```bash
    git submodule init
    git submodule update
    ```

    This configures the submodule, then pulls down the commit of it pointed to in the main repo.

2. run the following command:

    `git config submodule.recurse true`

    This configures automatic updating of the working tree state of the training data submodule.

## Training Data
private--not part of this repo. Instead, git points to the data as a submodule, which is its own repo. For setup,
request access to the private training data repo. Once you have access, run:

    ```bash
    git submodule init
    git submodule update
    ```

To streamline the submodule changes (if any) while switching between branches, run (one time only):

    `git config submodule.recurse true`

For changes to the training data, they can be made in place. In the training_data directory:
1. Run the normal sequence of add-commit-push. This will be committing to the sub-repo that holds the data.
2. To deploy with the updated training data, first run from project root:

    ```bash
    git add src/inference/cat/training_data
    git commit -m "update training data"
    ```

    `git add .` also works if this is part of a larger set of changes.