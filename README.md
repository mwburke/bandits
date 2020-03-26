# Bandits

Exploring some common multi-armed bandit algorithms. Check out the [walkthrough](walkthrough.ipynb) to see explanations of the 3 algorithms I looked into, comparisons of their hyperparameters within algorithms, between algorithms and with different reward distributions.

I also worked on implementing some contextual bandit algorithms in the [contextual bandits file](contextual_bandits.py). based on the following paper:

Adapting multi-armed bandits policies to contextual bandits scenarios by David Cortes

arXiv:1811.04383

The main idea is to have "oracles" (binary classification models) for each arm, and
to compare the predictions from each oracle to decide which arm to pull. This method
does make some strong assumptions about having binary rewards among other things, so
it may not be a valid approach for real-valued rewards.

NOTE: these are untested, just coding up to understand logic of the paper a little better and thought I would keep them here for reference.
