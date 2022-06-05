
Repository contains the code for 1st place solution of [BirdClef2022 competition](https://www.kaggle.com/competitions/birdclef-2022/leaderboard). 
The goal of the competition was to identify bird calls from given soundspaces.

- [Solution post](https://www.kaggle.com/competitions/birdclef-2022/discussion/327047)
- [Inference notebook](https://www.kaggle.com/code/ivanpan/fork-of-fork-of-cls-exp-1-870246-021187-967146?scriptVersionId=96433080)

## reproduce

run 

> sh 5folds.sh

## train

> sh train_single.sh 0 ./outs cls_nf0_v3 0 /kaggle/input nf0_v3 

respectively, *0* number of gpus, *./outs* validation output directory, *cls_nf0_v3*  config , *0* fold number, */kaggle/input/* is root data directory that contains `birdclef-2021` and `birdclef-2022` data.

## inputs

Download *noise_30sec*, *ff1010bird_nocall*, *train_soundscapes* folders from [here](https://www.kaggle.com/datasets/christofhenkel/birdclef2021-background-noise).

## manually chunked data

### *maupar*

- Download audio clips from [here](https://www.kaggle.com/datasets/realsleim/maupar-fix).
- Move audio files under birdclef-2022/train_audio/maupar
- Use folds.csv

## references

Parts of solution taken from

- https://github.com/selimsef/xview3_solution
- https://github.com/ChristofHenkel/kaggle-birdclef2021-2nd-place
- https://github.com/ChristofHenkel/kaggle-landmark-2021-1st-place
