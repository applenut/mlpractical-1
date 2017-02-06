A basic setup to run a copy of _[Lab
09b](https://github.com/CSTR-Edinburgh/mlpractical/blob/mlp2016-7/coursework3-4/notebooks/09b_Music_genre_classification_with_the_Million_Song_Dataset.ipynb)_
on the [Google Cloud Machine Learning Platform](https://cloud.google.com/ml/).

You will need to have followed instructions on [Getting Set
Up](https://cloud.google.com/ml/docs/how-tos/getting-set-up) with the Google
Cloud SDK.

You will also need to copy the data files to a Google Storage bucket.  Look in
`msd-10-genre/task.py` to modify the data provider so that it loads the data
from the right place.

Also `run_gcloud.sh` and `run_local.sh` should help get you started.

_Disclaimer_: This is meant as a starting point for anyone doing the MLP course
to run training on Google Cloud. I am not responsible for anything you do with it!
