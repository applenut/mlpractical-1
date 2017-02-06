#!/bin/bash
#
# Run with two arguments from same directory as script:
# {1} jobname
# {2} Google Cloud Storage bucket to use
#
gcloud beta ml jobs submit training ${1} --module-name=msd-10-genre.task --package-path=msd-10-genre --staging-bucket={2}
