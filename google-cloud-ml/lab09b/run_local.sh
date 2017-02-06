#!/bin/bash
#
# Run from same directory as script
gcloud beta ml local train --package-path=msd-10-genre --module-name=msd-10-genre.task
