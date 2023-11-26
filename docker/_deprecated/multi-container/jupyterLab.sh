#!/bin/bash

docker compose exec conda-main jupyter lab --notebook-dir=/user --ip='*' --port=9999 --no-browser --allow-root
