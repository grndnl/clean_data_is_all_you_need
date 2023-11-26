#!/bin/bash

docker compose exec layoutlmv3-pip jupyter notebook --notebook-dir=/user --ip='*' --port=8888 --no-browser --allow-root
