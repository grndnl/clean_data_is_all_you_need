#!/bin/bash

docker compose exec layoutlmv3-cuda-detectron-conda jupyter lab --notebook-dir=/user --ip='*' --port=8888 --no-browser --allow-root
