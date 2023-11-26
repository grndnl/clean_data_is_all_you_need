# DLA - Pipeline Based on DiT or LayoutLMV3

## References:
- [DiT Paper](https://github.com/microsoft/unilm/tree/master/dit)
- [DiT Repo](https://github.com/microsoft/unilm/tree/master/dit)
- [LayoutLMv3 Paper](https://arxiv.org/abs/2204.08387)
- [LayoutLMv3 Repo](https://github.com/microsoft/unilm/tree/master/layoutlmv3)

## Container scripts:
- From a terminal in the `docker/layoutlmv3-cuda-detectron` directory
- Build conainer: `sh build_image.sh`
- Start container (with docker compose): `sh up_compose.sh`
- Stop the container (with docker compose): `sh down_compose.sh`