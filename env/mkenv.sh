#!/bin/bash
export PATH=$HOME/.local/bin:$PATH
uv venv -q --python 3.12 ~/venv-score
uv pip install --python ~/venv-score/bin/python --find-links ~/g4/wh vllm==0.19.1 swegemma==0.2.11 adk-submission==0.2.13 adk-eval-core==0.1.0 google-adk==1.36.1 transformers==5.13.1 tokenizers==0.22.2 flashinfer-python==0.6.6 flashinfer-cubin==0.6.6 compressed-tensors==0.15.0.1 safetensors==0.8.0 litellm pandas pyarrow networkx pytest rich > ~/g4/logs_env.txt 2>&1
echo "EXIT $?" >> ~/g4/logs_env.txt
~/venv-score/bin/python -c "import vllm, swegemma, adk_submission, google.adk, torch; print(\"vllm\", vllm.__version__, \"torch\", torch.__version__, torch.version.cuda, torch.cuda.device_count())" >> ~/g4/logs_env.txt 2>&1
