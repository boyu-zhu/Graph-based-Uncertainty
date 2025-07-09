# 🚀 Graph-based Uncertainty Metrics for Long-form Language Model Outputs

## Coming Soon

We're working on something exciting! Stay tuned for updates.

### Published Research
Our work builds upon our research paper:  
"Graph-based Uncertainty Metrics for Long-form Language Model Outputs"  
*Authors: Mingjian Jiang, Yangjun Ruan, Prasanna Sattigeri, Salim Roukos, Tatsunori Hashimoto*  
Published in: NeurIPS 2024  
📄 [Read the paper](https://arxiv.org/pdf/2410.20783)

---
---
## Replication Instruction
1. Make sure you've have the environment with essential libraries listed in `requirements.txt` & `vllm`

2. Launch an OpenAI-style API server using vLLM to serve the LLM
```bash
bash vllm.sh
```
3. Set the `os.environ["HF_DATASETS_CACHE"] = ''`in `main.py` line 13

4. Run the main file
```bash
python main.py
```
