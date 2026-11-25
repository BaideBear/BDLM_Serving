# 核验与复现

## 1. 仅核验已发表结果

在仓库根目录运行 `python3 tools/verify_results.py --table`。比对的是归档中的逐请求指标，不是重新生成答案。包括全部非收敛请求；失败按零产出、零分记，已消耗的 F/Q 不扣除。截断不等于生成失败，按同一评分器判分。

`python tools/render_report.py --markdown docs/results.md` 重建总表；另装 `matplotlib` 后可用 `--plots runs/figures` 根据同一数据绘制六张新图。实验目录中的 PNG 是历史原图，不会被这条命令替换。

`N` 是截到第一个 EOS 为止的最终非特殊输出 token 数。`F` 是生成阶段实际发生的模型调用次数；`Q` 是这些调用中每个请求实际提交的行数。计入刷新、编辑和提交引发的调用，不把纯指针推进计为 forward。完整 prompt 块 prefill 不计，局部 prompt 和生成期重编码计入。`ΣQ/ΣF` 和最大单步 Q 是观察值。不同模型使用自己的 tokenizer，优先在模型内比较。

## 2. 准备环境与样本

历史环境：Linux x86‑64，H800 PCIe，Python 3.12.13，PyTorch 2.9.0+cu128、Transformers 4.57.1、NumPy 2.2.6，BF16、SDPA。CPU 核验与 GPU 生成是两条独立路径。

```bash
git submodule update --init
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

SDAR 还需要与 torch/CUDA 匹配的 `flash_attn==2.8.3.post2`（历史 Triton 3.5.0）。所有 SDAR 模式使用同一 RMSNorm 实现并固定 4 warps，运行时会检查源码哈希，不支持静默切换。HumanEval 评分需要 Linux `bubblewrap`、可用的 user namespace 和 `/usr/bin/python3`；沙箱不可用时报错，不退化成宿主执行。

从原发布方取得以下源文件，放到 `data/raw/`，按表中名称存放。以[冻结清单](../benchmarks/frozen30/manifest.json)的字节哈希为准。清单保留源文件的历史逻辑名称；不要求复刻旧目录层级。

| 来源 | 原文件 | 本地名称 |
|---|---|---|
| [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k/tree/main/main) | `main/test-00000-of-00001.parquet` | `gsm8k-test.parquet` |
| [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/blob/main/test.jsonl) | `test.jsonl` | `math500-test.jsonl` |
| [openai/human-eval](https://github.com/openai/human-eval) | `data/HumanEval.jsonl.gz` | `HumanEval.jsonl.gz` |

```bash
python -m bdlm_serving restore --raw-dir data/raw --output data/samples.json
python -m pytest -q
```

恢复程序核对三份数据、30 个 prompt 和最终 JSON 的 SHA-256；不重新抽样、不替换样本。默认 CPU 测试不加载模型权重。要额外验证真实层的输入 Q、关闭前缀缓存时的逐步等价性，将 `LLADA_TEST_CHECKPOINT` 指向可信本地 LLaDA checkpoint，设置 `CUDA_VISIBLE_DEVICES=""` 后再运行 pytest；它读取配置和模型代码，实例化随机小模型。

## 3. 运行和评分

使用可信本地 checkpoint（通过 `trust_remote_code=True` 装载自定义模型代码）。[归档清单](../benchmarks/checkpoints/)记录元数据及模型代码哈希、权重分片文件名和大小；分片大小不等于权重逐字节校验。显卡选择及资源分配由调用方负责。

```bash
python -m bdlm_serving run --experiment optimus-masked-obs --model llada21 \
  --checkpoint "$LLADA_CHECKPOINT" --samples data/samples.json --output runs/llada-masked
python -m bdlm_serving score --samples data/samples.json --output runs/llada-masked
```

`--experiment` 可选 `baselines`、`flowblock-prefix`、`optimus-masked-obs`；`--model` 可选 `llada21`、`sdar`。每次指定一个实验和模型，运行该实验的全部正式配置和同一批 30 题，输出到**新目录**。三组的默认并发上限分别为 1、32、32，可用 `--batch-size` 覆盖。PrefixCache 调度允许不同 W 共批；masked OBS 按窗口分组。`score` 生成 `scores.csv`；逐题输出仅在本地新目录中，不进入 Git。

LLaDA 的普通配置用 `.5` 接受/编辑阈值，最多 16 个 post 步；Optimus 用 argmax 和 `.9` 的 M2T 阈值，只对选中且已解码的位置尝试 `.5` T2T。SDAR 的 Optimus 保留 argmax / `.9`，其他策略用温度 `.6`、top‑k 20、top‑p `.95`。生成上限 2,048。GSM8K 比较数值，MATH‑500 对最后一个 boxed 答案用 Math-Verify 0.8.0，HumanEval 用原始测试的 pass@1。不同采样器之间的差异不能完全归因于 Q 选择策略。

归档发布时在 CPU 上重建样本、对 30 道标准答案验证评分器、运行控制器测试及总表核验；没有用重排后的发布包再跑一次全量 GPU 基准。正式数字来自各实验的历史 GPU 运行，而非 CPU 测试或性能预测。
