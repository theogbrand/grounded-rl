[3 tools called]

## Implementation Plan: Qwen3-VL Migration

### Phase 1: Environment & Dependencies (Day 1)

**Version Requirements:**
- `transformers>=4.51.0` (current: 4.51.0 ✓)
- `vllm>=0.8.5` (current: 0.8.3 ❌ upgrade needed)
- `qwen-vl-utils` (latest version)
- `torch>=2.1.0` with CUDA 12.x

**Actions:**

```bash
# Update requirements.txt
transformers==4.52.0
vllm==0.8.5
qwen-vl-utils
```

```bash
uv pip install --upgrade transformers==4.52.0 vllm==0.8.5 qwen-vl-utils
```

---

### Phase 2: Code Architecture Changes (Days 2-3)

**New Files to Create:**

1. **`src/vlmsearch/models/qwen3_vl.py`**
   - Copy from `qwen2_5_vl.py`
   - Update imports: `Qwen3VLForConditionalGeneration` (check HF docs if changed)
   - Register as `@register_model("qwen3_vl")`
   - Test image processor changes (likely same as 2.5)

2. **`src/trainer/rl/verl/models/transformers/qwen3_vl.py`**
   - Copy from `qwen2_vl.py` 
   - Update attention mechanics if changed
   - Test MROPE position encoding compatibility

**Files to Modify:**

1. **`src/vlmsearch/models/__init__.py`**
```python
"qwen3_vl": "Qwen3_VL",
```

2. **`src/trainer/rl/verl/models/monkey_patch.py`**
```python
elif model_type in ("qwen2_vl", "qwen2_5_vl", "qwen3_vl"):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLFlashAttention2
    # ... add qwen3 attention forward
```

3. **`scripts/vllm/serve_qwen.sh`**
   - Already handles model path as argument ✓
   - May need `--enable-reasoning --reasoning-parser deepseek_r1` for thinking models

---

### Phase 3: MCTS Rollout Setup (Days 4-5)

**For Qwen3-VL-235B-A22B-Thinking-FP8:**

**Hardware Requirements:**
- 4× GPUs with 48GB+ VRAM each (you have 8 GPUs ✓)
- ~240GB total GPU memory for FP8

**Create: `scripts/vllm/serve_qwen3_thinking.sh`**

```bash
#!/bin/bash

CHECKPOINT_PATH=$1
NUM_GPUS=$2
PORT=$3

echo "CHECKPOINT_PATH: $CHECKPOINT_PATH"
echo "NUM_GPUS: $NUM_GPUS"
echo "PORT: $PORT"

mkdir -p vllm_logs

# Fix CUDA_VISIBLE_DEVICES if it contains UUIDs
if [[ "$CUDA_VISIBLE_DEVICES" == *"GPU-"* ]]; then
  export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
fi

vllm serve $CHECKPOINT_PATH \
    --port $PORT \
    --served-model-name "qwen3_thinking_vllm" \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size $NUM_GPUS \
    --uvicorn-log-level info \
    --max-model-len 32768 \
    --limit-mm-per-prompt "image=30" \
    --mm-processor-kwargs '{"max_pixels":12960000,"min_pixels":4096}' \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \
    --api-key "qwen3" > "vllm_logs/vllm_qwen3_logfile_$(date '+%Y-%m-%d_%H-%M-%S').txt" 2>&1

if [ $? -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Command failed" | tee -a "logfile_$(date '+%Y-%m-%d_%H-%M-%S').txt"
fi
```

**Create: `scripts/mcts/run_mcts_qwen3_235b.sh`**

```bash
#!/bin/bash

NUM_GPUS=4  # 235B FP8 needs 4 GPUs minimum
NUM_PROCESSES=10
dataset="web_grounding"
PORT=9002

export PORT

MODEL="Qwen/Qwen3-VL-235B-A22B-Thinking-FP8"
ACTOR_MODEL="qwen3_thinking_vllm"

# ... copy cleanup function and CUDA fix from run_mcts_qwen72b.sh ...

# Start vLLM server
bash scripts/vllm/serve_qwen3_thinking.sh "$MODEL" "${NUM_GPUS}" "${PORT}" &

# Wait for server...

python -m src.vlmsearch \
    --num_processes=${NUM_PROCESSES} \
    --use_python_mp \
    --model ${ACTOR_MODEL} \
    --seed 42 \
    --judge ${JUDGE} \
    --search_method mcts \
    --max_depth 10 \
    --n_simulations 8 \
    --temperature 1.0 \
    --system_prompt "${SYSTEM_PROMPT}" \
    --pretrained "${MODEL}" \
    --data_files "${DATA_FILE}" \
    --save_tag "MCTS_QWEN3_235B_${dataset}"
```

**Update: `src/vlmsearch/models/qwen_vllm.py`**
- Add support for reasoning tokens `<think>...</think>` parsing
- Handle extended context (235B supports up to 262K tokens)

---

### Phase 4: Fine-tuning Setup for 8B (Days 6-8)

**Create: `examples/train_qwen3_vl_sft.sh`**

```bash
#!/bin/bash

# Based on train_qwen2_5_vl_sft.sh but updated for Qwen3

MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
NUM_GPUS=2  # 8B model fits on 2 GPUs

# Data preparation
DATASET="vigorl_osatlas_train"  # or your custom dataset

torchrun --nproc_per_node=${NUM_GPUS} \
    src/trainer/supervised_finetuning.py \
    --model_name_or_path ${MODEL_PATH} \
    --model_type qwen3_vl \
    --data_path ${DATASET} \
    --output_dir checkpoints/qwen3_vl_8b_sft \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --save_steps 500 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --bf16 \
    --tf32 True \
    --flash_attn True
```

**Update: `src/trainer/rl/verl/utils/dataset.py`**
```python
if self.processor.image_processor.__class__.__name__ in [
    "Qwen2VLImageProcessor", 
    "Qwen2VLImageProcessorFast",
    "Qwen3VLImageProcessor",  # Add this
    "Qwen3VLImageProcessorFast"  # Add this
]:
    # qwen2vl/qwen3vl mrope
```

---

### Phase 5: Testing & Validation (Days 9-10)

**Test Sequence:**

1. **Test vLLM serving:**
```bash
# Kill existing servers
pkill -f "vllm serve"

# Test 8B model first
bash scripts/vllm/serve_qwen.sh "Qwen/Qwen3-VL-8B-Instruct" 1 9003

# Wait for server up, then test
curl http://localhost:9003/v1/models
```

2. **Test 235B FP8 serving:**
```bash
bash scripts/vllm/serve_qwen3_thinking.sh "Qwen/Qwen3-VL-235B-A22B-Thinking-FP8" 4 9002
```

3. **Test MCTS with 235B:**
```bash
# Use small test dataset first
bash scripts/mcts/run_mcts_qwen3_235b.sh
```

4. **Test fine-tuning with 8B:**
```bash
# Use small subset for sanity check
bash examples/train_qwen3_vl_sft.sh
```

---

### Phase 6: Migration Checklist

**Pre-flight checks:**
- [ ] Backup current working Qwen2.5 configs
- [ ] Check HF model cards for breaking changes
- [ ] Verify 235B-FP8 model is actually FP8 quantized (memory requirements)
- [ ] Test with Qwen3-VL-7B first (smaller model to validate changes)

**Code validation:**
- [ ] `qwen3_vl.py` loads model without errors
- [ ] Image processor handles vision inputs correctly  
- [ ] Tokenizer chat template works
- [ ] MROPE position encoding for multi-turn
- [ ] Thinking tokens parsed correctly from 235B model

**System validation:**
- [ ] vLLM serves 8B model successfully
- [ ] vLLM serves 235B-FP8 with 4 GPUs
- [ ] MCTS completes rollouts without OOM
- [ ] Fine-tuning doesn't crash

---

### Phase 7: Optimization (Days 11+)

**Performance tuning:**
- Profile vLLM memory usage for 235B
- Tune `--max-model-len` based on actual context needs
- Optimize `NUM_PROCESSES` for MCTS parallelism
- Consider using `--swap-space` if hitting memory limits

**Expected improvements:**
- 235B model: Better reasoning, especially with thinking tokens
- Longer context support (32K-262K vs current)
- Potential speed improvements from FP8 quantization

---

### Quick Start Command Sequence

```bash
# Day 1: Upgrade deps
cd /data/projects/71001002/ob1/grounded-rl
uv pip install --upgrade transformers==4.52.0 vllm==0.8.5

# Day 2-3: Create new model files (I can do this for you)

# Day 4: Test 235B serving
bash scripts/vllm/serve_qwen3_thinking.sh "Qwen/Qwen3-VL-235B-A22B-Thinking-FP8" 4 9002

# Day 5: Run MCTS
bash scripts/mcts/run_mcts_qwen3_235b.sh

# Day 6-8: Setup finetuning for 8B
bash examples/train_qwen3_vl_sft.sh
```

Want me to start implementing Phase 1-2 now?