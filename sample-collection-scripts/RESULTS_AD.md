# Artifact Descriptions for the Results Folder

## 📊 **Application Parameters & Tasks Summary**

### 🦙 **LLaMA (Text Generation)**
**Parameters Used:**
- `--benchmark --num-generations 3 --quiet --metrics`
- Model: `huggyllama/llama-7b` (LLaMA-7B, 7 billion parameters)
- Precision: `float16` 
- Device: CUDA
- Total generations: **24 text completions** across 8 prompts

**What it was doing:**
- **Text generation benchmark** with predefined prompts
- Generated 1,193 total tokens in 47.65 seconds (A100)
- Average throughput: **25 tokens/second**
- Model loading time: ~24 seconds (A100)
- Using PyTorch 2.1.0 with CUDA acceleration

---

### 🎨 **Stable Diffusion (Image Generation)**
**Parameters Used:**
- `--prompt "a photograph of an astronaut riding a horse" --steps 50 --job-id 21034496 --log-level INFO`
- Model: `CompVis/stable-diffusion-v1-4` (SD v1.4)
- Scheduler: DPM++ (optimized)
- Resolution: 512×512 pixels
- Precision: `float16`

**What it was doing:**
- **Single image generation** from text prompt
- Generated 1 image in **13.31 seconds** (A100)
- Used 2.58GB GPU memory
- Applied memory optimizations (attention slicing)
- Images were not saved (research/profiling mode)

---

### 👁️ **Vision Transformer (Image Classification)**
**Parameters Used:**
- `--benchmark --num-images 1200 --batch-size 4 --model google/vit-large-patch16-224 --precision float16`
- Model: `google/vit-large-patch16-224` (ViT-Large)
- Batch size: 4 images per batch
- Target: Process 1,200 images total
- Precision: `float16`

**What it was doing:**
- **Image classification benchmark** on sample images
- Download and classify images from web sources
- Model loaded successfully (13.5 seconds) (A100)
- Was designed to process 300 batches of 4 images each

---

### 🎤 **Whisper (Speech Recognition)**
**Parameters Used:**
- `--benchmark --model base --num-samples 10 --quiet`
- Model: `openai/whisper-base` (base model, ~74M parameters)
- Processing: 10 audio samples
- Mode: Benchmark (automated)

**What it was doing:**
- **Speech-to-text transcription** on audio samples
- Processed 10 audio samples in **30.99 seconds** (A100)
- Average: ~3.1 seconds per audio sample (A100)
- Used predefined audio dataset for consistency

---

## 🔬 **Experimental Setup Details**

### **Common Profiling Configuration:**
- **GPU Types**: A100 (40GB PCIE), H100, V100
- **Frequency Range**: 510-1410 MHz (A100), 510-1785 MHz (H100)
- **Profiling Mode**: DVFS (Dynamic Voltage/Frequency Scaling)
- **Monitoring**: DCGMI at 25-50ms intervals
- **Runs per frequency**: 5 repetitions for statistical validity
- **Metrics captured**: Power, temperature, utilization, memory usage

### **Key Research Focus:**
1. **Energy-Delay Product (EDP)** optimization
2. **Cross-architecture comparison** (V100, A100, H100)
3. **Frequency scaling impact** on different AI workloads
4. **Real-world inference scenarios** with practical parameters

The experiments were designed to represent **realistic AI inference workloads** rather than synthetic benchmarks, using standard model sizes and parameters that would be encountered in production deployments.