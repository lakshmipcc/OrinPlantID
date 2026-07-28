# 🌿 Jetson Orin Nano: Edge AI Plant Identifier PWA

An end-to-end, locally hosted Progressive Web Application (PWA) running on an **NVIDIA Jetson Orin Nano**. The system leverages a dual-agent local model architecture using **Ollama**, wrapped in a **FastAPI** backend, reverse-proxied by **Caddy** with automatic HTTPS, and delivered through a mobile-optimized frontend.

## 🛠️ Architecture Overview
---
```text
┌────────────────────────────────────────────────────────────────────────┐
│                   📱 Mobile Phone (iOS / Android PWA)                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    │ HTTPS (Port 443)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         🔒 Caddy Reverse Proxy                         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    │ Subpath Route: /plantid*
                                    ▼
┌──────────────────────────────────────────────────┐   ┌─────────────────┐
│             ⚡ FastAPI Backend Engine            │◄──┤ ⚙️ systemd      │
│             Port 8000 (Loopback)                 │   │ (Auto-restart)  │
└─────────────────────────┬────────────────────────┘   └─────────────────┘
                          │
                          │ Base64 Payload (Port 11434)
                          ▼
┌──────────────────────────────────────────────────┐   ┌─────────────────┐
│            🧠 Ollama Inference Engine            │──►│ 🟢 NVIDIA CUDA  │
│            Model: qwen2.5vl:3b-lowmem            │   │ Orin VRAM & Swap│
└──────────────────────────────────────────────────┘   └─────────────────┘
```
---

## ✨ Key Features & Optimizations

* **Local Hardware Acceleration:** Runs quantized LLM/VLMs natively on Jetson CUDA cores without external cloud API dependencies.
* **Unified Memory & Swap Spilling:** Configured to dynamically spill VRAM allocations into system RAM and NVMe swap when VRAM fragments.
* **Multi-Day Uptime & Fault Tolerance:** Uses a short `5m` VRAM keep-alive policy, reduced swappiness (`10`), and `systemd` auto-restarts to prevent memory leaks and system freezes.
* **Disconnect Protection:** Handles client-side socket drops (`BrokenPipeError`, `asyncio.CancelledError`) gracefully in FastAPI without dropping background workers or cluttering system logs.
* **Dual-Model Agent Architecture:** Orchestrates between a high-resolution Vision Model (`qwen2.5vl:3b-lowmem`) for botanical accuracy and a Hybrid DeltaNet Text Model (`qwen3.5:2b`) for reasoning and tool usage.
* **Chain-of-Thought (CoT) Visual Reasoning:** Prompts the VLM to analyze growth habits, leaf margins, and surface textures *before* determining species identity to minimize hallucinations.
* **Backend Reasoning Truncation:** Strips internal AI reasoning traces on the backend to return clean, structured identification cards to the mobile client.
* **Mobile PWA Integration:** Supports native hardware camera capture (`capture="environment"`) and photo gallery uploads on both iOS Safari and Android Chrome.

---

## 🧠 Model Rationale & Architectural Comparison

Running AI on an 8GB Jetson Orin Nano requires selecting models whose visual and text architectures match specific tasks without exceeding the ~6GB available system memory.

| Feature / Metric | `qwen2.5vl:3b` (Plant-ID Agent) | `qwen3.5:2b` (Text / Writing Agent) |
| :--- | :--- | :--- |
| **Primary Strength** | Fine-grained visual feature extraction | Reasoning, tool-use, long-text generation |
| **Attention Architecture** | Full Attention (high-resolution ViT encoder) | Hybrid Gated DeltaNet (extremely low RAM overhead) |
| **Native Context** | 64K | 262K |
| **Ollama Quantized RAM** | ~5.0 GB | ~2.7 GB |
| **Botanical Accuracy** | **High** (detects trichomes, leaf veins, tubercles) | **Medium-Low** (prone to generic guessing) |
| **Tool Calling / JSON** | Moderate | **Exceptional** (built for agent environments) |

### Why Two Models?
1. **`qwen2.5vl:3b` for Botany:** Botanical identification requires analyzing micro-textures (vein patterns, stem hairs, 3D tubercles). `qwen2.5vl` uses a high-resolution visual transformer (ViT) encoder that preserves spatial awareness across aspect ratios.
2. **`qwen3.5:2b` for Text/Tools:** `qwen3.5` uses early-fusion image compression designed for UI screenshots and charts (which blurs micro-textures), but its Hybrid Gated DeltaNet architecture allows it to handle 256K contexts and function calling with almost zero memory overhead.

---

## ⚙️ Jetson Orin Nano System Setup

### 1. SSH into the Jetson & Install Ollama
```
ssh username@jetson-ip-address
curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh
```
### 2. Configure 16GB NVMe Swapfile & Tune Swappiness

Why: On an 8GB Jetson Orin Nano, the operating system and system services consume ~2GB of RAM, leaving only ~6GB for AI models.
 A 16GB NVMe swapfile prevents Out-Of-Memory (OOM) crashes during model swaps.
 Setting swappiness=10 prevents Ubuntu from thrashing the SSD, avoiding disk-I/O locks that freeze the board over multiple days.

Check your current swap memory:
```
free -h
```
If Swap is less than 8GB, allocate a 16GB swapfile on your NVMe storage:
```
sudo fallocate -l 16G /var/swapfile
sudo chmod 600 /var/swapfile
sudo mkswap /var/swapfile
sudo swapon /var/swapfile
echo '/var/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
Set swappiness to 10 for multi-day stability:
sudo sysctl vm.swappiness=10
echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
```

### 3. Optimize Ollama Systemd Override & Keep-Alive Policy

Why: We limit model instances to 1, enable 8-bit KV caching (q8_0), allow dynamic CUDA VRAM spilling, and auto-restart on crashes.\ Crucially, OLLAMA_KEEP_ALIVE=5m unloads the model from VRAM after 5 minutes of inactivity so the OS isn't choked 24/7.

Create and write the override configuration:
```
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf << 'EOF_OLLAMA'
[Service]
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_KEEP_ALIVE=5m"
Environment="GGML_CUDA_ENABLE_UNIFIED_MEMORY=1"
Restart=always
RestartSec=3s
EOF_OLLAMA
```
Apply changes and restart the Ollama background daemon:
```
sudo systemctl daemon-reload
sudo systemctl restart ollama
```


## 📥 Model Installation & Custom Modelfile Build
### 1. Pull Base Models

Note: The ollama pull command is strictly a network and disk-write operation.\ It downloads model weights directly to your storage without filling active RAM.

#### Pull the 2B text/agent model
`
ollama pull qwen3.5:2b
`
#### Pull the 3B Vision-Language model (avoiding the default 7B tag which overwhelms 8GB devices)
`
ollama pull qwen2.5vl:3b
`
#### Pull the 2B text/agent model 
`
ollama pull qwen3.5:2b
`
#### Pull the 3B Vision-Language model (avoiding the default 7B tag which overwhelms 8GB devices)
`
ollama pull qwen2.5vl:3b
`
### 2. Create the Low-Memory VLM Variant (qwen2.5vl:3b-lowmem)

Why: By default, Vision Models attempt to reserve massive context windows (64K+ tokens) in RAM. For plant identification, we only need ~2048 tokens.Capping num_ctx reduces VRAM consumption dramatically.

Create a custom Modelfile:
```
echo -e "FROM qwen2.5vl:3b\nPARAMETER num_ctx 2048" > ~/Modelfile.vlm
Build the optimized local variant:
ollama create qwen2.5vl:3b-lowmem -f ~/Modelfile.vlm
```

## 🧪 Verification & Hardware Testing
### 1. Test Function Calling on qwen3.5:2b

Verify that tool calling works properly without running out of memory:
```
curl -s http://localhost:11434/api/chat 
  -H "Content-Type: application/json" 
  -d '{
    "model": "qwen3.5:2b"
    "messages": [{"role": "user", "content": "What is the weather in Madrid?"}],
    "stream": false,
    "options": {"num_ctx": 16384},
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters":{ 
          "type": "object",
          "required": ["city"],
          "properties": {
            "city": {"type": "string", "description": "City name"}
          }
        }
      }
    }]
  }'
```

If this command throws an OOM error, remove the "num_ctx": 16384 option line to fall back to safe default context limits).
### 2. Image Pre-processing Optimization Test

Why: Passing raw high-resolution (12MP+) images directly into a local VLM creates a massive compute graph (~1.78 GB allocation).Downscaling photos to a maximum dimension of 512px reduces the allocation to <150 MB, speeding up processing significantly.

Downscale a test image using Python PIL:
```
python3 -c "from PIL import Image; img = Image.open('test_plant.jpg'); img.thumbnail((512, 512)); img.save('test_plant_512.jpg')"
Run a CLI test with the low-memory VLM:
ollama run qwen2.5vl:3b-lowmem
```

>>> Identify this plant and tell me its botanical name: /path/to/test_plant_512.jpg
>>> /exit
## 🚀 Application Server & Reverse Proxy Setup
### 1. Repository File Structure

Ensure your files are organized in your project directory (~/orin-plant-id):
```text 
├── server.py              # FastAPI application with disconnect exception handling
├── public/
│   ├── index.html         # Mobile PWA user interface
│   ├── manifest.json      # Progressive Web App manifest
│   ├── sw.js              # Service Worker for offline/cached asset handling
│   └── icon.png           # App launch icon (512x512)
├── .gitignore             # Git ignore file for secrets and environments
└── README.md              # Documentation
```

### 2. Configure FastAPI as an Auto-Restarting System Service

Why: Running Uvicorn manually in a terminal causes app downtime whenever SSH disconnects. 
Setting up plantid.service with Restart=always ensures FastAPI runs continuously in the background and recovers instantly from unhandled exceptions.

Create the systemd file:
```
sudo tee /etc/systemd/system/plantid.service << 'EOF_SERVICE'
[Unit]
Description=Jetson Plant ID FastAPI Application
After=network.target ollama.service

[Service]
Type=simple
User=YOUR_JETSON_USERNAME
WorkingDirectory=/home/YOUR_JETSON_USERNAME/orin-plant-id
ExecStart=/usr/bin/python3 -m uvicorn server:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=2s
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
EOF_SERVICE~
(Replace YOUR_JETSON_USERNAME with your actual Linux user).

Enable and start the service:
sudo systemctl daemon-reload
sudo systemctl enable plantid
sudo systemctl start plantid
```

### 3. Caddy Reverse Proxy Configuration

Configure /etc/caddy/Caddyfile to enable path-stripping, extended proxy timeouts, and SSL termination:\
YOUR_DOMAIN.servebeer.com 
1. Route traffic for the Plant ID app (strips /plantid prefix internally)
```
    handle_path /plantid* {
        reverse_proxy localhost:8000 {
            transport http {
                read_timeout 90s
            }
        }
    }
```
2. Fallback route for static file browser
```
    handle {
        root * /var/www/html
        file_server browse
    }
    encode gzip zstd
}
```

Validate and restart Caddy:
```
sudo caddy validate --config /etc/caddy/Caddyfile\
sudo systemctl restart caddy
```

## 📱 Mobile Installation (PWA)
#### 1. Open https://YOUR_DOMAIN.servebeer.com/plantid on a mobile browser using cellular data or an external network.
#### 2. iOS Safari: Tap Share $\rightarrow$ Add to Home Screen.
#### 3. Android Chrome: Tap Menu (⋮) $\rightarrow$ Install app or Add to Home Screen.

#### 📄 LicenseDistributed under the MIT License.
