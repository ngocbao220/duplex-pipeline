# Duplex-Pipelines

Hệ thống xử lý và tách kênh âm thanh cuộc thoại hai người nói (Full-Duplex Speech Separation) gồm các pipeline **DuplexChat** (DialogueSidon), **Sommelier** (SepReformer), và bộ đánh giá chất lượng **Reference-Free Stereo Benchmark**. Toàn bộ cấu hình được quản lý tập trung và phân cấp bằng **Hydra**.

---

## 1. Cài đặt Môi trường Chuẩn chỉ (Tránh Xung đột Dependency)

Để tránh xung đột thư viện giữa các giai đoạn xử lý (đặc biệt giữa NeMo / Streaming Sortformer của DuplexChat và stack SepReformer của Sommelier), dự án hỗ trợ 2 giải pháp thiết lập môi trường tối ưu:

### Yêu cầu hệ thống chung
```bash
# Ubuntu / Debian
apt-get update && apt-get install -y ffmpeg libsndfile1

# macOS
brew install ffmpeg libsndfile
```

---

### Phương án 1: Tự động qua script `setup_env.sh` (Khuyến nghị)
Script tự động kiểm tra và cài đặt đúng phiên bản tương thích không xung đột:

```bash
# Cài đặt môi trường cho DuplexChat & Benchmark (Phase 1, 2-DuplexChat, 3)
bash setup_env.sh --duplexchat

# Cài đặt môi trường riêng cho Sommelier (Phase 2-Sommelier)
bash setup_env.sh --sommelier

# Hoặc cài đặt đầy đủ cả 2 môi trường Conda độc lập:
bash setup_env.sh --all

# Hoặc thiết lập môi trường cô lập bằng uv cho từng pipeline:
bash setup_env.sh --uv
```

---

### Phương án 2: Thiết lập Conda thủ công theo từng Phase

#### A. Môi trường DuplexChat & Pipeline chính (`duplexchat` - Python 3.12)
Dùng cho **Convert**, **Phase 1: Dialogue Filtering**, **Phase 2: DuplexChat Separation**, và **Phase 3: Stereo Benchmark**:
```bash
conda create -n duplexchat python=3.12 -y
conda activate duplexchat

# Cài đặt PyTorch tương thích CUDA
pip install torch==2.8.0 torchaudio==2.8.0

# Cài đặt dependencies chuẩn của DuplexChat
pip install -r requirements-duplexchat.txt

# Cài đặt NeMo Toolkit cho NVIDIA Streaming Sortformer v2.1
pip install Cython packaging
pip install "nemo_toolkit[asr]>=2.3.0"
```

#### B. Môi trường Sommelier (`sommelier` - Python 3.11)
Dùng riêng cho **Phase 2: Sommelier (SepReformer & Reconstruction)**:
```bash
conda create -n sommelier python=3.11 -y
conda activate sommelier

pip install -r requirements-sommelier.txt
```

---

### Cấu hình Chế độ Chạy
- **Server Offline** (`env=sever` - Mặc định): Chạy trên server cluster không có internet, sử dụng trực tiếp các mô hình và dữ liệu tại `/storage-voice/...`. Tự động thiết lập `HF_HUB_OFFLINE=1`.
  - **Mô hình SepReformer**: Được trỏ đường dẫn trực tiếp trong config (`SEPREFORMER: ${env.paths.base_models}/epoch.0180.pth`), không cần export thủ công từ terminal.
- **Local / Dev Online** (`env=dev`): Chạy trên máy local/dev có internet, tự động tải mô hình từ Hugging Face Hub. Cần export token:
  ```bash
  export HF_TOKEN="hf_your_huggingface_token"
  ```

---

## 2. Các Lệnh Chạy Đầy Đủ Flag & Giải Thích Ý Nghĩa

Mọi shell script đều hỗ trợ trực tiếp các cờ nguồn dữ liệu (`--youtube`, `--podcast-index`, `--all`), cờ GPU (`--gpu <id>`), cờ môi trường (`--dev`, `--sever`), và cờ chạy thử an toàn (`--dry-run`).

### 2.0. Chuyển đổi Dữ liệu Thô (Convert to WAV)
Chuyển đổi các file crawled `.webm`, `.m4a`, `.tar.gz` sang chuẩn WAV PCM 16kHz mono. Sử dụng đa luồng CPU (mỗi tiến trình gán cứng `-threads 1` để CPU $\le$ 100%):

```bash
bash convert.sh --youtube --workers 8         # Chuyển đổi dữ liệu YouTube (8 workers)
bash convert.sh --podcast-index --workers 8   # Chuyển đổi dữ liệu Podcast Index (8 workers)
bash convert.sh --all --workers 8             # Chuyển đổi cả hai nguồn (mặc định)
```

---

### 2.1. Chạy Toàn bộ Pipeline (End-to-End)
Chạy tuần tự toàn bộ các pha: Lọc cuộc thoại $\rightarrow$ Tách kênh stereo $\rightarrow$ Đánh giá benchmark.

```bash
# Bằng Shell Script:
bash run_pipeline.sh --youtube --gpu 0 --workers 2      # Chạy full cho YouTube trên GPU 0 (2 workers)
bash run_pipeline.sh --podcast-index --gpu 1 --workers 2# Chạy full cho Podcast Index trên GPU 1
bash run_pipeline.sh --all --gpu 0 --workers 2          # Chạy tuần tự cho cả hai nguồn
bash run_pipeline.sh --sommelier --youtube --workers 2  # Chạy pipeline Sommelier
bash run_pipeline.sh --youtube --dev                    # Chạy chế độ Dev (online HuggingFace)
bash run_pipeline.sh --youtube --dry-run                # Chạy thử (không ghi dữ liệu)

# Hoặc bằng Python (Hydra):
python run_pipeline.py \
    step=all \
    pipeline=duplexchat \
    env=sever \
    gpu=0 \
    optimization.workers=2 \
    data.source=youtube \
    dry_run=false
```

**Ý nghĩa các cờ:**
- `--youtube` / `data.source=youtube`: Chọn folder dữ liệu YouTube (`data/raw/youtube`).
- `--podcast-index` / `data.source=podcast_index`: Chọn folder dữ liệu Podcast Index (`data/raw/podcast_index`).
- `--all`: Chạy lần lượt cho cả hai folder dữ liệu.
- `--duplexchat` / `--sommelier`: Chọn pipeline tách âm (`duplexchat` hoặc `sommelier`).
- `--gpu <id>` / `gpu=<id>`: ID card GPU thực thi (ví dụ: `0`, `1`, `2`).
- `--workers <N>` / `-w <N>` / `optimization.workers=<N>`: Số worker chạy song song (tối ưu cho A100 40GB).
- `--dev`: Chuyển sang môi trường dev online.
- `--dry-run` / `dry_run=true`: In câu lệnh sẽ thực thi mà không can thiệp dữ liệu.

---

### 2.2. Chạy Riêng Từng Bước (Phase-by-Phase)

#### Bước 1: Tiền xử lý & Lọc cuộc thoại (Dialogue Filtering)
Resample 16kHz, phân đoạn người nói (Streaming Sortformer), trích xuất các đoạn đối thoại đúng 2 người, lọc bỏ nhạc nền (Demucs) và kiểm tra tiếng Việt (Whisper LID).

```bash
# Bằng Shell Script:
bash dialogue_split.sh --youtube --gpu 0 --workers 4    # Lọc thoại YouTube (4 workers lấp đầy GPU A100)
bash dialogue_split.sh --podcast-index --gpu 1 --workers 4
bash dialogue_split.sh --all --gpu 0 --workers 4        # Chạy tuần tự cả 2 nguồn
bash dialogue_split.sh --youtube --dry-run               # Chạy thử

# Hoặc bằng Python (Hydra):
python run_pipeline.py \
    step=split_dialogue \
    env=sever \
    gpu=0 \
    optimization.workers=4 \
    diarization=sortformer \
    pipeline.music_filter.enabled=true \
    pipeline.lid.enabled=true \
    pipeline.lid.code=vi \
    data.source=youtube
```

**Ý nghĩa các cờ:**
- `step=split_dialogue`: Chạy pha phân đoạn và lọc thoại 2 người.
- `--workers 4`: Chạy song song 4 file audio trên GPU A100 (chiếm ~15-20GB / 40GB VRAM).
- `diarization=sortformer`: Sử dụng mô hình `nvidia/diar_streaming_sortformer_4spk-v2.1`.
- `pipeline.music_filter.enabled=true`: Bật bộ lọc loại bỏ nhạc nền bằng Demucs (`false` để tắt).
- `pipeline.lid.enabled=true`: Bật nhận diện ngôn ngữ bằng Whisper để chỉ giữ thoại tiếng Việt.
- `pipeline.lid.code=vi`: Mã ngôn ngữ cần lọc (mặc định `vi`).

---

#### Bước 2A: Tách kênh âm thanh bằng DuplexChat (DialogueSidon)
Cắt chunk hội thoại và sử dụng mô hình khuếch tán DialogueSidon để phân tách 2 giọng nói đè lên nhau thành 2 kênh stereo 24kHz.

```bash
# Bằng Shell Script:
bash duplexchat.sh --youtube --chunk 60 --gpu 0 --workers 2  # Tách thoại 2 workers song song
bash duplexchat.sh --podcast-index --chunk 60 --gpu 1 --workers 2
bash duplexchat.sh --all --gpu 0 --workers 2
bash duplexchat.sh --youtube --chunk 60                      # Mặc định GPU 2
bash duplexchat.sh --youtube --dry-run                       # Chạy thử

# Hoặc bằng Python (Hydra):
python run_pipeline.py \
    step=separate_dialogue \
    pipeline=duplexchat \
    env=sever \
    gpu=0 \
    optimization.workers=2 \
    pipeline.separation.chunk=60.0 \
    pipeline.separation.num_steps=30 \
    data.source=youtube
```

**Ý nghĩa các cờ:**
- `step=separate_dialogue`: Chạy pha tách kênh thoại với DuplexChat.
- `--chunk 60` / `pipeline.separation.chunk=60.0`: Cửa sổ cắt nhỏ âm thanh khi tách (giây, mặc định 60s).
- `--workers 2`: Tách song song 2 folder thoại cùng lúc (~12-16GB VRAM).
- `pipeline.separation.num_steps=30`: Số bước khử nhiễu (diffusion sampling steps).

---

#### Bước 2B: Tách kênh âm thanh bằng Sommelier (SepReformer)
Phát hiện vùng overlap, tách âm chồng lấn bằng SepReformer và tái tạo lại toàn bộ file stereo 24kHz.

```bash
# Bằng Shell Script:
bash sommelier.sh --youtube --gpu 0 --workers 2         # Tách thoại Sommelier (2 workers)
bash sommelier.sh --podcast-index --gpu 1 --workers 2
bash sommelier.sh --all --gpu 0 --workers 2
bash sommelier.sh --youtube --dry-run                   # Chạy thử

# Hoặc bằng Python (Hydra):
python run_pipeline.py \
    step=sommelier \
    pipeline=sommelier \
    env=sever \
    gpu=0 \
    optimization.workers=2 \
    data.source=youtube
```

**Ý nghĩa các cờ:**
- `step=sommelier`: Chạy pha tách và tái tạo âm thoại của Sommelier.
- `--workers 2`: Tách song song 2 folder thoại cùng lúc.
- `pipeline=sommelier`: Tự động nạp đường dẫn checkpoint `SEPREFORMER` từ config.

---

#### Bước 3: Đánh giá Chất lượng Stereo (Stereo Benchmark)
Đo đạc chất lượng âm thanh tham chiếu tự do: DNSMOS, NISQA, ITD/ITC (Speaker Distinctiveness & Consistency), Turn Dynamics và sinh báo cáo tỷ lệ giữ lại dữ liệu.

```bash
# Bằng Shell Script:
bash benchmark.sh --youtube --workers 4                 # Đánh giá DuplexChat YouTube (4 workers)
bash benchmark.sh --podcast-index --workers 4           # Đánh giá Podcast Index
bash benchmark.sh --sommelier --all --workers 4         # Đánh giá Sommelier cả hai nguồn
bash benchmark.sh --youtube --gpu 0 --workers 4         # Chỉ định GPU 0

# Hoặc bằng Python (Hydra):
python run_pipeline.py \
    step=benchmark \
    pipeline=duplexchat \
    env=sever \
    gpu=0 \
    optimization.workers=4 \
    benchmark.check_models=true \
    data.source=youtube
```

**Ý nghĩa các cờ:**
- `step=benchmark`: Chạy benchmark đánh giá âm thanh đầu ra stereo và xuất báo cáo.
- `--workers 4`: Đo song song 4 file audio cùng lúc.
- `benchmark.check_models=true`: Kiểm tra đầy đủ các checkpoint (DNSMOS ONNX, NISQA, SQUIM) trước khi đo.

---

## 3. Cách Thay đổi Tham số với Hydra

### 3.1. Ghi đè tham số trực tiếp từ CLI (`key=value`)

| Nhu cầu thay đổi | Lệnh ghi đè (CLI Flag) |
| :--- | :--- |
| **Đổi GPU** | `gpu=0` hoặc cờ `--gpu 0` |
| **Số worker GPU (A100 40GB)** | `optimization.workers=4` hoặc cờ `--workers 4` / `-w 4` |
| **Số worker Convert (CPU)** | `optimization.convert_workers=8` hoặc cờ `--workers 8` |
| **Đổi nguồn dữ liệu** | `data.source=podcast_index` hoặc cờ `--podcast-index` |
| **Đổi độ dài chunk tách** | `pipeline.separation.chunk=90.0` hoặc cờ `--chunk 90` |
| **Tắt lọc nhạc nền** | `pipeline.music_filter.enabled=false` |
| **Đổi backend Diarization** | `diarization=pyannote` hoặc `diarization=sortformer` |
| **Chuyển môi trường dev/local** | `env=dev` hoặc cờ `--dev` |
| **Chuyển môi trường server** | `env=sever` hoặc cờ `--sever` |
| **Đổi đường dẫn SepReformer** | `env.paths.sepreformer=/path/to/custom/epoch.pth` |
| **Tùy biến thư mục dữ liệu** | `env.paths.base_data=/my/custom/path/data` |
| **Tùy biến thư mục model** | `env.paths.base_models=/my/custom/path/models` |

---

### 3.2. Cơ chế Tối ưu Đa luồng (GPU A100 40GB) & Giới hạn CPU 1-Thread

Hệ thống được thiết kế đặc thù để khai thác tối đa năng lực tính toán của **GPU A100 40GB** trong khi bảo vệ CPU không bao giờ bị nghẽn (thrashing / 100% saturation):

1. **Khống chế CPU không vượt quá 100% per process (1 thread)**:
   - Module `core.runtime_cpu` tự động thiết lập cố định `1 thread` cho toàn bộ các thư viện toán học nền tảng:
     ```text
     OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, 
     VECLIB_MAXIMUM_THREADS=1, NUMEXPR_NUM_THREADS=1, TORCH_NUM_THREADS=1
     torch.set_num_threads(1), torch.set_num_interop_threads(1)
     ffmpeg -threads 1
     ```
   - Đảm bảo mỗi tiến trình worker chỉ sử dụng đúng 1 CPU core, không sinh đa luồng ngầm làm treo CPU server.

2. **Tận dụng tối đa VRAM GPU A100 40GB**:
   - **Convert Audio**: Chạy song song đa tiến trình (`--workers 8` hoặc `16`), mỗi worker gọi ffmpeg với `-threads 1`.
   - **Split Dialogue**: Chạy song song 2 đến 4 audio stream (`--workers 2` hoặc `4`). Mỗi worker chiếm ~5GB VRAM (Streaming Sortformer + Demucs + Whisper LID), tổng chỉ ~15–20GB / 40GB VRAM, tăng thông lượng gấp 2–4 lần.
   - **DuplexChat Separation**: Chạy song song các folder đối thoại (`--workers 2`), DialogueSidon chỉ tốn ~6GB VRAM per worker.
   - **Sommelier**: Chạy song song SepReformer (`--workers 2`), chỉ tốn ~5GB VRAM per worker.
   - **Stereo Benchmark**: Đo đạc song song hàng loạt file WAV (`--workers 4`).

**Ví dụ kết hợp ghi đè:**
```bash
bash dialogue_split.sh --youtube --gpu 0 pipeline.music_filter.enabled=false
bash duplexchat.sh --youtube --gpu 1 --chunk 90
```

---

### 3.2. Chế độ Chạy Thử (Dry-Run)
Kiểm tra chính xác câu lệnh và đường dẫn sẽ được thực thi mà không ghi đè dữ liệu:
```bash
bash run_pipeline.sh --all --dry-run
# hoặc
python run_pipeline.py step=all dry_run=true
```

---

### 3.3. Xem Cấu hình đã Resolve trước khi chạy
In toàn bộ cây cấu hình đã kết hợp đầy đủ các nhóm defaults và biến môi trường:
```bash
python run_pipeline.py --cfg job
```

Xem trợ giúp các nhóm cấu hình có sẵn của Hydra:
```bash
python run_pipeline.py --help
```

---

### 3.4. Cấu trúc thư mục `configs/`
Khi cần thay đổi cấu hình mặc định lâu dài, bạn có thể chỉnh sửa trực tiếp các file YAML:
```text
configs/
├── config.yaml               # Cấu hình gốc kết nối các modules
├── env/
│   ├── sever.yaml            # Đường dẫn cụm server offline (/storage-voice/...) & SEPREFORMER
│   └── dev.yaml              # Đường dẫn local & tải online HuggingFace
├── pipeline/
│   ├── duplexchat.yaml       # Tham số riêng của DuplexChat (chunk, num_steps, LID, music)
│   ├── sommelier.yaml        # Tham số riêng của Sommelier (overlap threshold, sample rate)
│   └── cholimex.yaml         # Cấu hình VAD refinement Cholimex
├── diarization/
│   ├── sortformer.yaml       # Cấu hình NVIDIA Streaming Sortformer v2.1
│   └── pyannote.yaml         # Cấu hình Pyannote Diarization Community 1
├── separation/
│   ├── dialoguesidon.yaml    # Cấu hình tách DialogueSidon 24kHz
│   └── sepreformer.yaml      # Cấu hình tách SepReformer
└── benchmark/
    └── default.yaml          # Cấu hình đo lường DNSMOS, NISQA, Data retention
```
