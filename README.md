# duplex-pipelines

Chạy các pipeline tách hai speaker trên cùng audio mono: `vilier`,
`cholimex`, `duplexchat`, và bản gốc vendor `sommelier`. Mỗi pipeline có môi
trường `uv` riêng để dependency không xung đột.

## Cài đặt

```bash
UV_CACHE_DIR=.uv-cache uv sync --project pipeline/vilier
UV_CACHE_DIR=.uv-cache uv sync --project pipeline/cholimex
UV_CACHE_DIR=.uv-cache uv sync --project pipeline/duplexchat
UV_CACHE_DIR=.uv-cache uv sync --project pipeline/sommelier
```

Để chạy diarization bằng NVIDIA Streaming Sortformer, cài thêm thư viện hệ
thống và NeMo trong môi trường DuplexChat:

```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython packaging
pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]
```

Trong môi trường `uv`, có thể dùng lệnh tương đương sau khi đã sync project:

```bash
uv pip install Cython packaging
uv pip install 'nemo_toolkit[asr]'
```

Vilier và Sommelier cần checkpoint SepReformer thật (`.pth`/`.pt`, không phải
Git-LFS pointer):

```bash
export VILIER_SEPREFORMER_CHECKPOINT=/path/to/epoch.0180.pth
```

Sommelier và DuplexChat cần `HF_TOKEN` hoặc `HUGGINGFACE_TOKEN` để tải pyannote và các mô hình trên Hugging Face Hub:

```bash
export HF_TOKEN="your_huggingface_token"
```

## Tải trước các model cần thiết cho DuplexChat

DuplexChat trải qua 3 giai đoạn xử lý chính (Preprocess -> Diarization -> Dialogue Separation). Để tránh bị nghẽn mạng hay timeout khi đang xử lý audio dài, bạn có thể **tải trước toàn bộ model** vào cache bằng script tự động:

```bash
UV_CACHE_DIR=.uv-cache uv run --project pipeline/duplexchat python tools/download_duplexchat_models.py --include-demucs
```

### Danh sách mô hình được tải trước theo từng Phase:
1. **Phase 02 (Speaker Diarization & Linking)**:
   - **Silero VAD** (`snakers4/silero-vad`): Phát hiện khoảng có tiếng nói và cắt chunk im lặng.
   - **SpeechBrain ECAPA-TDNN** (`speechbrain/spkrec-ecapa-voxceleb`): Trích xuất vector đặc trưng giọng nói đại diện để ghép nhãn speaker toàn cục (Global Speaker Linking).
   - **Pyannote Diarization** (`pyannote/speaker-diarization-community-1`): Mô hình phân đoạn người nói.
   - **Demucs Music Filter** (`htdemucs` - tùy chọn): Loại bỏ nhạc nền nếu bật `--filter-music`.
2. **Phase 03 (Dialogue Separation)**:
   - **DialogueSidon** (`sarulab-speech/DialogueSidon`): Tách âm thanh cuộc thoại 2 người nói thành kênh stereo (`ssl_encoder.pt2`, `diffusion_head.pt2`, `vae_decoder.pt2`).

### DialogueSidon local trên server offline

Tại máy giữ bundle gốc đáng tin cậy, tạo `model_manifest.json`; copy manifest cùng bốn artifact sang server. Không tạo lại manifest sau khi copy lên server, vì manifest phải phát hiện file bị hỏng/truncated trong lúc chuyển.

```bash
python tools/create_dialoguesidon_manifest.py --model-dir /trusted/DialogueSidon
python -m duplexchat validate_model --separation-model /raid/models/DialogueSidon
python scripts/run_pipeline_batch.py --step separate_dialogue \
  --input-dir data/processed/dialogue/youtube --output-dir data/processed/separated/youtube \
  --gpu 2 --separation-model /raid/models/DialogueSidon
```

### Theo dõi tốc độ batch pipeline

Mỗi batch chạy thật ghi `outputs/pipeline_timing.json`: tổng thời gian, thời
gian từng item, exit code, RTF và tốc độ audio/giây nếu đọc được metadata WAV.
File cũng sắp xếp `slowest_phases` và `slowest_items` để xác định điểm nghẽn.

`run_pipeline.sh` tự đặt `PIPELINE_RUN_ID`, nên các phase batch của cùng một
workflow được gộp vào một report. Khi chạy từng lệnh thủ công, export cùng ID
trước các phase cần so sánh:

```bash
export PIPELINE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/run_pipeline_batch.py --step split_dialogue ...
python scripts/run_pipeline_batch.py --step separate_dialogue ...
```

Một workflow ID mới thay report cũ. `--dry-run` chỉ in command và không sửa
report timing.

---

## Chạy một audio

```bash
uv run --project pipeline/vilier python -m vilier single \
  --input mixture.wav --output-dir outputs/vilier --debug

uv run --project pipeline/cholimex python -m cholimex single \
  --input mixture.wav --output-dir outputs/cholimex --debug

uv run --project pipeline/duplexchat python -m duplexchat single \
  --input mixture.wav --output-dir outputs/duplexchat \
  --separation-chunk 60 --debug

uv run --project pipeline/sommelier python -m sommelier single \
  --input mixture.wav --output-dir outputs/sommelier --debug
```

Vilier, Cholimex và Sommelier ghi `audio.stereo.wav`, `run.json`, và
`benchmark.json` ngay trong `--output-dir`. Hai nguồn tách nằm lần lượt ở kênh
0 và 1; tên file không gán danh tính speaker.

DuplexChat mặc định diarize toàn episode, tự lấy các đoạn hội thoại hợp lệ có
đúng hai speaker, rồi tách từng đoạn. `--separation-chunk` mặc định là `120`
giây (overlap nội bộ là 10 giây). Kết quả là các WAV stereo 24 kHz theo thứ tự
conversation, không phải một WAV stereo cho toàn episode:

```text
outputs/duplexchat/
├── config.json
├── stereo_1.wav
├── stereo_2.wav
└── run.json
```

Không tìm được dialogue hợp lệ vẫn là một run thành công. Với `--debug`, các
pha có thể kiểm tra được được ghi tại `debug/`: `phase_00_input/{input.json,input.wav}`;
`phase_01_preprocess/audio_16k_mono.wav`; `phase_02_diarization/{speakers.txt,conversation.txt}`;
và `phase_03_separation/conversation_N/{left_N.wav,right_N.wav,stereo_N.wav}`.
Label text dùng định dạng `start<TAB>end<TAB>label`. `input.wav` luôn là PCM16
16 kHz và giữ số channel của nguồn; audio cho diarization/separation được downmix
thành mono 16 kHz trong phase 01.

Để phân phối các dialogue độc lập qua nhiều GPU, chỉ định danh sách GPU. GPU
đầu tiên chạy diarization; các task separation được xếp round-robin trên các
GPU được chỉ định:

```bash
uv run --project pipeline/duplexchat python -m duplexchat single \
  --input mixture.wav --output-dir outputs/duplexchat-gpu \
  --device-ids 0 1
```

### Cholimex refinement cho một conversation DuplexChat

Cholimex nhận trực tiếp một `stereo_N.wav` từ DuplexChat. Nó chạy Silero VAD
trên hai kênh: ở vùng chỉ một kênh active, chỉ giữ kênh đó; overlap giữ hai kênh;
silence được zero. Input không bị ghi đè và không dùng symlink.

```bash
uv run --project pipeline/cholimex python -m cholimex collection \
  --input outputs/duplexchat-easy1/stereo_1.wav \
  --output-dir outputs/cholimex-refined
```

Kết quả gồm `left.wav`, `right.wav`, `vad_left.txt`, `vad_right.txt`, và
`cholimex_stereo_1.wav` PCM16 24 kHz. Hai file VAD dùng format
`start<TAB>end<TAB>speech`.

Nếu input không có đúng hai speaker, Vilier/Sommelier fail thay vì tạo output giả.
Với `--debug`, Cholimex ghi từng overlap tại
`debug/overlaps/<id>/{mixture,audio.stereo}.wav`; Vilier giữ các native source
debug ở `debug/overlaps/<id>/{mixture,source_01,source_02}.wav`.

## Metrics và benchmark

Dự án áp dụng hệ thống **Reference-Free Stereo Benchmark** thống nhất cho các kết quả đầu ra âm thanh stereo:

```bash
uv run --project . python scripts/benchmark_stereo.py \
  --audio outputs/duplexchat/stereo_1.wav \
  --output-dir outputs/benchmark-one
```

Bộ metrics bao gồm:
- **Acoustic Quality**: DNSMOS (`ovrl`, `sig`, `bak`), NISQA-MOS (`nisqa_mos`), SQ-STOI, SQ-PESQ.
- **Speaker Identity**: ITC (Intra-track consistency), ITD (Inter-track distinctiveness).
- **Speech Activity & Turn Dynamics**: Overlap %, Backchannels/min, Turn Exchanges/min, Overlap Transition %, Leakage Proxy (dB).

## Smoke test sáu checkpoint separation

`tools/test_separation_models.py` chỉ tách audio, không diarization hay
reconstruction. Nó ghi native sources, elapsed time và RTF; **không thể tính
SI-SDR/PESQ** vì overlap WAV không có source reference.

```bash
git clone --depth 1 https://github.com/alibabasglab/MossFormer2 third_party/MossFormer2

UV_CACHE_DIR=.uv-cache uv run --project pipeline/vilier \
  --with 'diffusers>=0.30' --with 'asteroid>=0.7' --with 'rotary-embedding-torch' \
  python tools/test_separation_models.py \
  --overlaps-json compare/vi/vilier-1/debug/overlaps.json \
  --output-dir outputs/vilier-1-overlap-models \
  --models all --device cpu \
  --mossformer2-source third_party/MossFormer2
```

Model: SepFormer WSJ02Mix, SepFormer WHAMR, MossFormer2 LibriMix,
DialogueSidon, Rahma89 Conv-TasNet (native 3 source), và SepReformer Base
WSJ0. Output: `<output>/<model>/<overlap-id>/report.json` và WAV sources;
`<output>/summary.json` tổng hợp success/failure. Rahma89 không có
`audio.stereo.wav` vì checkpoint có ba source.

`MOSSFORMER2_SOURCE` có thể thay `--mossformer2-source`. Thêm
`SEPARATION_SMOKE_TRACEBACK=1` để ghi traceback vào report khi debug.

## Test code

```bash
UV_CACHE_DIR=.uv-cache uv run --project . --extra dev pytest -q
```
