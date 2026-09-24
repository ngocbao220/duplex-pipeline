# Duplex-Pipelines

Pipeline xử lý cuộc thoại hai người gồm DuplexChat (DialogueSidon), Sommelier (SepReformer) và Reference-Free Stereo Benchmark. Điểm vào duy nhất là `run_pipeline.py`; mọi lựa chọn chạy được truyền bằng Hydra `key=value`.

## Chuẩn bị

Chạy từ thư mục gốc dự án với Python 3.12. Mỗi pipeline dùng môi trường riêng: DuplexChat và benchmark dùng `requirements-duplexchat.txt`; Sommelier dùng `requirements-sommelier.txt`. Không cài hai file này trong cùng một môi trường vì chúng dùng các bộ Torch/NeMo khác nhau.

```text
python -m pip install --upgrade pip setuptools wheel
python -m pip install --prefer-binary -r requirements-duplexchat.txt
python -m pip check
```

Để cài Sommelier, thay file requirements trong lệnh thứ hai bằng `requirements-sommelier.txt`. Hai file đã chốt ONNX có wheel Python 3.12, tránh pip backtrack về ONNX 1.10 và cố build source. `env=sever` là chế độ offline, dùng đường dẫn model/dữ liệu cục bộ trong `configs/env/sever.yaml`. `env=dev` cho phép tải model từ Hugging Face theo các đường dẫn ở `configs/env/dev.yaml`.

Để tạo bản đồ chủ đề từ transcript, cài thêm các phụ thuộc tùy chọn trong cùng môi trường DuplexChat:

```text
python -m pip install -r requirements-topic-map.txt
```

## Lệnh chạy

Mẫu chung:

```text
python run_pipeline.py step=<step> pipeline=<pipeline> env=<env> gpu=<gpu> data.source=<source> <override>=<value>
```

`step` nhận một trong các giá trị `convert`, `split_dialogue`, `separate_dialogue`, `sommelier`, `cholimex`, `benchmark`, `topic_map`, hoặc `all`. `pipeline` có đúng ba lựa chọn: `duplexchat`, `sommelier`, `cholimex`; `source` là `youtube` hoặc `podcast_index`.

### Chạy toàn bộ pipeline

```text
python run_pipeline.py step=all pipeline=duplexchat env=sever gpu=0 optimization.workers=2 data.source=youtube
python run_pipeline.py step=all pipeline=duplexchat env=sever gpu=1 optimization.workers=2 data.source=podcast_index
python run_pipeline.py step=all pipeline=sommelier env=sever gpu=0 optimization.workers=2 data.source=youtube
python run_pipeline.py step=all pipeline=cholimex env=sever gpu=0 optimization.workers=2 data.source=youtube
```

Để chạy cả hai nguồn, gọi một lệnh cho mỗi `data.source`; Hydra không có giá trị `all` cho trường này.

### Chạy từng pha

Chuyển đổi crawl sang WAV:

```text
python run_pipeline.py step=convert env=sever optimization.convert_workers=8
```

Lọc dialogue hai người:

```text
python run_pipeline.py step=split_dialogue pipeline=duplexchat env=sever gpu=0 data.source=youtube optimization.workers=4 diarization=sortformer dialogue_split.music_filter.enabled=true dialogue_split.lid.enabled=true dialogue_split.lid.code=vi
```

Sau batch, xem `split_dialogue_report.json` ngay trong `data.dialogue_dir`: báo cáo ghi LID đang bật hay tắt, model/ngưỡng xác suất, tổng candidate/clip xuất ra, số clip bị loại theo từng lý do, và lý do của mọi audio không tạo được clip. Phần `retention` cho biết tổng giờ audio gốc, số giờ và phần trăm còn lại sau lọc dialogue và sau LID; terminal cũng in phase làm giảm thời lượng nhiều nhất. Các phần trăm retention tính trên tổng audio gốc. Audio không có manifest được tính vào tổng nguồn nhưng không gán phần thời lượng thiếu cho bộ lọc nào; xem `unprocessed_source_hours`. Mỗi `<audio>/manifest.json` có `candidate_dialogues` để truy ngược timestamp và quyết định của từng candidate.

### Bản đồ phân bố transcript

Đặt mỗi transcript dialogue thành một file UTF-8 `.txt` trong `transcripts/`. Tên file (không gồm `.txt`) được dùng làm ID; dưới đây là ví dụ một file mỗi dialogue.

Tải snapshot model một lần, giữ bản `model.safetensors` để không tải thêm trọng số trùng `pytorch_model.bin`:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="bkai-foundation-models/vietnamese-bi-encoder",
    local_dir="/storage-voice/voice/vdt/baottn/duplex-model-dir/vietnamese-bi-encoder",
    ignore_patterns=["pytorch_model.bin"],
)
```

Mặc định `env=sever` dùng đường dẫn model từ `configs/env/sever.yaml`; `env=dev` dùng đường dẫn từ `configs/env/dev.yaml`. `topic_map.model_dir` phải trỏ tới thư mục snapshot local; bước này không fallback sang tải model từ Hub khi thiếu file. Chạy t-SNE:

```text
python run_pipeline.py step=topic_map env=sever
```

Có thể đổi các đường dẫn bằng Hydra overrides, ví dụ `topic_map.input_dir=transcripts`, `topic_map.model_dir=/models/vietnamese-bi-encoder`, `topic_map.output_dir=outputs/topic-map`, hoặc `topic_map.device=cpu`. Kết quả gồm `transcript_tsne.png` và `transcript_tsne.csv` trong thư mục output. Cần tối thiểu ba transcript không rỗng; các file hiện có trong `transcripts/` là ví dụ ngắn, chưa đủ để tạo bản đồ có ý nghĩa. Model card yêu cầu word-segment tiếng Việt; bước này tự dùng PyVi. Transcript dài được chia thành các khúc không vượt quá 256 token rồi gộp thành một embedding cho mỗi dialogue. Bản đồ t-SNE dùng để khám phá cụm/phân bố nội dung, không phải phép đo độ phủ chủ đề tuyệt đối.

Nguồn triển khai: [Vietnamese Bi-Encoder model card](https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder) (word segmentation, mean pooling, sequence length 256), [PyVi](https://pypi.org/project/pyvi/) (Vietnamese tokenizer), [Transformers `from_pretrained`](https://huggingface.co/docs/transformers/main_classes/model) (local-only loading), và [scikit-learn TSNE](https://scikit-learn.org/1.4/modules/generated/sklearn.manifold.TSNE.html).

Tách bằng DuplexChat:

```text
python run_pipeline.py step=separate_dialogue pipeline=duplexchat env=sever gpu=0 data.source=youtube optimization.workers=2 pipeline.separation.chunk=60.0 pipeline.separation.num_steps=30
```

Tách và tái tạo bằng Sommelier:

```text
python run_pipeline.py step=sommelier pipeline=sommelier env=sever gpu=0 data.source=youtube optimization.workers=2
```

Refine stereo của DuplexChat bằng Cholimex. Mỗi `stereo_N.wav` được ghép với `dialogue_N.wav` tương ứng; thiếu một trong hai file sẽ được báo và bỏ qua.

```text
python run_pipeline.py step=cholimex pipeline=cholimex env=sever gpu=0 data.source=youtube optimization.workers=2
```

Đánh giá đầu ra stereo:

```text
python run_pipeline.py step=benchmark pipeline=duplexchat env=sever gpu=0 data.source=youtube optimization.workers=4 benchmark.check_models=true
python run_pipeline.py step=benchmark pipeline=sommelier env=sever gpu=0 data.source=youtube optimization.workers=4 benchmark.check_models=true
```

## Hydra overrides thường dùng

| Nhu cầu | Override Hydra |
| --- | --- |
| Chọn GPU | `gpu=0` |
| Chọn nhiều GPU | `gpu=0,1` |
| Worker GPU | `optimization.workers=4` |
| Auto-tune GPU workers | `optimization.resource_mode=auto` |
| VRAM reserve cho auto-tune | `optimization.gpu_memory_reserve_ratio=0.10` |
| Worker/GPU tối đa | `optimization.max_workers_per_gpu=8` |
| Worker convert CPU | `optimization.convert_workers=8` |
| Chọn nguồn | `data.source=podcast_index` |
| Chọn pipeline | `pipeline=duplexchat`, `pipeline=sommelier` hoặc `pipeline=cholimex` |
| Chuyển môi trường | `env=dev` hoặc `env=sever` |
| Đổi chunk DialogueSidon | `pipeline.separation.chunk=90.0` |
| Đổi diffusion steps | `pipeline.separation.num_steps=50` |
| Tắt lọc nhạc | `dialogue_split.music_filter.enabled=false` |
| Tắt LID | `dialogue_split.lid.enabled=false` |
| Đổi ngôn ngữ LID | `dialogue_split.lid.code=en` |
| Đổi ngưỡng LID | `dialogue_split.lid.min_prob=0.7` |
| Đổi backend diarization của bước lọc | `dialogue_split.diarization.backend=pyannote` |
| Đổi model diarization của bước lọc | `dialogue_split.diarization.model=pyannote/speaker-diarization-community-1` |
| Đặt đường dẫn model | `env.paths.base_models=/path/to/models` |
| Đặt đường dẫn data | `env.paths.base_data=/path/to/data` |
| Đặt checkpoint SepReformer | `env.paths.sepreformer=/path/to/epoch.pth` |
| Chạy thử, không ghi dữ liệu | `dry_run=true` |

Ví dụ kết hợp override:

```text
python run_pipeline.py step=split_dialogue pipeline=duplexchat env=sever gpu=0 data.source=youtube optimization.workers=4 dialogue_split.music_filter.enabled=false
python run_pipeline.py step=separate_dialogue pipeline=duplexchat env=sever gpu=1 data.source=youtube pipeline.separation.chunk=90.0
python run_pipeline.py step=all pipeline=duplexchat env=sever data.source=youtube dry_run=true
```

## Kiểm tra cấu hình trước khi chạy

```text
python run_pipeline.py --cfg job
python run_pipeline.py --help
```

## Cấu hình mặc định

Hydra ghép `configs/config.yaml` với các config group sau:

```text
configs/
├── config.yaml
├── env/
│   ├── sever.yaml
│   └── dev.yaml
├── pipeline/
│   ├── duplexchat.yaml
│   ├── sommelier.yaml
│   └── cholimex.yaml
├── dialogue_split/default.yaml
├── diarization/
│   ├── sortformer.yaml
│   └── pyannote.yaml
├── separation/
│   ├── dialoguesidon.yaml
│   └── sepreformer.yaml
└── benchmark/default.yaml
```

Sửa YAML khi muốn đổi mặc định lâu dài; dùng override trên lệnh Python cho từng lần chạy.
