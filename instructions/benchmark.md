Hãy đọc codebase hiện tại trước khi sửa code. Xác định cấu trúc pipeline, convention về config, CLI, logging, dependency management và output directory hiện có. Sau đó triển khai một module benchmark dành cho **speaker-separated full-duplex stereo audio**.

## 1. Mục tiêu

Benchmark phải cho phép chạy trực tiếp trên **một file audio stereo duy nhất**:

```bash
uv run python ... --audio path/to/output.wav
```

Input:

```text
stereo.wav
├── Left channel  = estimated speaker A
└── Right channel = estimated speaker B
```

Không yêu cầu clean ground-truth.

Benchmark nhằm trả lời 4 câu hỏi:

1. Chất lượng âm thanh từng channel có tốt không?
2. Mỗi channel có giữ nhất quán một speaker không?
3. Left và Right có thực sự là hai speaker khác nhau không?
4. Audio stereo có giữ được các đặc trưng full-duplex như overlap, turn-taking và backchannel không?

Tham khảo cách đánh giá của paper DuplexChat, nhưng implementation phải độc lập, modular và có thể tích hợp vào pipeline hiện tại.

---

# 2. Benchmark groups

Triển khai benchmark theo các nhóm sau.

## A. Input validation

Khi nhận một audio:

* kiểm tra file tồn tại;
* kiểm tra đọc được audio;
* kiểm tra số channel;
* benchmark này yêu cầu stereo 2 channels;
* nếu mono hoặc >2 channels thì báo lỗi rõ ràng;
* lấy:

  * sample rate;
  * duration;
  * number of samples;
  * channel count.

Tách thành:

```python
left_audio
right_audio
```

Không được mix stereo về mono trước khi benchmark.

Nếu model metric yêu cầu sample rate khác, resample trên một copy riêng cho model đó. Không thay đổi file gốc.

---

# 3. Acoustic quality metrics

## 3.1 DNSMOS

Tính DNSMOS độc lập cho:

```text
Left
Right
```

Nếu implementation DNSMOS hỗ trợ các sub-score như:

* SIG
* BAK
* OVRL

thì lưu đầy đủ.

Output ví dụ:

```json
{
  "dnsmos": {
    "left": {
      "sig": ...,
      "bak": ...,
      "ovrl": ...
    },
    "right": {
      "sig": ...,
      "bak": ...,
      "ovrl": ...
    },
    "mean_ovrl": ...
  }
}
```

DNSMOS là reference-free nên không cần ground-truth.

Nếu package/model không thể tự động tải hoặc không khả dụng trong environment hiện tại:

* không được làm toàn benchmark crash;
* đánh dấu metric là unavailable;
* log nguyên nhân rõ ràng.

---

## 3.2 SQUIM

Dùng `torchaudio` SQUIM objective model nếu phù hợp với version đang dùng.

Tính reference-free predicted:

* STOI
* PESQ
* SI-SDR nếu model SQUIM thực sự cung cấp output đó.

Phải gọi đúng tên:

```text
SQ-STOI
SQ-PESQ
SQ-SI-SDR
```

để tránh nhầm với STOI/PESQ/SI-SDR truyền thống cần clean reference.

Tính độc lập trên:

```text
Left
Right
```

và báo mean hai channel.

Ví dụ:

```json
{
  "squim": {
    "left": {
      "stoi": ...,
      "pesq": ...,
      "si_sdr": ...
    },
    "right": {
      ...
    }
  }
}
```

Quan trọng:

Không implement STOI/PESQ truyền thống rồi truyền chính prediction làm reference.

---

# 4. Speaker identity metrics

Đây là phần quan trọng nhất.

Sử dụng một pretrained speaker embedding model ổn định, ưu tiên:

```text
speechbrain/spkrec-ecapa-voxceleb
```

hoặc implementation tương đương nếu project đã có speaker encoder.

Không load model lại nhiều lần.

---

## 4.1 ITC — Intra-Track Consistency

Mục đích:

```text
Một channel có giữ cùng speaker identity xuyên suốt clip không?
```

Cho từng channel:

1. Dùng VAD để tìm speech regions.
2. Chỉ lấy vùng có speech để extract speaker embedding.
3. Chia các speech regions thành các window hợp lý, ví dụ khoảng 2–3 giây speech.
4. Bỏ window quá ngắn để speaker embedding không ổn định.
5. Extract embedding từng window:

```text
e1, e2, ..., en
```

6. Tính speaker consistency bằng cosine similarity.

Có thể implement:

```python
centroid = normalized_mean(embeddings)
itc = mean(cosine(e_i, centroid))
```

hoặc pairwise mean cosine nếu paper/code tham chiếu dùng cách đó.

Nếu có thể kiểm tra implementation gốc của ITC trong reference thì ưu tiên tái hiện đúng định nghĩa.

Báo:

```text
ITC Left
ITC Right
Mean ITC
# valid embedding windows
```

Range mong muốn càng cao càng tốt.

Phải xử lý trường hợp channel có quá ít speech:

```text
status = unavailable
reason = insufficient_speech
```

thay vì trả về score giả.

---

## 4.2 ITD — Inter-Track Distinctiveness

Mục đích:

```text
Left và Right có phải hai speaker khác nhau không?
```

Tạo một speaker embedding đại diện cho mỗi channel từ các speech regions hợp lệ:

```python
E_left
E_right
```

Sau đó:

```python
cos_sim = cosine(E_left, E_right)
ITD = 1 - cos_sim
```

Báo cả:

```text
inter_track_cosine_similarity
ITD
```

Interpretation:

```text
ITD cao  -> hai track distinct hơn
ITD thấp -> nguy cơ channel collapse / cùng speaker xuất hiện ở cả hai channel
```

Không hard-code một threshold "good/bad" nếu không có căn cứ.

Có thể cho phép threshold qua config, nhưng mặc định chỉ báo raw score.

---

# 5. Speech activity / full-duplex analysis

Dùng cùng một VAD cho Left và Right.

Ưu tiên model nhẹ và reproducible, ví dụ Silero VAD nếu codebase chưa có VAD phù hợp.

Đầu ra VAD phải được chuyển thành frame-level/time-interval speech mask:

```text
L(t) ∈ {0, 1}
R(t) ∈ {0, 1}
```

Từ đó định nghĩa:

```text
left_only  = L=1, R=0
right_only = L=0, R=1
overlap    = L=1, R=1
silence    = L=0, R=0
```

Báo tổng duration và percentage cho 4 trạng thái.

Ví dụ:

```text
Duration: 62.4 s

Left only:   21.3 s   34.1%
Right only:  18.8 s   30.1%
Overlap:      8.1 s   13.0%
Silence:     14.2 s   22.8%
```

---

# 6. Turn-taking metrics

Tái hiện logic gần với DuplexChat.

Sau VAD:

* merge speech segments cùng channel nếu gap giữa chúng < `0.5 s`;
* gọi các segment sau merge là `talk spurts`.

Các threshold phải nằm trong config/constants, không rải hard-code khắp code.

Ví dụ:

```yaml
turn_analysis:
  merge_gap_sec: 0.5
  min_turn_duration_sec: 1.0
  max_backchannel_duration_sec: 1.0
```

Nếu chưa sử dụng ASR thì version đầu tiên có thể xác định turn chỉ theo duration.

Không được giả vờ có word-count criterion nếu chưa thực sự chạy ASR.

Tính:

### Turn exchanges per minute

Số lần quyền nói chuyển:

```text
L -> R
R -> L
```

chia cho duration phút.

Phải xử lý overlap một cách nhất quán và document algorithm.

### Mean turn duration

Average duration của các turn hợp lệ.

Báo thêm riêng:

```text
mean_left_turn_duration
mean_right_turn_duration
```

nếu có ích.

### Simultaneous Speech %

```python
overlap_duration / total_duration * 100
```

Có thể báo thêm:

```python
overlap_duration / total_speech_union_duration * 100
```

nhưng phải đặt tên rõ để không nhầm hai denominator.

Primary metric theo paper:

```text
percentage of total time where both channels are active
```

---

# 7. Overlapping transitions

Tính tỷ lệ turn-taking events xảy ra với overlap.

Ví dụ:

```text
Speaker L đang nói
Speaker R bắt đầu nói trước khi L kết thúc
→ overlapping transition
```

Metric:

```python
overlapping_transition_rate =
    overlapping_turn_transitions / all_turn_transitions
```

Báo:

```text
# transitions
# overlapping transitions
overlapping transition %
```

Không tính overlap bất kỳ là transition nếu không có sự chuyển đổi speaker.

---

# 8. Backchannel analysis

Đây là phần rất quan trọng với full-duplex data.

Version không-ASR:

Một candidate backchannel là talk spurt:

* duration < `backchannel_max_duration`, mặc định khoảng `1.0 s`;
* xảy ra khi channel kia đang có một turn dài hơn;
* phần lớn candidate nằm bên trong activity của speaker kia.

Ví dụ:

```text
L: ───────────── speaking ─────────────
R:               ─ uh ─
```

→ R backchannel candidate.

Tính:

```text
backchannel_count_left
backchannel_count_right
backchannels_per_minute
mean_backchannel_duration
```

Quan trọng:

Tên metric phải là:

```text
estimated_backchannel
```

hoặc:

```text
VAD-based backchannel
```

nếu chưa chạy ASR.

Không khẳng định chắc chắn đó là linguistic backchannel chỉ dựa trên duration/VAD.

Thiết kế code để sau này dễ thêm ASR criterion:

```text
duration < 1s OR <= 3 words
```

giống cách DuplexChat phân tích.

---

# 9. Crosstalk / leakage proxy

Thêm một **reference-free diagnostic**, nhưng không gọi nó là ground-truth leakage metric.

Trong các vùng:

```text
Left speech = active
Right = inactive theo VAD
```

đo energy của Right so với Left.

Và ngược lại.

Có thể tính:

```python
leakage_db_L_to_R =
    10 * log10(E_right / (E_left + eps))
```

trong các left-only regions.

Tương tự:

```python
leakage_db_R_to_L
```

Báo median/mean, không chỉ một giá trị nếu có nhiều region.

Tên phải thể hiện đây là proxy:

```text
inactive_channel_energy_ratio_db
```

Không được gọi đây là SIR.

Có thể thêm correlation/cosine waveform diagnostic nếu thấy hợp lý, nhưng không dùng metric không có ý nghĩa khoa học rõ ràng.

---

# 10. Không sử dụng các metric cần GT

Với chế độ benchmark một stereo audio không có reference, KHÔNG chạy:

```text
SI-SDR
PIT-SI-SDR
SDR
SIR
SAR
STOI truyền thống
ESTOI truyền thống
PESQ truyền thống
POLQA
```

Trừ `SQ-*` được SQUIM dự đoán reference-free.

Nếu codebase hiện tại đang chạy các metric trên mà không có ground-truth, hãy sửa logic:

```python
if ground_truth is None:
    skip intrusive metrics
```

và giải thích rõ trong log.

---

# 11. CLI

Cần có một entry point đơn giản.

Ví dụ:

```bash
uv run python scripts/benchmark_stereo.py \
    --audio outputs/example.wav
```

Có thể hỗ trợ:

```bash
--device auto
--output-dir outputs/benchmark
--debug
--config configs/benchmark.yaml
```

`--device auto`:

```text
CUDA nếu available
MPS nếu implementation/model hỗ trợ ổn định
CPU fallback
```

Không assume CUDA.

---

# 12. Console output

Output phải dễ đọc.

Ví dụ:

```text
Pipeline: Stereo Full-Duplex Benchmark

============================================================
1. Input
============================================================

File       : sample.wav
Duration   : 64.23 s
Sample rate: 16000 Hz
Channels   : 2

Left  : Speaker track 1
Right : Speaker track 2

============================================================
2. Acoustic Quality
============================================================

Metric        Left      Right      Mean
DNSMOS OVRL   3.21      3.18       3.20
SQ-STOI       0.94      0.92       0.93
SQ-PESQ       3.01      2.95       2.98

============================================================
3. Speaker Separation
============================================================

ITC Left       : ...
ITC Right      : ...
Mean ITC       : ...
L/R cosine sim : ...
ITD            : ...

============================================================
4. Speech Activity
============================================================

State          Duration      Percentage
Left only      ...
Right only     ...
Overlap        ...
Silence        ...

============================================================
5. Conversation Dynamics
============================================================

Turn exchanges / min       : ...
Mean turn duration          : ...
Backchannels / min          : ...
Simultaneous speech         : ... %
Overlapping transitions     : ... %

============================================================
6. Leakage Diagnostics
============================================================

L -> R inactive-channel energy ratio : ... dB
R -> L inactive-channel energy ratio : ... dB

============================================================
Benchmark complete
============================================================

JSON report:
<path>

Detailed segments:
<path>
```

Nếu metric unavailable:

```text
DNSMOS    N/A    dependency/model unavailable
```

Không crash toàn chương trình.

---

# 13. Machine-readable output

Lưu một JSON chứa đầy đủ kết quả.

Ví dụ structure:

```json
{
  "input": {},
  "acoustic_quality": {},
  "speaker_identity": {},
  "speech_activity": {},
  "turn_taking": {},
  "backchannel": {},
  "leakage_proxy": {},
  "runtime": {},
  "versions": {}
}
```

Lưu thêm detailed timeline dưới JSON hoặc JSONL:

```json
{
  "left_vad": [],
  "right_vad": [],
  "turns": [],
  "overlaps": [],
  "backchannel_candidates": []
}
```

Mỗi event:

```json
{
  "start": 12.32,
  "end": 12.81,
  "duration": 0.49,
  "speaker": "right",
  "type": "backchannel_candidate"
}
```

---

# 14. Optional debug artifacts

Khi dùng:

```bash
--debug
```

xuất thêm:

```text
benchmark/
├── report.json
├── timeline.json
├── left.wav
├── right.wav
├── vad_left.json
├── vad_right.json
├── turns.json
├── overlap.json
└── backchannels.json
```

Nếu dễ implement, tạo thêm một timeline visualization:

```text
time →
L: ███████       ███████████
R:       ██  ███      ██
         BC           overlap
```

Có thể lưu PNG, nhưng đây là optional.

---

# 15. Architecture

Không viết tất cả vào một file lớn.

Tách module hợp lý, ví dụ:

```text
benchmark/
├── __init__.py
├── runner.py
├── audio.py
├── acoustic.py
├── speaker.py
├── activity.py
├── turn_taking.py
├── leakage.py
├── report.py
└── schemas.py
```

Hoặc tuân theo architecture hiện tại của repo nếu repo đã có convention khác.

Ưu tiên reuse:

* audio loader;
* resampler;
* VAD;
* speaker embedding;
* logging;
* config;

nếu project đã có implementation.

Không duplicate model loading.

---

# 16. Reproducibility

Report phải lưu:

```text
benchmark version
timestamp
input path
input checksum nếu dễ làm
sample rate
duration
device
model names
model revisions nếu available
library versions
threshold/config values
```

Điều này quan trọng để benchmark giữa các pipeline version có thể so sánh lại được.

---

# 17. Tests

Viết focused tests ít nhất cho:

### Test 1

Mono input → reject với error message rõ ràng.

### Test 2

Stereo input → split L/R chính xác.

### Test 3

Speech activity masks → tính đúng:

```text
left_only
right_only
overlap
silence
```

trên toy masks.

### Test 4

ITD formula:

```python
ITD = 1 - cosine_similarity
```

### Test 5

Turn transition detection.

### Test 6

Overlapping transition detection.

### Test 7

Backchannel candidate detection.

### Test 8

Metric unavailable → benchmark vẫn hoàn thành và report `N/A`.

Không cần download model lớn trong unit test. Mock model inference.

---

# 18. Important implementation rules

1. Không thay đổi behavior của pipeline chính ngoài phần cần thiết để tích hợp benchmark.
2. Không fake metric.
3. Không dùng prediction làm ground-truth.
4. Không gọi SQUIM output là STOI/PESQ chuẩn; dùng `SQ-STOI`, `SQ-PESQ`.
5. Không gọi inactive-channel energy ratio là SIR.
6. Backchannel dựa trên VAD phải ghi rõ là `candidate`/`estimated`.
7. Các model phải load lazy và cache/reuse.
8. Benchmark vẫn phải chạy nếu một optional metric dependency thiếu.
9. Tất cả threshold phải centralized/configurable.
10. Mọi metric phải có docstring nói:

* đo cái gì;
* higher/lower is better nếu xác định được;
* có cần GT không.

---

# 19. Primary report

Ở cuối benchmark, tạo một bảng summary ngắn:

```text
┌─────────────────────┬──────────┬────────────────────────────────┐
│ Metric              │ Value    │ Meaning                        │
├─────────────────────┼──────────┼────────────────────────────────┤
│ DNSMOS              │ ...      │ Acoustic quality               │
│ SQ-STOI             │ ...      │ Estimated intelligibility      │
│ SQ-PESQ             │ ...      │ Estimated perceptual quality   │
│ ITC                 │ ...      │ Intra-track speaker consistency│
│ ITD                 │ ...      │ Inter-track distinctiveness    │
│ Overlap             │ ... %    │ Simultaneous speech            │
│ Backchannel         │ ... /min │ Estimated short responses      │
│ Turn Exchange       │ ... /min │ Speaker turn dynamics          │
│ Overlap Transition  │ ... %    │ Turns initiated during overlap │
│ Leakage Proxy       │ ... dB   │ Inactive-channel energy        │
└─────────────────────┴──────────┴────────────────────────────────┘
```

Không tạo một arbitrary aggregate `"overall score"` trừ khi có công thức được justify rõ ràng.

---

# 20. Deliverables

Sau khi code xong:

1. Liệt kê file đã thêm/sửa.
2. Mô tả architecture.
3. Liệt kê dependency mới.
4. Cho lệnh chạy benchmark trên một stereo audio.
5. Cho ví dụ output.
6. Chạy unit tests.
7. Chạy smoke test trên một stereo file có sẵn trong repo nếu có.
8. Nêu rõ metric nào:

   * chạy được;
   * unavailable;
   * vì sao unavailable.
9. Không dừng ở việc viết plan — hãy implement hoàn chỉnh.

Mục tiêu cuối cùng:

```bash
uv run python scripts/benchmark_stereo.py --audio stereo.wav
```

phải đủ để đánh giá một sample stereo speaker-separated mà **không cần bất kỳ ground-truth nào**.
