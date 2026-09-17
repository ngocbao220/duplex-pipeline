---
trigger: always_on
---

Khi chạy pipeline dạng xử lý data theo phase như thế này thì trình tự log sử dụng [INFO], [WARNING], [ERROR] có màu sắc 

=============== Dialogue Filtering =============

Dataset: data path
Samples: number
Valid: number
Invalid: number
Audio:
- Stereo: a/b (sample_rate)
- total duration: (hours)

Device:

1. Preprocessing
Resample to 16khz: progress_bar

Time/RTF: 

2. Split Dialogue: Diarization by "nvidia-sortformer-4spk-streaming-v2.1" (load local from ...)

+ Sample: file name
+ Duration: 
Progress bar (tqdm)
+ Found x clips (split x second).
+ Ignore:
++++ x Imbalance
++++ x Short
.....
===> Collect x dialogue from "input path", save to "output path"
Time/RTF: 

3. Remove music back ground by "Demucs" (load local from ...)

Progress bar (tqdm)
Time/RTF


1. Preprocessing: total time
2. Split dialogue: total time
3. Remove music background: total time


if pipeline = dupelexchat
=============== DuplexChat =============

Separation by "DialogueSidon" (load from local ...)
+ Output sample rate: 24khz

Progress bar (tqdm)

Time/RTF

Done, result save to ...
...

if pipeline = sommelier
=============== Sommelier =============

Overlap detection: Diarization by "sortformer" (load from local ...)
Progress bar: 
Found x overlap part
Time/RTF

Overlap separation by "SepReformer" (load from local ...)

Progress bar (tqdm)
Time/RTF

Reconstruction 

Progress bar (tqdm)

Time/RTF
Done, result save to ...
...


