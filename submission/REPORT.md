# Lab 21 — Fine-tuning LLMs: Evaluation Report

**Họ tên:** Phạm Quang Đạt

**MSSV:** 2A202602704

**Ngày:** 07/10/2026

**Tier:** T4 · **Base model:** `unsloth/Qwen3.5-4B`

**GPU:** Tesla T4, CUDA sm_75; runtime báo 14,6 GiB · **Precision thực tế:** fp16.

**Kết quả chính:** LoRA `correct` đạt target **0,9700**, hơn baseline prompt tối ưu **0,2050**, nhưng regression giảm **0,068889**, vượt ngưỡng **0,0200**. Phán quyết cuối cùng là **FAILED**; chưa nên thay base model dùng chung bằng adapter này.

## 1. Bài toán, lựa chọn và cấu hình

Tác vụ là phân loại ticket CSKH tiếng Việt thành JSON gồm `intent`, `urgency`, `product`, `sentiment`. Tôi dùng corpus tổng hợp mặc định của repo, sinh xác định bởi `scripts/make_seed_data.py`, để kiểm tra pipeline và các đối chứng trước khi thay dữ liệu. Nhãn rõ ràng cho phép chấm từng trường mà không cần LLM judge. Dữ liệu có cấu trúc lặp lại, nên kết quả chưa chứng minh khả năng xử lý ticket thực tế hoặc diễn đạt ngoài phân phối.

Tôi dùng model mặc định của tier T4 để phù hợp GPU Colab đã sử dụng và giữ cùng base model cho baseline lẫn bốn adapter. Không thay model, corpus hoặc prompt tối ưu giữa hai lượt đánh giá.

| Thành phần | Cấu hình / số đo |
|---|---|
| Corpus huấn luyện | 250 ticket |
| Train / validation | 225 / 25; split seed 42 |
| Eval đầy đủ | 50 target / 15 regression |
| `max_length` | 1024 theo cấu hình tier T4 |
| Độ dài token | mean 93,1; p50 93; p95 98; p99 100; max 101 |
| `suggested_max_length` | 256 |
| Loss mask | assistant-only |
| Batch / gradient accumulation | 1 / 16; batch hiệu dụng 16 |
| Ngân sách train | 2 epoch theo mặc định; cả bốn run ghi 30 optimizer step |
| Seed train | 42 theo cấu hình SFT |
| LoRA chính | text-linear, r=16, alpha=32, LR=0,0001 |
| Packing | Tắt để giữ căn chỉnh labels đã tokenize |

Nguồn: `results/token_stats.json`, `results/mask_proof.json`, `results/runs.csv`, `results/baselines_frozen.json`, `data/split/`, `src/labkit/config.py`, `src/labkit/train.py`.

Lượt train giữ `max_length=1024` mặc định thay vì điều chỉnh xuống 256 theo số đo. Cả 250 mẫu đều ngắn hơn giới hạn nên chưa có dấu hiệu bị cắt do độ dài. Đây là lựa chọn bảo thủ giữ cấu hình chung, chưa phải lựa chọn đã tối ưu theo p95. Tôi chưa chạy đối chứng 256/1024 nên không khẳng định 1024 cải thiện chất lượng hoặc tốc độ. Nếu lặp lại, tôi sẽ đặt 256 cho mọi run từ đầu và đo lại chi phí.

`results/log_lan2` báo Tesla T4 và precision fp16; CSV cũng ghi fp16. Không gọi đây là lượt train bf16. Log báo dùng kernel PyTorch tham chiếu do thiếu `causal_conv1d` và `flash-linear-attention`; latency dưới đây thuộc môi trường này.

## 2. Chat template và bằng chứng loss mask — NB1

`results/template_check.json` có `ok=true`, `open_tag_present=true`, `body_present=true`: mẫu thử giữ nội dung giữa `<think>` và `</think>`. Không cần sửa template để bảo toàn reasoning trong mẫu kiểm tra. Tuy nhiên, corpus mặc định chứa câu trả lời JSON thuần, không phải reasoning trace.

| Kiểm tra | Kết quả |
|---|---:|
| Token được giám sát | 39 / 94 |
| `supervised_fraction` | 0,4149 |
| `answer_is_supervised` | true |
| `question_is_masked` | true |

Đoạn được tính loss, chép từ `results/mask_proof.json`:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn này chứa câu trả lời, dấu đóng `</think>` của template và token kết thúc; không chứa câu hỏi. Tỷ lệ giám sát 0,4149 thấp hơn ngưỡng sai mask 0,95. NB3/NB4 dùng labels pre-tokenize theo mask của lab thay vì chỉ tin cờ `assistant_only_loss`. Bằng chứng NB1 trực tiếp kiểm tra một mẫu, không phải kiểm định thủ công mọi mẫu trong corpus.

## 3. Lịch sử chạy và ba baseline

### 3.1. Trình tự thực tế

1. Lượt đầu chạy NB1 → NB5 với `EVAL_LIMIT=8`. Baseline 8 mẫu được đo trước train theo thứ tự pipeline. Kết quả cũ giữ trong `results/archive/`.
2. Sau train, giữ nguyên bốn adapter, bỏ giới hạn eval và chạy `STAGES="nb2 nb5"` trên 50 target / 15 regression. `results/log_lan2` xác nhận hai stage hoàn tất. File baseline mới có `smoke_mode=false`, `eval_limit=null`.
3. Các bảng chính dùng kết quả đầy đủ trong `results/`; `runs.csv` vẫn là số đo train của lượt đầu. Lần chạy bổ sung không train lại.

**Giới hạn về trình tự:** baseline đầy đủ được đo sau train; mốc đã đo trước train chỉ là bản 8 mẫu. Tôi không mô tả baseline đầy đủ như đã đóng băng trước train. Prompt tối ưu giữ SHA `719e74d3b6232053` ở cả hai lượt, và gatekeeper Colab xác nhận corpus không đổi. Đánh giá bổ sung khắc phục thiếu mẫu nhưng không thay đổi lịch sử. Một lượt tái lập chặt chẽ hơn sẽ đo baseline đầy đủ trước huấn luyện.

### 3.2. Kết quả đầy đủ

| Run | Target | Regression | Format | Latency (ms/mẫu) |
|---|---:|---:|---:|---:|
| (a) Base + naive prompt | 0,0000 | 0,7911 | 0,0000 | 3151,1 |
| (b) Base + optimized prompt | 0,7650 | 0,7911 | 1,0000 | 1008,2 |
| (c) LoRA correct + naive prompt | 0,9700 | 0,7222 | 1,0000 | 1389,4 |

Nguồn: `results/verdict.json`; baseline chưa làm tròn trong `results/baselines_frozen.json`. Target là trung bình tỷ lệ đúng bốn trường, không phải tỷ lệ ticket đúng hoàn toàn. Regression là keyword recall trên 15 câu phổ thông. Format theo hàm chấm JSON/khóa của lab. Greedy decode, batch sinh mặc định 4; latency là thời gian sinh theo batch chia số mẫu, không phải latency end-to-end của dịch vụ.

Baseline (b) tốt hơn (a): target từ 0 lên 0,765 và format từ 0 lên 1. Tôi không sửa prompt tối ưu của repo để làm yếu đối thủ. Điểm 0 của (a) là kết quả theo bộ chấm JSON, chưa chứng minh base không hiểu nội dung ticket.

Fine-tune dùng prompt ngắn nhưng chậm hơn (b) khoảng 381,2 ms/mẫu. Prompt ngắn hơn chưa bảo đảm latency thấp hơn. Chưa có đo lặp hoặc khoảng tin cậy để khẳng định ưu thế tốc độ ổn định.

## 4. Bốn cấu hình và tính công bằng — NB3/NB4/NB5

| Run | Vị trí | r / alpha | Tham số train | LR | Training loss | Target | Train (s) | Peak VRAM (GiB) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| correct | text-linear | 16 / 32 | 32.464.896 | 0,0001 | 0,6265 | 0,9700 | 393,6 | 8,78 |
| attn_only | q,v | 283 / 566 | 32.456.704 | 0,0001 | 0,5375 | 0,9650 | 263,2 | 8,79 |
| wrong_lr | text-linear | 16 / 32 | 32.464.896 | 0,00001 | 1,5702 | 0,0000 | 392,8 | 8,78 |
| qlora | text-linear | 16 / 32 | 32.464.896 | 0,0001 | 0,7058 | 0,9400 | 464,4 | 3,86 |

Nguồn: `results/runs.csv`, `results/autopsy.json`. Cột `final_loss` được code lấy từ `result.training_loss`, không phải điểm eval hoặc nhất thiết là loss riêng của step cuối. Cả bốn run đều ghi 30 step. CSV chỉ ghi `mask_mode` trực tiếp cho correct; code NB4 dùng cùng mặc định assistant-only nhưng không ghi riêng cột này cho ba đối chứng.

### 4.1. Vị trí adapter và rank

Attention-only tăng rank lên 283 để khớp ngân sách correct. Hai run chênh 8.192 tham số, khoảng 0,0252%, dưới ngưỡng 5%. Biến đối chứng là placement; rank/alpha thay đổi đi kèm để giữ ngân sách và alpha/r=2. Đây không phải phép đo riêng tác động của rank khi giữ nguyên vị trí.

Xếp theo target: correct > attn_only > qlora > wrong_lr. Xếp theo training loss lại đưa attn_only lên trước correct: 0,5375 < 0,6265. Loss thấp hơn chưa bảo đảm khả năng tổng quát hóa tốt hơn. Chênh target hai run đầu chỉ 0,005, tương đương một trường trên tổng 200 trường; một lượt chưa đủ chứng minh ưu thế placement trên mọi dataset. Attention-only còn có latency 880,0 ms/mẫu, nhưng chưa được chấm regression riêng nên chưa thể coi là phương án triển khai tốt hơn.

### 4.2. Learning rate

Wrong_lr giữ placement, rank, alpha, ngân sách tham số và số step như correct, chỉ giảm LR mười lần. Training loss là 1,5702 so với 0,6265; target và format đều 0, latency 5182,0 ms/mẫu. Kết quả phù hợp với giả thuyết LR thấp chưa học đủ hành vi JSON trong 30 step. Tuy nhiên, artifact không chứa toàn bộ đường loss hoặc output đầy đủ của run này, nên tôi không khẳng định đường loss phẳng hay một cơ chế sinh lỗi cụ thể. Nếu bỏ qua LR, tôi có thể nhầm rằng LoRA không phù hợp, trong khi cùng ngân sách với LR cao hơn đã đạt target 0,97.

### 4.3. QLoRA

QLoRA giảm VRAM từ 8,78 xuống 3,86 GiB: tiết kiệm 4,92 GiB, khoảng 56,0%. Đổi lại, target giảm 0,03; train tăng từ 393,6 lên 464,4 giây; latency tăng từ 1389,4 lên 1760,2 ms/mẫu. Format vẫn đạt 1,0. NB5 dùng base 4-bit khi chấm adapter QLoRA, tránh sai khớp base/adapter.

Nếu đủ bộ nhớ, số đo ủng hộ LoRA 16-bit để ưu tiên chất lượng và tốc độ ở cấu hình này. Nếu thiếu VRAM, mức tiết kiệm là lợi ích thực tế; chưa có cơ sở loại bỏ QLoRA trong mọi tình huống. Ba đối chứng chỉ được chấm target/format/latency ở NB5, chưa có regression riêng, nên chưa có phán quyết bốn nhóm cho chúng.

## 5. Phán quyết và diễn giải

**FAILED** · `target_delta=+0,205000` · `regression_delta=-0,068889` · `valid_trace_rate=0,0000`.

Fine-tune cải thiện target 20,5 điểm phần trăm và đạt format 1,0. Tuy nhiên, regression giảm từ 0,7911 xuống 0,7222, khoảng 6,89 điểm phần trăm, vượt ngưỡng 2 điểm. Cổng đánh giá yêu cầu đồng thời thắng baseline trên target và giữ năng lực phổ thông trong ngưỡng, vì vậy target 97% vẫn dẫn tới FAILED. Tôi giữ nguyên ngưỡng và không diễn giải thành thắng chung chỉ dựa vào tác vụ chuyên biệt. Output bổ sung trong `results/qualitative_pairs.json` cho thấy model áp dụng JSON phân loại ngay cả khi được hỏi một năm có bao nhiêu tháng, thay vì trả lời 12. Điều này phù hợp với giả thuyết corpus triage hẹp làm model áp dụng quá mức hành vi chuyên biệt, nhưng chưa chứng minh cơ chế quên vì chưa có đối chứng replay. Tập regression chỉ 15 câu và dùng keyword recall, nên đây là cảnh báo trong phạm vi lab, chưa phải phép đo toàn diện năng lực phổ thông. Fine-tune còn chậm hơn baseline prompt tối ưu ở lượt đo này. Tôi chưa đề xuất thay base model dùng chung bằng adapter hiện tại.

**Giới hạn của scorer:** ở regression i=11, baseline nói về dâu tây nhưng vẫn được 0,2 điểm vì từ “đưa” sau bỏ dấu trùng “dứa”. Ngược lại, ở i=14, fine-tune giải thích quang hợp có “thực vật” nhưng thiếu đúng từ khóa “cây”, nên chỉ được 0,5. Vì vậy, bốn ca thua là thua theo scorer; không phải cả bốn câu trả lời đều sai hoàn toàn về nội dung. Tôi giữ nguyên cách chấm và phán quyết đã đóng băng, đồng thời đọc output để tránh diễn giải quá mức.

Valid trace bằng 0 chưa chứng minh reasoning-trace collapse: corpus là JSON thuần và hàm sinh mặc định tắt thinking. Chưa thực hiện đối chứng mask trên dữ liệu reasoning.

### 5.1. Vì sao lượt smoke PASS còn lượt đầy đủ FAIL?

| Chỉ số | 8 mẫu — results/archive/ | Đầy đủ — results/ |
|---|---:|---:|
| Baseline (b) target | 0,6875 | 0,7650 |
| Fine-tune target | 0,9375 | 0,9700 |
| Baseline (b) regression | 0,7500 | 0,7911 |
| Fine-tune regression | 0,8750 | 0,7222 |
| Phán quyết | PASSED | FAILED |

Adapter không train lại; thay đổi chính là mở rộng tập chấm. Tám câu regression đầu cho tín hiệu tích cực, toàn bộ 15 câu lại cho thấy suy giảm. Tôi không dùng PASS trong archive làm kết luận cuối hay trộn hai lượt vào bảng chính.

## 6. Phân tích định tính

<!-- qualitative:start -->

Nguồn: `results/qualitative_pairs.json`. Output được sinh bổ sung với cùng model, prompt và cấu hình sinh của NB2/NB5; điểm cả hai model khớp kết quả đã lưu trong sai số làm tròn. Không thay baseline, verdict hoặc adapter. Bảng dưới trích các phần liên quan; output nguyên văn nằm trong JSON. Chỉ số i đếm từ 0 trong từng tập.

| Nhóm / i | Câu hỏi hoặc ticket rút gọn | Nhãn / từ khóa tham chiếu | Baseline (b), trích đoạn | Fine-tune (c), trích đoạn | Điểm b / c | Kết quả |
|---|---|---|---|---|---|---|
| target / 0 | Chuột không dây; “Cho tôi trả lại. Gấp. Shop hỗ trợ tốt.” | doi_tra / cao / chuột không dây / tich_cuc | intent=hoan_tien; ba trường còn lại đúng | intent=doi_tra; đúng cả bốn trường | 0,75 / 1,00 | FT thắng |
| target / 1 | Ốp lưng điện thoại; “Hoàn tiền. Sớm nhé. Bực mình.” | hoan_tien / trung_binh / ốp lưng điện thoại / tieu_cuc | urgency=cao; ba trường còn lại đúng | urgency=trung_binh; đúng cả bốn trường | 0,75 / 1,00 | FT thắng |
| target / 2 | Đèn bàn LED; “Hoàn tiền. Quá hạn rồi. Cảm ơn shop nhiều.” | hoan_tien / cao / đèn bàn LED / tich_cuc | Đúng cả bốn trường | Đúng cả bốn trường | 1,00 / 1,00 | Hòa |
| regression / 9 | Một năm có bao nhiêu tháng? | 12 | “Một năm bình thường có **12 tháng**.” | `{"intent":"hoi_thong_tin","urgency":"thap","product":null,"sentiment":"trung_tinh"}` | 1,00 / 0,00 | FT thua |
| regression / 12 | Ý nghĩa “Có công mài sắt có ngày nên kim” | kiên trì / cố gắng / bền | “…kiên trì, nỗ lực và sự bền bỉ…” | JSON có summary “…phải có tinh thần kiên trì, không nản lòng trước khó khăn thì sẽ thành công.” | 0,6667 / 0,3333 | FT thua theo scorer |

Target có **33 ca thắng, 17 ca hòa, không có ca thua**; regression có **1 ca thắng, 4 ca thua, 10 ca hòa**. Tổng 65 mẫu là 34 thắng / 4 thua / 27 hòa. Hai ca thua được chọn ở nhóm regression, không gọi các lỗi target là thua baseline khi dữ liệu không chứng minh điều đó.

Ở target / 0, fine-tune phân biệt trả hàng với hoàn tiền tốt hơn baseline; ở target / 1, nó không đồng nhất cảm xúc tiêu cực với mức khẩn cao. Regression / 9 cho thấy tác dụng phụ rõ: model trả JSON phân loại thay cho đáp án kiến thức. Regression / 12 vẫn giải thích đúng tinh thần câu tục ngữ nhưng có ít từ khóa hơn, nên điểm giảm chưa đủ để kết luận nội dung sai hoàn toàn.

Sáu lỗi target tại i=3,5,12,39,41,46 đều dự đoán urgency=trung_binh thay vì thap với cụm “Khi nào tiện”. Có 44/50 ticket đúng hoàn toàn và 194/200 trường đúng: **88% ticket accuracy**, tương ứng **97% field accuracy**. Đây là lỗi so với nhãn; trên cùng mẫu, fine-tune vẫn hòa hoặc thắng baseline. Nếu sửa dữ liệu để xử lý kiểu lỗi này, cần một tập đánh giá độc lập, tránh đưa chính các ca eval vào train rồi chấm lại.

<!-- qualitative:end -->

## 7. Kết luận và điều rút ra

### 7.1. Kết luận

Tôi chưa chọn triển khai adapter correct như bản thay thế cho base model dùng chung. Nó xử lý triage tốt hơn prompt tối ưu trên dữ liệu tổng hợp, nhưng không vượt qua điều kiện bảo toàn năng lực phổ thông và không có lợi thế latency trong lần đo này. Target tăng cho thấy model học hành vi chuyên biệt; regression giảm cho thấy không thể dùng riêng target để quyết định triển khai. Ngay cả với luồng triage đóng, tôi vẫn cần kiểm tra ticket thực tế, câu ngoài phạm vi và phương án chuyển tiếp trước khi thử nghiệm có giới hạn.

Đòn bẩy quan sát rõ nhất là thang learning rate: cùng ngân sách tham số và số step, LR thấp dẫn đến target/format bằng 0. Với placement, text-linear chỉ hơn attention-only một trường trên 200 trường; đó là ưu thế nhỏ trong lượt đo này, chưa phải quy luật tổng quát. Mask đúng là điều kiện để kết quả có ý nghĩa, nhưng tôi không train đối chứng mask sai nên không định lượng riêng tác động của nó. QLoRA tiết kiệm bộ nhớ và đánh đổi chất lượng, tốc độ. Lượt smoke PASS rồi lượt đầy đủ FAIL nhắc tôi kiểm tra phạm vi đánh giá trước khi kết luận. Output từng mẫu đã được lưu và đối chiếu; hướng nghiên cứu tiếp theo là thử replay dữ liệu phổ thông với ngân sách kiểm soát và đánh giá trên bộ đo xác định trước.

### 7.2. Ba điều học được từ kết quả

1. **Smoke test chưa đủ để quyết định:** cùng adapter, regression 8 mẫu là 0,875 còn đầy đủ là 0,7222, đảo phán quyết. Cần kiểm tra số mẫu và smoke_mode trước khi so điểm.
2. **Loss có thể xếp hạng sai:** attn_only có training loss thấp hơn correct nhưng target thấp hơn. Cần đọc autopsy.json cùng runs.csv.
3. **Điểm trung bình che lỗi cụ thể:** field accuracy 97% đi cùng ticket accuracy 88%; sáu lỗi còn lại cùng liên quan urgency của “Khi nào tiện”. Phân tích từng trường tạo ra hướng kiểm tra cụ thể hơn.

**Nếu có thêm thời gian:** thử trộn 1–5% dữ liệu phổ thông không trùng tập eval, kiểm soát ngân sách và đo lại bốn nhóm; kiểm tra thêm đánh giá nội dung để bổ sung cho keyword recall. Đây là thí nghiệm tiếp theo, không ghi đè lịch sử hiện tại hoặc khẳng định trước replay sẽ khắc phục regression.

**Sử dụng AI assistant:** đọc repo, đối chiếu artifact, tính chênh lệch và hỗ trợ soạn report. Điểm cần kiểm soát là phân biệt “sai so với nhãn” với “thua baseline”. Tôi đã chạy script bổ sung trên Colab để có dự đoán cả hai model trên cùng mẫu, thay vì suy đoán quan hệ thắng/thua từ điểm tổng hợp.

## 8. Kiểm tra và phạm vi nộp

Log `log_o_4` trước khi điền report ghi **25 passed · 1 warning · 1 failure**, với **119 unit test passed**. Failure duy nhất là report còn sáu placeholder. Warning là verdict FAILED, một kết quả được chấp nhận để phân tích, không phải lý do nới ngưỡng.

Cần chạy gatekeeper lại trên Colab sau khi cập nhật report để có log mới. Gatekeeper không xác nhận đầy đủ ví dụ định tính hay baseline đầy đủ trước train; các giới hạn đã khai báo vẫn cần người chấm xem xét.

Tôi dự kiến nộp bằng link GitHub với nội dung code-only theo Option C: `submission/REPORT.md`, `submission/REFLECTION.md`, đầy đủ `results/` và `requirements.txt`; mã nguồn/notebook trong repo hỗ trợ tái lập. Giữ archive và log để truy nguyên lượt smoke. Các JSON/CSV kết quả đang bị `.gitignore` bỏ qua, cần thêm chúng vào Git trước khi push. Adapter được sao lưu riêng, chưa có public HuggingFace Hub nên chưa chọn hình thức GitHub + Hub của Option B. Checksum trên Windows có thể khác do CRLF/LF; log Colab đã xác nhận eval sets unmodified.

**Điểm thưởng:** chưa có bằng chứng hoàn thành NB6 merge/hot-swap, dataset miền riêng, đối chứng reasoning mask, quét rank có kiểm soát hoặc public HuggingFace Hub; không yêu cầu điểm thưởng cho các mục này.
