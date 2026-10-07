# Reflection — Lab 21

**Phạm Quang Đạt · 2A202602704**

**1. Điều làm tôi chú ý nhất**

Cùng một adapter, đánh giá 8 mẫu cho PASS nhưng đánh giá đầy đủ lại FAIL. Target đạt 97% vẫn không đủ vì regression giảm khoảng 6,89 điểm phần trăm.

**2. Thời gian và việc phát sinh**

Trong lượt đánh giá bổ sung, NB5 mất 661 giây, lâu hơn NB2 với 309 giây. Tôi còn phải thu thêm output từng mẫu vì các file ban đầu chỉ lưu điểm baseline tổng hợp. Không có nhật ký thời gian thao tác để kết luận phần nào mất nhiều công nhất.

**3. Cách tôi nhìn fine-tuning sau bài này**

Tôi sẽ không xem loss giảm hoặc target tăng là đủ để chọn model. Attention-only có loss thấp hơn correct nhưng target thấp hơn; model chuyên biệt còn trả JSON phân loại khi được hỏi một năm có bao nhiêu tháng.

**4. Tôi dùng AI assistant thế nào?**

AI hỗ trợ đọc repo, kiểm tra kết quả, viết script đối chiếu và soạn báo cáo. Điểm cần sửa trong hướng dẫn ban đầu là phân biệt ca fine-tune sai nhãn với ca thua baseline; phải có dự đoán trên cùng mẫu mới xác nhận được. Điểm keyword recall cũng cần đọc cùng output thực tế.

**5. Bước đầu với khách hàng thật**

Tôi sẽ làm rõ tác vụ và yêu cầu chất lượng, lập tập đánh giá đại diện, rồi đo baseline với prompt tốt trước khi train. Sau đó mới quyết định có cần fine-tune và kiểm tra cả năng lực ngoài tác vụ.
