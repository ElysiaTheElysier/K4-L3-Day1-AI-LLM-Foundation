# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi tăng temperature từ 0.0 lên 1.5, mức độ đa dạng và tính ngẫu nhiên của phản hồi tăng dần rõ rệt. Ở mức 0.0, câu trả lời mang tính quy chuẩn, lặp lại các sự thật phổ biến nhất với giọng văn trung tính. Khi tăng lên 0.5 và 1.0, từ ngữ trở nên sinh động và câu văn uyển chuyển hơn. Ở mức 1.5, câu trả lời rất bay bổng và sáng tạo nhưng bắt đầu xuất hiện tình trạng dài dòng, câu cú kém chặt chẽ hơn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature ở mức thấp từ 0.0 đến 0.2 (tối đa 0.3). Lý do là chatbot hỗ trợ khách hàng đòi hỏi tính chính xác tuyệt đối, bám sát các điều khoản chính sách của doanh nghiệp, đảm bảo tính nhất quán (nhiều khách hàng cùng hỏi một câu sẽ nhận được giải đáp đồng nhất) và hạn chế tối đa nguy cơ ảo giác (hallucination).

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Với workload 10,5 triệu token output mỗi ngày, GPT-4o tiêu tốn $105/ngày trong khi GPT-4o-mini chỉ tốn $6.3/ngày, tức GPT-4o đắt hơn khoảng 16.7 lần. GPT-4o xứng đáng sử dụng cho các bài toán phân tích logic phức tạp, tư vấn tài chính, giải thích hợp đồng pháp lý hoặc sinh code; còn GPT-4o-mini là lựa chọn tối ưu cho các tác vụ tóm tắt hội thoại ngắn, phân loại ticket hỗ trợ hoặc trả lời các câu hỏi thường gặp (FAQ).

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Hai phản hồi hoàn toàn khác biệt: Persona giáo viên tiểu học dùng hình ảnh ẩn dụ gần gũi (cuốn sổ ghi chép chung của cả lớp mà ai cũng nhìn thấy và không thể tẩy xóa), câu văn ngắn và từ ngữ đơn giản. Ngược lại, persona chuyên gia tài chính sử dụng thuật ngữ học thuật chuyên sâu (sổ cái phân tán DLT, cơ chế đồng thuận PoW/PoS, hàm băm mật mã cryptographic hash và tính bất biến). Điều này chứng minh system prompt có khả năng định hình mạnh mẽ vai trò, đối tượng độc giả và độ sâu chuyên môn của câu trả lời.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Đoạn văn tiếng Việt ~100 từ theo công thức ước lượng thô (số từ / 0.75) cho ra khoảng 133 token, nhưng thực tế đếm bằng tiktoken thường đạt khoảng 150 - 180 token, chênh lệch từ 15% đến 35%. Tiếng Việt tốn nhiều token hơn tiếng Anh cùng độ dài vì các thuật toán phân tách token (BPE) được huấn luyện chủ yếu trên văn bản tiếng Anh; trong tiếng Việt, các từ có dấu thanh và dấu phụ (ă, â, ê, ô, ơ, ư...) thường không nằm nguyên vẹn trong từ điển vocab nên bị phân tách thành nhiều subword hoặc byte riêng rẽ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các giao diện tương tác trực tiếp với người dùng (chatbot CLI, web chat) nhằm tối ưu thời gian phản hồi đầu tiên (Time-To-First-Token - TTFT), giúp người dùng đọc được nội dung ngay khi model bắt đầu sinh ra thay vì phải chờ đợi lâu. Ngược lại, non-streaming phù hợp hơn cho các pipeline ngầm (backend API, cron job xử lý theo lô, trích xuất dữ liệu JSON có cấu trúc) hoặc khi hệ thống cần thực hiện kiểm duyệt nội dung (content moderation) trên toàn bộ văn bản hoàn chỉnh trước khi gửi trả.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff dãn dần khoảng cách chờ giữa các lần thử (0.1s -> 0.2s -> 0.4s...), tạo điều kiện cho server có thời gian giải phóng tài nguyên nghẽn và hồi phục sau sự cố. Nếu hàng nghìn client cùng sử dụng delay cố định (ví dụ luôn chờ 1 giây), hiện tượng 'Thundering Herd' (đàn bò sấm sét) sẽ xảy ra: toàn bộ các client sẽ đồng loạt gửi lại request tại cùng một thời điểm sau 1 giây, tạo ra một đợt tấn công quá tải mới khiến server tiếp tục sập và không thể nào phục hồi được.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> System prompt tôi chọn: 'Bạn là trợ giảng lập trình thân thiện của khóa AI. Hãy giải thích ngắn gọn, dễ hiểu và trả lời hoàn toàn bằng tiếng Việt có kèm ví dụ code minh họa ngắn.' Trong prompt này, việc sử dụng cụm từ 'ngắn gọn, dễ hiểu' giúp kiểm soát độ dài phản hồi của model, tiết kiệm token input/output và giảm độ trễ; chỉ định 'hoàn toàn bằng tiếng Việt' ngăn model tự ý chuyển sang tiếng Anh khi gặp thuật ngữ lập trình chuyên ngành, giữ tính đồng nhất cho trải nghiệm người học.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất của trợ lý hiện tại là bộ nhớ ngắn hạn bị giới hạn ở 3 lượt gần nhất và mất toàn bộ lịch sử trò chuyện khi tắt chương trình terminal. Đề xuất cải thiện: Tích hợp cơ chế Tóm tắt hội thoại (Conversation Summarization) kết hợp lưu trữ lâu dài vào SQLite/JSON. Cách triển khai: Khi số lượt vượt quá 3, thay vì cắt bỏ thẳng tay, gọi một model nhẹ (gpt-4o-mini) để tóm tắt các lượt cũ thành một đoạn văn ngắn lưu trong system prompt, đồng thời ghi log toàn bộ nội dung ra file local để tự động khôi phục lại ngữ cảnh ở các phiên làm việc tiếp theo.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
