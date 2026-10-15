# 67Sight Live

Trợ lý mô tả môi trường bằng AI — nguyên mẫu nghiên cứu dùng Gradio và YOLOv8n.

## Chạy local

Yêu cầu Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Mở địa chỉ local mà Gradio in ra. Cấp quyền camera và microphone khi được hỏi.

## Deploy lên Render

1. Đẩy các file trong repository này lên GitHub.
2. Trong Render, chọn **New + → Web Service** và kết nối repository.
3. Chọn gói Free.
4. Dùng build command `pip install --no-cache-dir -r requirements.txt`.
5. Dùng start command `python app.py`.
6. Đợi build hoàn tất, mở URL HTTPS được cấp và thử camera/micro.

`render.yaml` có thể dùng làm cấu hình tham khảo cho Blueprint.

## Lưu ý

- Model `yolov8n.pt` được tải bởi Ultralytics khi ứng dụng khởi động lần đầu.
- Ứng dụng gửi khung hình webcam lên máy chủ để suy luận; không coi xử lý là hoàn toàn trên thiết bị.
- Render Free có giới hạn tài nguyên và có thể ngủ khi không hoạt động.
- Nhận dạng giọng nói trên trình duyệt phụ thuộc thiết bị và trình duyệt.
- Đây là nguyên mẫu nghiên cứu, không phải thiết bị dẫn đường được chứng nhận. Không dùng làm chỉ dẫn di chuyển duy nhất.
