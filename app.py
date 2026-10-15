import os
import threading
import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO

# 67Sight Live — research prototype, not a certified mobility aid.
MODEL_NAME = os.getenv("MODEL_NAME", "yolov8n.pt")
model = YOLO(MODEL_NAME)
model_lock = threading.Lock()

VI_LABELS = {
    "person":"người", "bicycle":"xe đạp", "car":"ô tô", "motorcycle":"xe máy",
    "airplane":"máy bay", "bus":"xe buýt", "train":"tàu hỏa", "truck":"xe tải",
    "boat":"thuyền", "traffic light":"đèn giao thông", "fire hydrant":"trụ cứu hỏa",
    "stop sign":"biển dừng", "parking meter":"máy tính tiền đỗ xe", "bench":"ghế băng",
    "bird":"chim", "cat":"mèo", "dog":"chó", "horse":"ngựa", "sheep":"cừu",
    "cow":"bò", "elephant":"voi", "bear":"gấu", "zebra":"ngựa vằn",
    "giraffe":"hươu cao cổ", "backpack":"ba lô", "umbrella":"ô hoặc dù",
    "handbag":"túi xách", "tie":"cà vạt", "suitcase":"va li", "frisbee":"đĩa ném",
    "skis":"ván trượt tuyết", "snowboard":"ván trượt tuyết", "sports ball":"quả bóng",
    "kite":"diều", "baseball bat":"gậy bóng chày", "baseball glove":"găng bóng chày",
    "skateboard":"ván trượt", "surfboard":"ván lướt sóng", "tennis racket":"vợt tennis",
    "bottle":"chai", "wine glass":"ly", "cup":"cốc", "fork":"nĩa", "knife":"dao",
    "spoon":"thìa", "bowl":"bát", "banana":"chuối", "apple":"táo",
    "sandwich":"bánh mì kẹp", "orange":"quả cam", "broccoli":"bông cải xanh",
    "carrot":"cà rốt", "hot dog":"xúc xích", "pizza":"pizza", "donut":"bánh vòng",
    "cake":"bánh", "chair":"ghế", "couch":"ghế sofa", "potted plant":"chậu cây",
    "bed":"giường", "dining table":"bàn", "toilet":"bồn cầu", "tv":"ti vi",
    "laptop":"máy tính xách tay", "mouse":"chuột máy tính", "remote":"điều khiển",
    "keyboard":"bàn phím", "cell phone":"điện thoại", "microwave":"lò vi sóng",
    "oven":"lò nướng", "toaster":"máy nướng bánh", "sink":"bồn rửa",
    "refrigerator":"tủ lạnh", "book":"sách", "clock":"đồng hồ", "vase":"lọ hoa",
    "scissors":"kéo", "teddy bear":"gấu bông", "hair drier":"máy sấy tóc",
    "toothbrush":"bàn chải đánh răng"
}
ATTENTION_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
    "boat", "bench", "chair", "couch", "suitcase", "backpack", "umbrella",
    "fire hydrant", "stop sign"
}

def analyze_frame(frame, confidence, detail_level):
    if frame is None:
        return None, "Đang chờ camera. Hãy bật webcam để bắt đầu."
    if isinstance(frame, Image.Image):
        rgb = np.array(frame.convert("RGB"))
    else:
        rgb = np.asarray(frame)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]

    h, w = rgb.shape[:2]
    # A single model lock prevents simultaneous inference from overloading a small CPU instance.
    with model_lock:
        result = model.predict(
            source=rgb,
            conf=float(confidence),
            imgsz=416,
            device="cpu",
            verbose=False
        )[0]

    annotated = result.plot()[:, :, ::-1].copy()
    items = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            en = model.names.get(cls_id, str(cls_id))
            vi = VI_LABELS.get(en, en)
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            cx = (x1 + x2) / 2
            pos = "bên trái" if cx < w / 3 else ("ở giữa" if cx < 2 * w / 3 else "bên phải")
            area = max(0, (x2 - x1) * (y2 - y1)) / max(1, w * h)
            items.append({"en": en, "vi": vi, "pos": pos, "score": score, "area": area})

    if not items:
        speech_text = "Mình chưa nhận diện rõ vật thể nào trong các nhóm đã học."
    else:
        groups = {}
        for it in items:
            key = (it["vi"], it["pos"])
            groups[key] = max(groups.get(key, 0), it["area"])
        ordered = sorted(groups.items(), key=lambda kv: kv[1], reverse=True)
        max_items = 8 if detail_level == "Chi tiết" else 5
        phrases = [f"{name} {pos}" for (name, pos), _ in ordered[:max_items]]
        speech_text = "Mình thấy " + ", ".join(phrases) + "."
        attention = [x for x in items if x["en"] in ATTENTION_CLASSES]
        central_large = [x for x in attention if x["pos"] == "ở giữa" and x["area"] > 0.12]
        central_any = [x for x in attention if x["pos"] == "ở giữa"]
        if central_large:
            speech_text += " Có vật thể chiếm diện tích lớn ở giữa hình ảnh. Hãy dừng lại và kiểm tra bằng gậy trắng hoặc phương tiện định hướng quen thuộc."
        elif central_any:
            speech_text += " Có vật thể ở giữa hình ảnh. Hãy thận trọng và kiểm tra trước khi bước tiếp."

    full_text = speech_text + (
        "\n\nLưu ý an toàn: vị trí trái, giữa, phải chỉ là vị trí trong ảnh. "
        "AI không đo được khoảng cách thật, có thể bỏ sót vật cản và không xác nhận "
        "được lối đi an toàn. Không dùng kết quả này làm chỉ dẫn di chuyển duy nhất."
    )
    return Image.fromarray(annotated), full_text

CSS = """
.gradio-container {max-width: 1250px !important; margin: auto !important;}
#hero {padding: 24px; border-radius: 20px; background: linear-gradient(120deg,#12283f,#176b70); color: white;}
#hero h1 {margin: 0 0 8px 0; font-size: 2.2rem;}
#hero p {margin: 0; font-size: 1.05rem; line-height: 1.6;}
#voice-controls {padding: 16px; border-radius: 16px; border: 2px solid #168b86; margin: 8px 0 16px;}
#voice-controls button {font-size: 1.05rem; padding: 12px 18px; margin: 5px; border-radius: 12px; cursor: pointer;}
.warning {padding: 14px 16px; border-radius: 12px; border: 1px solid #d6a43b; line-height: 1.6;}
"""

VOICE_JS = r"""() => {
  const root = document.querySelector('#voice-controls');
  if (!root || root.dataset.ready === '1') return;
  root.dataset.ready = '1';
  root.innerHTML = `
    <h3>Điều khiển bằng giọng nói</h3>
    <p id="voice-status" role="status">Chưa bật trợ lý giọng nói. Hãy nhấn nút bắt đầu một lần rồi sử dụng câu lệnh.</p>
    <button type="button" id="voice-start">Bắt đầu trợ lý giọng nói</button>
    <button type="button" id="voice-stop">Tắt nghe lệnh</button>
    <p><b>Câu lệnh:</b> “Mô tả xung quanh”, “Nhắc lại”, “Tạm dừng đọc”, “Tiếp tục đọc”, “Nói chậm”, “Nói nhanh”, “Trợ giúp”.</p>`;
  const status = root.querySelector('#voice-status');
  let recognition = null, listening = false, paused = false, lastText = '', rate = 1.0;
  let lastQueuedText = '';

  const cleanText = text =>
    (text || '').replace(/\n\nLưu ý an toàn:[\s\S]*/i, '').trim();

  const speak = (text, force=false) => {
    const clean = cleanText(text);
    if (!clean || paused) return;
    if (!force && clean === lastText) return;
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(clean);
    utter.lang = 'vi-VN';
    utter.rate = rate;
    utter.pitch = 1.05;
    const voices = window.speechSynthesis.getVoices();
    const viVoices = voices.filter(v => /^vi(-|_)/i.test(v.lang) || /vietnam/i.test(v.name));
    const preferred = viVoices.find(v => /female|woman|mai|linh/i.test(v.name));
    if (preferred || viVoices[0]) utter.voice = preferred || viVoices[0];
    window.speechSynthesis.speak(utter);
    lastText = clean;
  };

  const latestText = () => {
    const box = document.querySelector('#scene-text textarea');
    return box ? box.value : '';
  };

  const command = raw => {
    const t = raw.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').trim();
    if (t.includes('dung doc') || t.includes('tam dung') || t === 'im lang') {
      paused = true;
      window.speechSynthesis.cancel();
      status.textContent = 'Đã tạm dừng đọc. Nói “tiếp tục đọc” để bật lại.';
    } else if (t.includes('tiep tuc')) {
      paused = false;
      status.textContent = 'Đã tiếp tục đọc mô tả.';
      speak(latestText(), true);
    } else if (t.includes('nhac lai') || t.includes('noi lai')) {
      paused = false;
      speak(latestText(), true);
    } else if (t.includes('mo ta') || t.includes('xung quanh') || t.includes('co gi truoc mat')) {
      paused = false;
      speak(latestText(), true);
    } else if (t.includes('noi cham')) {
      rate = Math.max(0.75, rate - 0.15);
      speak('Mình sẽ nói chậm hơn.', true);
    } else if (t.includes('noi nhanh')) {
      rate = Math.min(1.25, rate + 0.15);
      speak('Mình sẽ nói nhanh hơn.', true);
    } else if (t.includes('tro giup') || t.includes('lenh')) {
      speak('Bạn có thể nói: mô tả xung quanh, nhắc lại, tạm dừng đọc, tiếp tục đọc, nói chậm hoặc nói nhanh.', true);
    }
  };

  const start = () => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      status.textContent = 'Trình duyệt này chưa hỗ trợ nhận dạng giọng nói. Hãy thử Chrome hoặc Edge.';
      speak(status.textContent, true);
      return;
    }
    if (recognition) {
      listening = true;
      try { recognition.start(); } catch(e) {}
      return;
    }
    recognition = new SR();
    recognition.lang = 'vi-VN';
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.onresult = event => {
      const r = event.results[event.results.length - 1];
      if (r.isFinal && r[0]) command(r[0].transcript);
    };
    recognition.onerror = event => {
      status.textContent = 'Micro: ' + event.error + '. Nếu được hỏi, hãy cho phép dùng micro.';
    };
    recognition.onend = () => {
      if (listening) {
        try { recognition.start(); } catch(e) {}
      }
    };
    listening = true;
    try {
      recognition.start();
      status.textContent = 'Đang nghe lệnh tiếng Việt. Camera sẽ tiếp tục phân tích tự động.';
      speak('67Sight đã sẵn sàng. Mình sẽ mô tả những gì camera nhận diện được.', true);
    } catch(e) {
      status.textContent = 'Chưa bật được micro. Hãy kiểm tra quyền micro của trình duyệt.';
    }
  };

  root.querySelector('#voice-start').addEventListener('click', start);
  root.querySelector('#voice-stop').addEventListener('click', () => {
    listening = false;
    if (recognition) recognition.stop();
    window.speechSynthesis.cancel();
    status.textContent = 'Đã tắt nghe lệnh.';
  });

  // Speak a changed description only after the current utterance finishes.
  setInterval(() => {
    const current = cleanText(latestText());
    if (!current || current === lastQueuedText) return;
    lastQueuedText = current;
    if (listening && !paused && !window.speechSynthesis.speaking && !window.speechSynthesis.pending) {
      speak(current, true);
    }
  }, 1200);
}"""

with gr.Blocks(
    title="67Sight Live — AI hỗ trợ người khiếm thị",
    theme=gr.themes.Soft(primary_hue="teal", secondary_hue="blue"),
    css=CSS
) as demo:
    gr.HTML(
        "<div id='hero'><h1>67Sight Live</h1>"
        "<p>Trợ lý mô tả môi trường bằng AI · Tiếng Việt · Điều khiển bằng giọng nói</p></div>"
    )
    gr.Markdown(
        "**Cách sử dụng:** bật webcam trực tiếp, sau đó nhấn “Bắt đầu trợ lý giọng nói” "
        "để cấp quyền micro. Ứng dụng gửi khung hình lên máy chủ để nhận diện và tự mô tả "
        "các vật thể được phát hiện."
    )
    gr.HTML("<div id='voice-controls' aria-label='Điều khiển trợ lý giọng nói'></div>")
    with gr.Row():
        with gr.Column(scale=6):
            webcam = gr.Image(
                sources=["webcam"], type="numpy", streaming=True,
                label="Camera trực tiếp", height=390
            )
            with gr.Row():
                confidence = gr.Slider(
                    0.20, 0.80, value=0.40, step=0.05, label="Độ nhạy nhận diện"
                )
                detail = gr.Radio(
                    choices=["Gọn", "Chi tiết"], value="Chi tiết", label="Mức mô tả"
                )
        with gr.Column(scale=5):
            annotated = gr.Image(
                label="Khung hình đã nhận diện (phần xem thêm)", height=390
            )
            scene_text = gr.Textbox(
                label="Mô tả và cảnh báo", lines=8,
                interactive=False, elem_id="scene-text"
            )
    gr.HTML(
        """<div class='warning'><b>Quyền riêng tư:</b> Khi bật camera, các khung hình
        được truyền tới máy chủ để phân tích bằng AI. Ứng dụng không chủ động lưu ảnh
        hoặc video vào cơ sở dữ liệu. Vui lòng không hướng camera vào thông tin nhạy cảm
        hoặc người khác khi chưa được đồng ý. Hãy tắt camera khi không còn sử dụng.
        <br><br><b>Lưu ý an toàn:</b> Đây là nguyên mẫu nghiên cứu, không phải thiết bị
        dẫn đường đã được kiểm định. AI có thể bỏ sót vật cản, không đo khoảng cách thật
        và không xác nhận được lối đi an toàn. Không dùng lời mô tả để tự quyết định
        rẽ trái/phải; tiếp tục dùng gậy trắng/chó dẫn đường và phương pháp định hướng
        quen thuộc. Nên thử nghiệm cùng người hỗ trợ.</div>"""
    )
    webcam.stream(
        fn=analyze_frame,
        inputs=[webcam, confidence, detail],
        outputs=[annotated, scene_text],
        time_limit=3600,
        stream_every=2.0,
        concurrency_limit=1
    )
    demo.load(fn=None, js=VOICE_JS)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "10000")),
        share=False,
        debug=False
    )
