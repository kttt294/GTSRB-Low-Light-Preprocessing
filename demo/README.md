# GTSRB · demo kết quả nghiên cứu

Web tĩnh bằng HTML/CSS/JavaScript: **30 ảnh thuộc 30 lớp × 4 trạng thái = 120 trường hợp**. Ảnh trước/sau và 240 dự đoán được tính trước bằng checkpoint ResNet-18 seed 11 thật. Website không nhận ảnh tải lên, không chạy suy luận trực tiếp và không cần FastAPI/PyTorch trên server.

## Xem local

Từ thư mục gốc repo, chỉ cần Python tiêu chuẩn:

```powershell
python -m demo
```

Mở **http://127.0.0.1:8000**. Đổi cổng bằng `python -m demo --port 8001`. Cũng có thể dùng bất kỳ HTTP static server nào với thư mục `demo/public/`; không mở trực tiếp bằng `file://` vì giao diện cần đọc JSON.

Chọn ảnh mẫu và một trong bốn trạng thái: ảnh gốc, tối nhẹ, tối vừa, tối nặng. Hai ảnh và khung trượt dùng cùng dữ liệu trước/sau. Thanh trượt ở đầu “Trước” hiện ảnh chưa tăng cường; ở đầu “Sau” hiện ảnh qua Gaussian → Gamma. Hàng kết quả gồm dự đoán trước, dự đoán sau và ground truth. Giao diện dùng nhãn “Demo”.

## Deploy Vercel

Import repo GitHub vào Vercel, dùng các thiết lập sau:

| Thiết lập | Giá trị |
|---|---|
| Root Directory | `demo/public` |
| Framework Preset | `Other` |
| Build Command | Không chạy build |
| Install Command | Không cài dependency |
| Output Directory | `.` |

`public/vercel.json` đã khai báo cấu hình tĩnh tương ứng. Thư mục này chứa sẵn mọi tài nguyên, không cần biến môi trường, key, checkpoint, Python hay Node.js. Không chọn thư mục gốc toàn repo vì ở đó còn mã thí nghiệm Python.

Có thể dùng Vercel CLI nếu đã đăng nhập:

```powershell
vercel demo/public
```

Lệnh trên tạo preview; dùng `--prod` khi muốn xuất bản production. Hướng dẫn chính thức: [Vercel deployments](https://vercel.com/docs/deployments).

## Tái tạo kết quả tính sẵn

Chỉ cần bước này khi đổi mẫu, checkpoint hoặc phương pháp. Dùng môi trường nghiên cứu hiện có, hoặc cài `demo/requirements.txt` vào môi trường Python có cặp PyTorch/torchvision tương thích. Chạy từ gốc repo:

```powershell
python -m demo.prepare
python -m demo.export_static
python -m unittest demo.tests -v
```

Không train lại. `prepare` dùng ZIP Output phiên 2 của [notebook Kaggle](https://www.kaggle.com/code/kieuthithutrang/gtsrb-low-light-enhancement-for-traffic-signs) tại `outputs/kaggle-20261025-v2/gtsrb-notebook-results.zip` cùng `GTSRB_Final_Test_Images.zip` và `GTSRB_Final_Test_GT.zip` tại `data/raw/gtsrb/`. Có thể đổi đường dẫn bằng `python -m demo.prepare --archive <zip> --data-root <thư-mục>`.

Nguồn ảnh và nhãn: [GTSRB chính thức trên archive của tác giả](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html). Giữ sáu ví dụ trong báo cáo và bổ sung 24 lớp khác, lấy ID test đầu tiên theo quy tắc cố định, không chọn dựa trên dự đoán. Checksum của archive và checkpoint được kiểm tra trước khi dùng.

Export giữ float32 khi xử lý/suy luận; chỉ lượng tử hóa PNG để hiển thị. 60 dự đoán ở mức vừa được so lại với kết quả full-test đã lưu. `public/data/provenance.json` ghi hash checkpoint, mã xử lý và từng tài nguyên; `samples.json` ghi ảnh, nhãn và dự đoán cho đủ 120 trường hợp. Đây là minh họa có chọn mẫu, không thay thế metric trên toàn tập test.

Pipeline cố định: Gaussian 5×5, sigma 0,8 → Gamma 0,8, được chọn trên validation. Ba mức suy giảm dùng đúng gamma/Poisson/Gaussian của nghiên cứu, noise seed 101. Cùng một CNN seed 11 dự đoán trên ảnh trước và sau xử lý. Demo không bổ sung kết quả camera đêm thực, video hoặc detector toàn cảnh.

## Cấu trúc

- `public/`: toàn bộ web được deploy, gồm giao diện, 240 PNG và JSON kết quả/nguồn.
- `prepare.py`, `engine.py`, `export_static.py`: chuẩn bị và suy luận offline.
- `__main__.py`: HTTP server local bằng thư viện chuẩn; `tests.py`: kiểm tra tính toàn vẹn gói tĩnh.
- `.cache/`, `qa/`: checkpoint/cache và ảnh kiểm tra chỉ lưu local, đã gitignore.

Mã nguồn: [GTSRB-Low-Light-Preprocessing](https://github.com/kttt294/GTSRB-Low-Light-Preprocessing).
