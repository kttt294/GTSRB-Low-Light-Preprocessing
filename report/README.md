# Báo cáo bài tập lớn môn Xử lý ảnh

Mã nguồn: [GTSRB-Low-Light-Preprocessing](https://github.com/kttt294/GTSRB-Low-Light-Preprocessing).

Đề tài: **Khảo sát ảnh hưởng của xử lý ảnh đến hiệu năng nhận dạng biển báo giao thông trong điều kiện thiếu sáng.**

- [Báo cáo PDF](bao_cao.pdf) · [Slide PDF](thuyet_trinh.pdf).
- [Demo trực tuyến](https://image-preprocessing.techvi.io.vn).
- [Notebook Kaggle](https://www.kaggle.com/code/kieuthithutrang/gtsrb-low-light-enhancement-for-traffic-signs): phiên 2 đã hoàn tất toàn bộ khảo sát kết hợp.
- `latex/sections/`: nội dung; `latex/tables/` và `latex/images/`: bảng/hình; `slides/main.tex`: nguồn slide.
- `evidence/final/`: bằng chứng hiện tại; `provenance.json`: nguồn và giới hạn kiểm chứng. Các CSV lịch sử ngoài `final/` không được dùng để sinh bảng hiện tại.

Nghiên cứu so sánh bảy enhancer, Gaussian/bilateral/NLM và hai thứ tự với Gamma/CLAHE trên cùng ba checkpoint ResNet-18. Phần khử nhiễu nằm trong phương pháp, thực nghiệm, thảo luận và kết luận; không còn phụ lục dự kiến. Validation chọn Gaussian 5×5, sigma 0,8 rồi Gamma 0,8; không chọn lại bằng test.

Đã tính lại metric từ **5.847.990 dòng dự đoán**: baseline subset/full-test và khảo sát kết hợp subset/full-test. Một phần đối chứng được dùng lại, không phải quan sát độc lập mới. So khớp 954 confusion matrix lưu trong ZIP, kiểm tra hash ba CNN và mọi file xuất. Tương quan dùng chất lượng ảnh và metric nhận dạng trên cùng phạm vi. Hình xử lý minh họa được phân biệt với bằng chứng dự đoán hiện tại trong provenance.

## Tái tạo bảng và hình

Tải `gtsrb-notebook-results.zip` từ Output của notebook, giữ nguyên gói gốc. Từ thư mục gốc repo:

```powershell
python -m src.tools.verified_results --archive outputs/kaggle-20261025-v2/gtsrb-notebook-results.zip --output outputs/verified-new --require-extension
python report/latex/build_assets.py --archive outputs/kaggle-20261025-v2/gtsrb-notebook-results.zip --evidence outputs/verified-new
```

Thư mục kiểm chứng phải mới. Bộ sinh bảng chỉ chấp nhận ZIP có hash khớp biên bản và đủ cả bốn phạm vi; không trích số từ notebook lịch sử. Dùng Python, NumPy, Matplotlib và PyTorch của dự án.

Trang đối chiếu sáu mẫu dùng ảnh test và nhãn chính thức. Đặt hai archive `GTSRB_Final_Test_Images.zip` và `GTSRB_Final_Test_GT.zip` trong `data/raw/gtsrb/`, rồi chạy:

```powershell
python report/latex/visual_cases.py --archive outputs/kaggle-20261025-v2/gtsrb-notebook-results.zip
```

Công cụ kiểm tra checksum, ROI, nhãn và suy luận lại 12 đầu vào bằng checkpoint 11; không huấn luyện lại. Giữ hai ID nhỏ nhất mỗi nhóm được giúp/bị làm sai/vẫn sai, dùng cùng thang hiển thị. Kết quả kiểm chứng và hash ảnh nằm trong `evidence/final/visual_cases_audit.json`. Các archive ảnh gốc chỉ giữ local.

Phân tích độ bất định và tập trung dự đoán dùng cùng ZIP đã kiểm chứng:

```powershell
python report/latex/review_analysis.py --archive outputs/kaggle-20261025-v2/gtsrb-notebook-results.zip --repetitions 2000
```

Công cụ ghép 682.020 dòng dự đoán của hai thứ tự Gaussian–CLAHE, đối chiếu từng ma trận và tính hai khoảng percentile 95%: lấy mẫu ảnh phân tầng theo lớp, và lấy mẫu thêm nhóm checkpoint. Các noise seed không được coi là ảnh độc lập. CSV, hash và giới hạn suy luận ở `evidence/final/review_analysis_audit.json`; cache dự đoán và các lượt bootstrap chỉ giữ local. Công cụ cũng sinh bảng tập trung dự đoán vào lớp 5 và các hình confusion/F1 có cỡ chữ lớn hơn. Không train lại hoặc chọn lại pipeline bằng test.

## Biên dịch PDF

Chạy `powershell -File report/latex/build.ps1` với XeLaTeX; hoặc truyền `-Compiler <đường-dẫn-tuyệt-đối-tectonic.exe>`. Build trong `_build/`, sau đó cập nhật `bao_cao.pdf`. Tectonic có thể cần mạng để tải gói; XeLaTeX cần cài đủ gói trước.

Slide: từ `report/slides/`, dùng `tectonic --keep-logs --outdir _build main.tex`, rồi chép `_build/main.pdf` thành `report/thuyet_trinh.pdf`.

Báo cáo dùng A4, Times New Roman 12,5 bp; lề trái/phải 30/20 mm, trên/dưới 32/30 mm; bìa Nhóm 2, môn Xử lý ảnh, giảng viên TS. Nguyễn Ngọc Giang. Slide dùng Arial, khung 16:9. Khi dùng Overleaf chọn XeLaTeX và cung cấp font hợp lệ hoặc đổi sang font tương đương.

Git giữ PDF cuối, nguồn LaTeX, hình và bảng bằng chứng gọn. Ảnh GTSRB, checkpoint, ZIP thô, cache, mẫu và kết quả QA giữ local. Dữ liệu nghiên cứu là ROI thiếu sáng mô phỏng; chưa có đánh giá camera đêm thực, video hoặc detector.
