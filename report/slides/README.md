# Slide thuyết trình LaTeX Beamer

Mã nguồn: [GTSRB-Low-Light-Preprocessing](https://github.com/kttt294/GTSRB-Low-Light-Preprocessing).

Demo: [image-preprocessing.techvi.io.vn](https://image-preprocessing.techvi.io.vn), có liên kết bấm được trên trang đầu, slide website demo và slide kết thúc.

33 slide tiếng Việt, tỉ lệ 16:9, khoảng 20–25 phút tùy thời lượng giải thích phương pháp. Thiết kế theo phong cách bài giảng: nền trắng, dải tiêu đề xám nhạt, tiêu đề màu xanh dương, footer mỏng gồm bốn ô: Nhóm 2, Xử lý ảnh, trích dẫn paper (nếu có) và số trang. Nội dung tập trung xử lý ảnh, toàn bộ kết quả nhận dạng dùng ResNet-18.

- `main.tex`: mã nguồn LaTeX, có ghi chú `\note{...}` cho từng slide.
- `../thuyet_trinh.pdf`: bản trình chiếu cuối.
- `../loi_dan.md`: lời dẫn cá nhân lưu local; ghi chú theo từng trang nằm trong `main.tex`.
- `images/`: hình thực nghiệm lấy từ notebook đã chạy, logo từ paper mẫu.
- `data/`: số liệu nguồn dùng đối chiếu các bảng và biểu đồ.
- `build.ps1`: biên dịch hai lần bằng XeLaTeX và xuất PDF.

## Biên dịch

Từ gốc dự án, chạy PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File report/slides/build.ps1
```

Hoặc chạy thủ công tại thư mục `report/slides`:

```powershell
xelatex --disable-installer -interaction=nonstopmode -halt-on-error -output-directory=_build main.tex
xelatex --disable-installer -interaction=nonstopmode -halt-on-error -output-directory=_build main.tex
Copy-Item -LiteralPath _build/main.pdf -Destination ../thuyet_trinh.pdf
```

Máy hiện có MiKTeX, Beamer, PGFPlots và Arial. Không cài thêm dependency. Giữ cả `report/slides` và `report/latex/images/demo` để không thiếu ảnh chụp website. Với Overleaf chọn XeLaTeX; nếu không có Arial, đổi `Arial` thành `TeX Gyre Heros` trong hai lệnh đặt font.

## Nội dung và nguồn

Các kết quả được đối chiếu với báo cáo hiện tại và bằng chứng từ ZIP Kaggle đã kiểm chứng. Phân biệt subset 430 ảnh với full-test 12.630 ảnh; không ghép metric ảnh của phạm vi này với F1 của phạm vi khác. Slide thứ tự xử lý trình bày giới hạn của khoảng tin cậy bootstrap khi lấy mẫu checkpoint. Slide website sử dụng ảnh chụp thật do nhóm cung cấp.

Thanh footer chỉ hiển thị paper liên quan theo tác giả, hội nghị/tạp chí và năm; có thể bấm để mở bài báo. Slide không có paper để trống ô giữa. Thông tin truy vết notebook/code được giữ trong `\note`, không xuất hiện trong PDF trình chiếu. Báo cáo đầy đủ và dẫn nguồn học thuật nằm ở `../bao_cao.pdf`.


## Cách đọc slide

Mỗi trang nêu rõ cơ chế, cách áp dụng trong dự án và ý nghĩa hoặc giới hạn. Các mục dữ liệu giải thích ROI, lề 5%, track và cách ly; hai trang mô phỏng phân biệt từng bước với ba mức suy giảm. Phần đánh giá nêu kết quả ảnh gốc, giới hạn của mô hình cố định và cách hiểu PSNR/SSIM so với Accuracy/Macro-F1. Ảnh demo nằm ở trang 32, kèm liên kết trực tiếp.
