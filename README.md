# MNIST Lightning Hydra Project

## 📌 Tổng quan
Dự án này thực hiện bài toán phân loại chữ số viết tay (MNIST) sử dụng **[Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)**. 

Kiến trúc mô hình và logic huấn luyện được tinh chỉnh để tương thích với cấu trúc tại **[Colab Research - MNIST Lightning](https://colab.research.google.com/drive/1HBlptp22dyCeSMFc_DIj2mqpypkJkx-9)**.

---
## 🚀 Hướng dẫn nhanh (Cài đặt & Chạy)

```bash
# Tạo môi trường bằng Conda
conda create -n mnist-env python=3.9 -y
conda activate mnist-env   

# Cài đặt các thư viện từ file requirements
pip install -r requirements.txt

# Huấn luyện mô hình với cấu hình mặc định từ Hydra
python src/train.py 

# Thay đổi tham số
python src/train.py model.lr=0.0005 trainer.max_epochs=5