# Base image
FROM python:3.11-slim

# Tùy chọn: Cài thêm hệ thống để đảm bảo pip hoạt động trơn tru
RUN apt-get update && apt-get install -y gcc build-essential libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirement trước để cache tốt hơn
COPY requirements.txt .

# Cài thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn
COPY . .

# Expose cổng Flask (tuỳ chỉnh nếu bạn chạy cổng khác)
EXPOSE 8051

# Biến môi trường Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8051

# Lệnh chạy app
CMD ["flask", "run"]
