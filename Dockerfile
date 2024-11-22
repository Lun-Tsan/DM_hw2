# 使用官方的 Python 基礎映像（這裡使用 Python 3.8）
FROM python:3.8-slim

# 設置工作目錄
WORKDIR /app

# 環境變量設置
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 更新系統並安裝必要的系統依賴
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 依賴項
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 複製項目代碼到容器內
COPY . .

# （可選）暴露端口（如果使用 Jupyter Notebook）
EXPOSE 8888

# 默認命令（可根據需要修改）
CMD ["bash"]
