# 颜值评分系统 - Web应用

基于Flask的在线颜值评分平台，支持通过浏览器上传图片进行评分。

## 功能特点

- 🌐 **在线访问**：通过浏览器即可使用，无需安装
- 📸 **图片上传**：支持拖拽上传或点击选择
- 🎯 **多种评分模式**：
  - 几何特征评分
  - 深度学习评分
  - 两者结合
- 📊 **详细结果**：显示综合得分和详细分析报告
- 🎨 **美观界面**：现代化、响应式设计

## 安装依赖

```bash
pip install -r web_app/requirements.txt
```

## 运行应用

```bash
cd web_app
python app.py
```

然后在浏览器中访问：`http://localhost:5000`

## 部署到服务器

### Windows 系统

**注意：Gunicorn 不支持 Windows 系统（会报 `fcntl` 模块错误）**

在 Windows 上，直接使用 Flask 内置服务器即可：

```bash
python app.py
```

或者使用 `waitress`（Windows 兼容的生产级 WSGI 服务器）：

```bash
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

### Linux/Mac 系统

#### 使用 Gunicorn（推荐）

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### 使用 Docker

```dockerfile
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## API接口

### POST /api/score

上传图片进行评分

**请求参数：**
- `image`: 图片文件（multipart/form-data）
- `mode`: 评分模式（geometric/dl/both）
- `model_path`: 模型路径（可选）

**响应示例：**
```json
{
  "success": true,
  "final_score": 3.85,
  "final_score_100": 77.0,
  "mode": "both",
  "results": {
    "geometric": {
      "overall_score": 3.8,
      "three_regions": 3.5,
      "five_eyes": 4.0,
      "symmetry": 4.0,
      "report": "..."
    },
    "dl": {
      "score": 3.9,
      "score_100": 78.0,
      "model_name": "resnet18",
      "device": "cuda"
    }
  }
}
```

### GET /api/models

获取可用的模型列表

**响应示例：**
```json
{
  "models": [
    {
      "name": "resnet18",
      "path": "outputs/resnet18/checkpoints/resnet18_best.pth"
    }
  ]
}
```

## 注意事项

1. **文件大小限制**：默认最大上传16MB
2. **模型路径**：确保模型文件路径正确
3. **GPU支持**：如果有GPU，会自动使用CUDA加速
4. **临时文件**：上传的图片会在处理后被自动删除

## 与TensorBoard的区别

- **TensorBoard**：用于可视化训练过程（损失曲线、指标等），仅本地使用
- **Web应用**：提供在线服务，用户可以通过浏览器上传图片并获得评分结果
