import streamlit as st
import cv2
import numpy as np
import os
import re
from PIL import Image
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

# --- 1. 配置页面与路径 ---
st.set_page_config(page_title="PCB Detection System", layout="wide")

# 路径改为相对路径（基于你创建的目录结构）
SE_MODEL_PATH = "models/best_se.pt"
CBAM_MODEL_PATH = "models/best_cbam.pt"
TEMPLATE_DIR = "data/detection"


# --- 2. 初始化环境与模型 ---
def register_modules():
    """注册自定义注意力机制模块"""
    try:
        from modules import CBAM, SEAttention
        setattr(tasks, 'CBAM', CBAM)
        setattr(tasks, 'SEAttention', SEAttention)
    except Exception as e:
        st.warning(f"模块注册提醒 (确保 modules.py 在根目录): {e}")


@st.cache_resource
def load_model(path):
    """缓存模型加载，避免每次刷新页面都重新加载"""
    if os.path.exists(path):
        return YOLO(path)
    return None


register_modules()
model_se = load_model(SE_MODEL_PATH)
model_cbam = load_model(CBAM_MODEL_PATH)


# --- 3. 核心算法逻辑 (从你的 Tkinter 代码中迁移) ---
def run_sift_logic(img_bgr, img_name):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    gray_large = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kp_large, des_large = sift.detectAndCompute(gray_large, None)

    found_data = []
    numbers = re.findall(r'\d+', img_name)

    if numbers and os.path.exists(TEMPLATE_DIR):
        target_id = numbers[0]
        for f in os.listdir(TEMPLATE_DIR):
            if f.startswith(f"{target_id}_") and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                tmp_path = os.path.join(TEMPLATE_DIR, f)
                # 处理中文路径读取
                img_tmp = cv2.imdecode(np.fromfile(tmp_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img_tmp is None: continue

                kp_tmp, des_tmp = sift.detectAndCompute(img_tmp, None)
                if des_tmp is None or des_large is None: continue

                matches = bf.match(des_tmp, des_large)
                if len(matches) > 15:
                    src_pts = np.float32([kp_tmp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_large[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if M is not None:
                        h, w = img_tmp.shape[:2]
                        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)
                        cv2.polylines(img_bgr, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

                        x_coords = [p[0][0] for p in dst]
                        y_coords = [p[0][1] for p in dst]
                        bbox = f"({int(min(x_coords))},{int(min(y_coords))})"
                        found_data.append(["SIFT", f"Match: {f}", f"{len(matches)} pts", bbox])

    return img_bgr, found_data


# --- 4. 网页界面布局 ---
st.title("PCB 智能检测系统 | PCB Detection Platform")
st.markdown("---")

# 侧边栏
st.sidebar.header("设置 / Settings")
model_type = st.sidebar.radio("选择模型 / Select Model", ["Model 1 (SE)", "Model 2 (CBAM)"])
conf_val = st.sidebar.slider("置信度阈值 / Confidence", 0.1, 0.9, 0.15)

# 主界面上传
uploaded_file = st.file_uploader("上传 PCB 图片 / Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 加载图片
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("检测结果 / Detection Result")
        if st.button("开始检测 / Run Detection"):
            # 选择模型
            active_model = model_se if "SE" in model_type else model_cbam

            if active_model:
                # 1. YOLO 预测 (强制 CPU 以适配云端)
                results = active_model.predict(img_bgr, conf=conf_val, device='cpu')
                res_img = results[0].plot()

                # 2. SIFT 逻辑
                final_img, sift_results = run_sift_logic(res_img, uploaded_file.name)

                # 3. 显示结果图
                st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), use_container_width=True)

                # 4. 数据展示
                with col2:
                    st.subheader("检测详情 / Details")
                    table_data = []
                    # 整理 YOLO 数据
                    for i, box in enumerate(results[0].boxes):
                        cls = results[0].names[int(box.cls[0])]
                        score = f"{float(box.conf[0]):.2%}"
                        table_data.append([f"YOLO-{i + 1}", cls, score, "See Box"])

                    # 合并 SIFT 数据
                    for item in sift_results:
                        table_data.append(item)

                    if table_data:
                        st.table(np.array(table_data))
                    else:
                        st.write("未发现异常 / No defects found.")
            else:
                st.error("模型未加载成功，请检查路径。")

    with col1:
        if not st.session_state.get('run_button'):  # 未点击时显示原图
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="待检测图片", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.write("Status: System Ready")