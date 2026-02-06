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

# 路径保持不变
SE_MODEL_PATH = "models/best_se.pt"
CBAM_MODEL_PATH = "models/best_cbam.pt"
TEMPLATE_DIR = "data/detection"

# --- 2. 初始化环境与模型 ---
def register_modules():
    try:
        from modules import CBAM, SEAttention
        setattr(tasks, 'CBAM', CBAM)
        setattr(tasks, 'SEAttention', SEAttention)
    except Exception as e:
        st.warning(f"模块注册提醒: {e}")

@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return YOLO(path)
    return None

register_modules()
model_se = load_model(SE_MODEL_PATH)
model_cbam = load_model(CBAM_MODEL_PATH)

# --- 3. 核心算法逻辑 (同步 GUI 的 Algorithm 1/2 与网格) ---
def run_matching_logic(img_bgr, img_name, algo_type):
    """同步 GUI 逻辑：绘制网格并执行 Algorithm 1 (SIFT) 或 Algorithm 2 (ORB)"""
    H, W = img_bgr.shape[:2]
    
    # 1. 绘制 9x9 绿色网格 (与 GUI 一致)
    for i in range(1, 9):
        cv2.line(img_bgr, (0, int(i * H / 9)), (W, int(i * H / 9)), (0, 255, 0), 1)
        cv2.line(img_bgr, (int(i * W / 9), 0), (int(i * W / 9), H), (0, 255, 0), 1)
    
    # 显示代号名称
    display_algo_name = "Algorithm 1" if algo_type == "SIFT" else "Algorithm 2"
    cv2.putText(img_bgr, f"Analysis Mode: {display_algo_name}", (15, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 初始化特征器
    if algo_type == "SIFT":
        engine = cv2.SIFT_create()
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    else:
        engine = cv2.ORB_create(nfeatures=5000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    gray_large = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kp_large, des_large = engine.detectAndCompute(gray_large, None)

    found_data = []
    numbers = re.findall(r'\d+', img_name)

    if numbers and des_large is not None and os.path.exists(TEMPLATE_DIR):
        target_id = numbers[0]
        for f in os.listdir(TEMPLATE_DIR):
            if f.startswith(f"{target_id}_") and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                tmp_path = os.path.join(TEMPLATE_DIR, f)
                img_tmp = cv2.imdecode(np.fromfile(tmp_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img_tmp is None: continue

                kp_tmp, des_tmp = engine.detectAndCompute(img_tmp, None)
                if des_tmp is None: continue

                matches = matcher.match(des_tmp, des_large)
                if len(matches) > 12:
                    src_pts = np.float32([kp_tmp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_large[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if M is not None:
                        h, w = img_tmp.shape[:2]
                        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)
                        cv2.polylines(img_bgr, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

                        center_coords = f"({int(np.mean(dst[:,0,0]))}, {int(np.mean(dst[:,0,1]))})"
                        found_data.append([display_algo_name, f"Match: {f}", f"{len(matches)} pts", center_coords])

    return img_bgr, found_data

# --- 4. 网页界面布局 (同步 GUI 样式) ---
st.title("PCB 缺陷检测平台 | PCB Defect Detection Platform")
st.markdown("---")

# 侧边栏配置：隐藏具体算法名，使用代号
st.sidebar.header("控制面板 / Control Panel")
model_option = st.sidebar.radio("选择检测模型 / Detection Model", ["Model 1", "Model 2"])
algo_option = st.sidebar.radio("选择分析算法 / Analysis Algorithm", ["Algorithm 1", "Algorithm 2"])
conf_val = st.sidebar.slider("置信度阈值 / Confidence", 0.05, 0.95, 0.15)

# 映射逻辑
model_map = {"Model 1": "SE", "Model 2": "CBAM"}
algo_map = {"Algorithm 1": "SIFT", "Algorithm 2": "ORB"}

uploaded_file = st.file_uploader("上传 PCB 图片 / Upload PCB Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns([3, 2])

    with col1:
        if st.button("开始执行检测 / Run Hybrid Detection", type="primary"):
            # 获取对应的实际模型和算法名
            real_model_name = model_map[model_option]
            real_algo_name = algo_map[algo_option]
            active_model = model_se if real_model_name == "SE" else model_cbam

            if active_model:
                # 1. YOLO 推理
                results = active_model.predict(img_bgr, conf=conf_val, device='cpu')
                res_img = results[0].plot()

                # 2. 传统视觉匹配 + 绘制网格
                final_img, match_results = run_matching_logic(res_img, uploaded_file.name, real_algo_name)

                # 3. 显示合成后的图
                st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), caption=f"Result: {model_option} + {algo_option}", use_container_width=True)

                # 4. 右侧展示详细表格数据
                with col2:
                    st.subheader("检测报告 / Detection Report")
                    summary_data = []
                    
                    # YOLO 结果
                    for i, box in enumerate(results[0].boxes):
                        cls = results[0].names[int(box.cls[0])]
                        score = f"{float(box.conf[0]):.2%}"
                        summary_data.append([f"{model_option}", cls, score, "Confirmed"])

                    # 匹配算法结果
                    for item in match_results:
                        summary_data.append(item)

                    if summary_data:
                        import pandas as pd
                        df = pd.DataFrame(summary_data, columns=["方法/Model", "类别/Class", "置信度/Score", "坐标/Status"])
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("未检测到异常。")
            else:
                st.error("模型加载失败，请检查 models/ 文件夹下的 .pt 文件。")
        else:
            # 默认显示原图，并画上初始网格预览
            preview_img = img_bgr.copy()
            H, W = preview_img.shape[:2]
            for i in range(1, 9):
                cv2.line(preview_img, (0, int(i * H / 9)), (W, int(i * H / 9)), (0, 255, 0), 1)
                cv2.line(preview_img, (int(i * W / 9), 0), (int(i * W / 9), H), (0, 255, 0), 1)
            st.image(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB), caption="待检测图片 (已叠加参考网格)", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info(f"System Mode: Hybrid Analysis\n\nReady for {model_option} & {algo_option}")
