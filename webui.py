import gradio as gr
import insightface
import cv2
import numpy as np
import os
import datetime
import shutil
import glob

# --- 0. 全局設定與初始化 ---
HISTORY_DIR = "history"
FAVORITES_DIR = "favorites"
os.makedirs(HISTORY_DIR, exist_ok=True) # 確保 history 文件夾存在
os.makedirs(FAVORITES_DIR, exist_ok=True) # 確保 favorites 文件夾存在

# --- 1. 初始化 InsightFace 模型 ---
print("InsightFace: Initializing FaceAnalysis model...")
app = insightface.app.FaceAnalysis(name='buffalo_l', root='~/.insightface', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("InsightFace: Model prepared.")

# --- 2. 設定相似度閾值 ---
SIMILARITY_THRESHOLD = 0.65

# --- 3. 輔助函數 ---
def get_sorted_images(directory):
    """通用函數：獲取指定文件夾中的所有圖片，並按修改時間降序排列"""
    files = glob.glob(os.path.join(directory, '*.[jp][pn]g')) # 支持 jpg, jpeg, png
    if not files:
        return []
    files.sort(key=os.path.getmtime, reverse=True)
    return files

def get_history_images():
    """獲取歷史圖片"""
    return get_sorted_images(HISTORY_DIR)

def get_favorites_images():
    """獲取收藏圖片"""
    return get_sorted_images(FAVORITES_DIR)

# --- 4. 核心功能函數 ---
def save_and_compare_faces(image1_path, image2_path):
    """保存上傳的圖片到 history，然後比較人臉。"""
    if image1_path is None or image2_path is None:
        return "Please upload two images.", 0.0, None, None, get_history_images()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    saved_path1 = os.path.join(HISTORY_DIR, f"{timestamp}_1.jpg")
    saved_path2 = os.path.join(HISTORY_DIR, f"{timestamp}_2.jpg")
    shutil.copy(image1_path, saved_path1)
    shutil.copy(image2_path, saved_path2)

    img1 = cv2.imread(saved_path1)
    img2 = cv2.imread(saved_path2)

    if img1 is None or img2 is None:
        return "Unable to read image.", 0.0, None, None, get_history_images()

    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    faces1 = app.get(img1_rgb)
    faces2 = app.get(img2_rgb)

    if not faces1:
        return "No faces detected in image 1.", 0.0, img1, None, get_history_images()
    if not faces2:
        return "No faces detected in image 2.", 0.0, None, img2, get_history_images()

    face1 = faces1[0]
    face2 = faces2[0]
    embedding1 = face1.normed_embedding
    embedding2 = face2.normed_embedding
    similarity = np.dot(embedding1, embedding2)

    if similarity > SIMILARITY_THRESHOLD:
        result_text = f"Result: The same person. (Similarity: {similarity:.4f} > Threshold: {SIMILARITY_THRESHOLD})"
    else:
        result_text = f"Result: Not the same person (Similarity: {similarity:.4f} <= Threshold: {SIMILARITY_THRESHOLD})"

    bbox1 = face1.bbox.astype(np.int32)
    bbox2 = face2.bbox.astype(np.int32)
    cv2.rectangle(img1, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]), (0, 255, 0), 2)
    cv2.putText(img1, "Face 1", (bbox1[0], bbox1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.rectangle(img2, (bbox2[0], bbox2[1]), (bbox2[2], bbox2[3]), (0, 255, 0), 2)
    cv2.putText(img2, "Face 2", (bbox2[0], bbox2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return result_text, f"{similarity:.4f}", img1, img2, get_history_images()

def clear_history():
    """清除所有歷史圖片"""
    files = glob.glob(os.path.join(HISTORY_DIR, '*'))
    for f in files: os.remove(f)
    print("History cleared.")
    gr.Info("History has been cleared!")
    return []

def use_image(selected_image_path, target_tab_index):
    """將選中的圖片路徑返回，並切換到指定標籤頁。"""
    if selected_image_path is None:
        gr.Warning("Please select an image first!")
        return None, gr.Tabs() # 保持在當前頁面
    return selected_image_path, gr.Tabs(selected=target_tab_index)

# --- MODIFIED FUNCTION ---
def add_to_favorites(selected_history_image):
    """將選中的歷史圖片複製到收藏夾，並顯示通知"""
    if not selected_history_image:
        gr.Warning("Please select an image from the history first!")
    elif os.path.exists(selected_history_image):
        filename = os.path.basename(selected_history_image)
        dest_path = os.path.join(FAVORITES_DIR, filename)
        if not os.path.exists(dest_path):
            shutil.copy(selected_history_image, dest_path)
            gr.Info(f"'{filename}' has been added to Favorites!")
        else:
            gr.Warning(f"'{filename}' is already in Favorites.")
    else:
        gr.Error("The selected image file does not exist!")

    return get_favorites_images() # 無論如何都返回更新後的收藏列表以刷新UI

def delete_favorite(selected_favorite_image):
    """從收藏夾中刪除選中的圖片"""
    if selected_favorite_image and os.path.exists(selected_favorite_image):
        filename = os.path.basename(selected_favorite_image)
        try:
            file_name = os.path.basename(selected_favorite_image)
            favorites_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'favorites')
            favorite_path = os.path.join(favorites_directory, file_name)
            os.remove(favorite_path)
            gr.Info(f"'{filename}' has been deleted from Favorites.")
        except OSError as e:
            gr.Error(f"Error deleting file: {e}")
    else:
        gr.Warning("Please select an image from favorites to delete!")
        
    return get_favorites_images() # 返回更新後的收藏列表以刷新UI

# --- 5. 構建 Gradio 界面 (使用 Blocks) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Facial Similarity Comparison and Judgment Applications")
    gr.Markdown(f"Upload two face images and I will determine whether they are of the same person. The default cosine similarity threshold is {SIMILARITY_THRESHOLD}. The higher the score, the more similar they are.")

    selected_history_image_state = gr.State(None)
    selected_favorite_image_state = gr.State(None)

    with gr.Tabs() as tabs:
        with gr.Tab("Comparison", id=0):
            with gr.Row():
                with gr.Column():
                    image1_input = gr.Image(type="filepath", label="Image 1")
                    image2_input = gr.Image(type="filepath", label="Image 2")
                    compare_btn = gr.Button("Compare", variant="primary")
                with gr.Column():
                    result_text = gr.Textbox(label="Result")
                    similarity_text = gr.Textbox(label="Cosine similarity")
                    output_img1 = gr.Image(label="Image 1 (face detected)", type="numpy")
                    output_img2 = gr.Image(label="Image 2 (face detected)", type="numpy")

        with gr.Tab("History", id=1):
            history_gallery = gr.Gallery(
                label="Upload History (click to select)", 
                value=get_history_images,
                columns=6, object_fit="contain", height="auto"
            )
            with gr.Row():
                add_to_favorites_btn = gr.Button("Add to Favorites", variant="secondary")
                use_as_img1_btn_hist = gr.Button("Use as Image 1", variant="secondary")
                use_as_img2_btn_hist = gr.Button("Use as Image 2", variant="secondary")
                clear_history_btn = gr.Button("Remove all uploads", variant="stop")

        with gr.Tab("Favorites", id=2):
            favorites_gallery = gr.Gallery(
                label="Favorites (click to select)",
                value=get_favorites_images,
                columns=6, object_fit="contain", height="auto"
            )
            with gr.Row():
                use_as_img1_btn_fav = gr.Button("Use as Image 1", variant="secondary")
                use_as_img2_btn_fav = gr.Button("Use as Image 2", variant="secondary")
                delete_fav_btn = gr.Button("Delete", variant="stop")
    
    # --- 6. 綁定組件事件 ---

    compare_btn.click(
        fn=save_and_compare_faces,
        inputs=[image1_input, image2_input],
        outputs=[result_text, similarity_text, output_img1, output_img2, history_gallery]
    )

    # --- DO NOT CHANGE THIS FUNCTION ---
    def on_gallery_select(evt: gr.SelectData):
        return evt.value['image']['path'] if evt.value else None

    history_gallery.select(fn=on_gallery_select, inputs=None, outputs=[selected_history_image_state])
    
    add_to_favorites_btn.click(
        fn=add_to_favorites,
        inputs=[selected_history_image_state],
        outputs=[favorites_gallery]
    )
    
    use_as_img1_btn_hist.click(fn=lambda path: use_image(path, 0), inputs=[selected_history_image_state], outputs=[image1_input, tabs])
    use_as_img2_btn_hist.click(fn=lambda path: use_image(path, 0), inputs=[selected_history_image_state], outputs=[image2_input, tabs])
    clear_history_btn.click(fn=clear_history, inputs=None, outputs=[history_gallery])

    favorites_gallery.select(fn=on_gallery_select, inputs=None, outputs=[selected_favorite_image_state])
    
    delete_fav_btn.click(
        fn=delete_favorite,
        inputs=[selected_favorite_image_state],
        outputs=[favorites_gallery]
    )
    
    use_as_img1_btn_fav.click(fn=lambda path: use_image(path, 0), inputs=[selected_favorite_image_state], outputs=[image1_input, tabs])
    use_as_img2_btn_fav.click(fn=lambda path: use_image(path, 0), inputs=[selected_favorite_image_state], outputs=[image2_input, tabs])

# --- 7. 啟動 Gradio 應用 ---
if __name__ == "__main__":
    print("Gradio: The application will be launched locally.")
    print("Open Local URL with your browser.")
    demo.launch(server_name="0.0.0.0", share=False)