import pandas as pd
import pickle
import os
import json
import time
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import threading
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RecommendationService')

app = Flask(__name__)
CORS(app)

# Đường dẫn lưu trữ dữ liệu và mô hình
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Tạo thư mục nếu chưa tồn tại
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Đường dẫn file dữ liệu
PRODUCT_CSV = os.path.join(DATA_DIR, 'product.csv')

# Đường dẫn file mô hình
MODEL_FILE = os.path.join(MODEL_DIR, 'recommendation_model.pkl')
LAST_UPDATED_FILE = os.path.join(MODEL_DIR, 'last_updated.txt')  # Lưu trữ thời gian cập nhật cuối cùng, phục vụ kiểm tra trạng thái

# Biến toàn cục lưu trữ mô hình đã được load
model_data = None
training_lock = threading.Lock()
model_load_time = None


def load_model():
    """Load mô hình từ file nếu đã tồn tại"""
    global model_data, model_load_time

    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'rb') as f:
                model_data = pickle.load(f)
                logger.info(f"Model loaded from {MODEL_FILE}")
                if os.path.exists(LAST_UPDATED_FILE):
                    with open(LAST_UPDATED_FILE, 'r') as f:
                        model_load_time = float(f.read().strip())
                return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    return False


def train_model(force=False):
    """Train mô hình từ file CSV và lưu lại"""
    global model_data, model_load_time

    # Kiểm tra xem có cần train lại hay không
    if not force and model_data is not None:
        return True

    # Chỉ cho phép một thread thực hiện training tại một thời điểm, tránh gây race condition
    if not training_lock.acquire(blocking=False):
        logger.info("Another training process is already running")
        return False

    try:
        logger.info("Starting model training...")

        # Kiểm tra file dữ liệu
        if not os.path.exists(PRODUCT_CSV):
            logger.error(f"Product CSV file not found at {PRODUCT_CSV}")
            return False

        # Đọc dữ liệu từ CSV
        df_products = pd.read_csv(PRODUCT_CSV)

        # Xử lý dữ liệu thiếu
        df_products = df_products.fillna('')

        # Tạo cột đặc trưng kết hợp cho content-based filtering
        # Giả sử rằng CSV có các cột như 'name', 'description', 'category', 'brand', 'tags'
        # Điều chỉnh theo cấu trúc thực tế của file CSV của bạn
        feature_cols = ['name', 'description', 'category', 'brand', 'tags']
        available_cols = [col for col in feature_cols if col in df_products.columns]

        if not available_cols:
            logger.error("No usable feature columns found in the CSV")
            return False

        # Tạo cột đặc trưng tổng hợp từ các cột có sẵn
        df_products['features'] = df_products[available_cols].apply(
            lambda row: ' '.join(str(row[col]) for col in available_cols), axis=1
        )

        # Tạo ma trận TF-IDF
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df_products['features'])

        # Tính toán ma trận tương đồng cosine
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Tạo các index để tra cứu nhanh
        indices = pd.Series(df_products.index, index=df_products['id'].astype(str))
        name_indices = pd.Series(df_products.index, index=df_products['name']).drop_duplicates()

        # Lưu mô hình vào file
        model_data = {
            'products': df_products,
            'tfidf': tfidf,
            'tfidf_matrix': tfidf_matrix,
            'cosine_sim': cosine_sim,
            'indices': indices,
            'name_indices': name_indices
        }

        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model_data, f)

        # Lưu thời gian cập nhật
        model_load_time = time.time()
        with open(LAST_UPDATED_FILE, 'w') as f:
            f.write(str(model_load_time))

        logger.info(f"Model trained successfully with {len(df_products)} products")
        return True

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return False
    finally:
        training_lock.release()


def get_recommendations_by_id(product_id, num_recommendations=5):
    """Lấy các sản phẩm tương tự dựa trên ID"""
    if model_data is None:
        return [], "Model not loaded"

    # Chuyển ID sang string để đảm bảo việc so sánh đúng
    product_id = str(product_id)

    try:
        if product_id not in model_data['indices']:
            return [], f"Product ID {product_id} not found"

        idx = model_data['indices'][product_id]
        sim_scores = list(enumerate(model_data['cosine_sim'][idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations + 1]
        product_indices = [i[0] for i in sim_scores]

        # Tạo danh sách kết quả từ DataFrame gốc
        result_df = model_data['products'].iloc[product_indices].copy()
        result_df['similarity'] = [score[1] for score in sim_scores]

        # Chuyển kết quả thành list dictionaries
        results = result_df.to_dict('records')
        return results, None

    except Exception as e:
        logger.error(f"Error getting recommendations by ID: {e}")
        return [], str(e)


def get_recommendations_by_name(product_name, num_recommendations=5):
    """Lấy các sản phẩm tương tự dựa trên tên sản phẩm"""
    if model_data is None:
        return [], "Model not loaded"

    try:
        if product_name not in model_data['name_indices']:
            return [], f"Product name '{product_name}' not found"

        idx = model_data['name_indices'][product_name]
        sim_scores = list(enumerate(model_data['cosine_sim'][idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations + 1]
        product_indices = [i[0] for i in sim_scores]

        # Tạo danh sách kết quả từ DataFrame gốc
        result_df = model_data['products'].iloc[product_indices].copy()
        result_df['similarity'] = [score[1] for score in sim_scores]

        # Chuyển kết quả thành list dictionaries
        results = result_df.to_dict('records')
        return results, None

    except Exception as e:
        logger.error(f"Error getting recommendations by name: {e}")
        return [], str(e)

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World Tôi là hoàng!'


if __name__ == '__main__':
    app.run()
