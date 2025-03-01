import logging
import traceback
import time
from pathlib import Path
import pickle

import pandas as pd
from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
from flask_cors import CORS

# Cấu hình Flask và CORS
app = Flask(__name__)
CORS(app)

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Trạng thái API toàn cục
api_status = {
    "status": "initializing",  # trạng thái: initializing, loading, ready, error
    "model_loaded": False,
    "last_updated": None,
    "error_message": None,
    "data_info": None
}

# Cấu hình thư mục dữ liệu và mô hình
DATA_DIR = Path(r"D:\data_train")
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Định nghĩa đường dẫn file dữ liệu
df_product_path = DATA_DIR / 'Product.csv'
df_description_path = DATA_DIR / 'Description.csv'
df_category_path = DATA_DIR / 'Category.csv'

# Đường dẫn file mô hình
model_data_path = MODEL_DIR / 'model_data.pkl'

# Biến lưu dữ liệu toàn cục
cleaned_df = None
indices = None
similarity_matrix = None  # Trước đây tên cosine_sim

def update_status(status, model_loaded=None, error_message=None, data_info=None):
    """Cập nhật trạng thái API và log thông tin"""
    global api_status
    api_status["status"] = status
    api_status["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if model_loaded is not None:
        api_status["model_loaded"] = model_loaded
    if error_message is not None:
        api_status["error_message"] = error_message
    if data_info is not None:
        api_status["data_info"] = data_info
    logging.info(f"API Status: {status}, Model loaded: {api_status['model_loaded']}")

def check_data_files():
    """Kiểm tra sự tồn tại của các file dữ liệu"""
    files_exist = {
        "product": df_product_path.exists(),
        "description": df_description_path.exists(),
        "category": df_category_path.exists()
    }
    return files_exist, all(files_exist.values())

def load_data():
    """Load và xử lý dữ liệu, tạo mô hình và lưu lại nếu cần"""
    global cleaned_df, indices, similarity_matrix
    update_status("LD0001: loading")

    files_status, all_exist = check_data_files()
    if not all_exist:
        error_message = f"Missing data files: {files_status}"
        logging.error("LD0002: " + error_message)
        update_status("error", model_loaded=False, error_message=error_message)
        return False

    # Thử load mô hình đã được lưu
    if model_data_path.exists():
        try:
            logging.info("Loading pre-trained model data...")
            with model_data_path.open('rb') as f:
                data = pickle.load(f)
                cleaned_df = data.get('cleaned_df')
                indices = data.get('indices')
                similarity_matrix = data.get('cosine_sim')
            update_status("ready", model_loaded=True,
                          data_info={"products": len(cleaned_df),
                                     "last_model_update": time.ctime(model_data_path.stat().st_mtime)})
            return True
        except Exception as e:
            logging.error(f"Error loading pre-trained data: {e}")
            update_status("error", model_loaded=False, error_message=str(e))

    # Xử lý dữ liệu từ đầu nếu không load được mô hình đã lưu
    try:
        logging.info("Processing data from CSV files at %s", DATA_DIR)
        try:
            df_product = pd.read_csv(df_product_path)
            df_description = pd.read_csv(df_description_path)
            df_category = pd.read_csv(df_category_path)
            logging.info("Data loaded successfully from %s", DATA_DIR)
        except Exception as file_error:
            error_message = f"Error loading data: {file_error}"
            logging.error("LD000-2: " + error_message)
            update_status("error", model_loaded=False, error_message=error_message)
            return False

        logging.debug("df_product columns: %s", df_product.columns.tolist())
        logging.debug("df_description columns: %s", df_description.columns.tolist())
        logging.debug("df_category columns: %s", df_category.columns.tolist())

        # Merge bảng product và description
        merged_df = pd.merge(df_product, df_description, on='id', how='inner')
        logging.info("Merged product & description: %d rows", merged_df.shape[0])

        # Xử lý dữ liệu của bảng category và merge
        category_cleaned_df = df_category.drop(columns=['image', 'status', 'created_at'], errors='ignore')\
                                         .rename(columns={'id': 'category'})
        merged_df2 = pd.merge(category_cleaned_df, merged_df, on='category', how='inner')
        logging.info("Merged with category: %d rows", merged_df2.shape[0])

        # Drop các cột không cần thiết và đổi tên cột
        columns_to_drop = ['slug_x', 'id', 'slug_y', 'category', 'unit', 'original_price',
                           'sale_price', 'expiry_period', 'status', 'created_at',
                           'modified_at', 'image']
        existing_columns = [col for col in columns_to_drop if col in merged_df2.columns]
        cleaned_df = merged_df2.drop(columns=existing_columns, errors='ignore')
        cleaned_df.rename(columns={'name_x': 'category', 'name_y': 'name'}, inplace=True)

        # Xử lý giá trị null
        if 'uses' in cleaned_df.columns:
            cleaned_df['uses'] = cleaned_df['uses'].fillna('')
        logging.info("Cleaned dataframe: %d rows, %d columns", cleaned_df.shape[0], cleaned_df.shape[1])

        # Tạo stop words tiếng Việt
        my_stop_words = ["là", "của", "và", "nhưng", "hay", "hoặc", "tôi", "bạn", "mình",
                         "họ", "nó", "rất", "quá", "lắm", "không", "có", "làm", "được",
                         "tốt", "xấu"]

        # Tạo TF-IDF vectorizer và tính toán ma trận
        tfv = TfidfVectorizer(min_df=3, strip_accents='unicode', analyzer='word',
                              token_pattern=r'\w{1,}', ngram_range=(1, 3),
                              stop_words=my_stop_words)
        logging.info("Calculating TF-IDF matrix...")
        tfv_matrix = tfv.fit_transform(cleaned_df['uses'])
        logging.info("TF-IDF matrix shape: %s", tfv_matrix.shape)

        # Tính toán similarity matrix sử dụng sigmoid kernel
        logging.info("Calculating similarity matrix using sigmoid kernel...")
        similarity_matrix = sigmoid_kernel(tfv_matrix, tfv_matrix)
        logging.info("Similarity matrix shape: %s", similarity_matrix.shape)

        # Tạo mapping từ tên sản phẩm sang chỉ số
        cleaned_df.reset_index(drop=True, inplace=True)
        indices = pd.Series(cleaned_df.index, index=cleaned_df['name']).drop_duplicates()
        logging.info("Created indices mapping for %d products", len(indices))

        # Lưu dữ liệu mô hình để sử dụng lại
        data = {'cleaned_df': cleaned_df, 'indices': indices, 'cosine_sim': similarity_matrix}
        with model_data_path.open('wb') as f:
            pickle.dump(data, f)
        logging.info("Model data saved successfully.")

        update_status("ready", model_loaded=True,
                      data_info={"products": len(cleaned_df), "last_model_update": time.ctime()})
        return True

    except Exception as e:
        logging.error("LD000-3: %s", traceback.format_exc())
        update_status("error", model_loaded=False, error_message=str(e))
        return False

def give_rec(name, count=10):
    """Hàm gợi ý sản phẩm dựa trên tên sản phẩm"""
    global indices, cleaned_df, similarity_matrix
    try:
        idx = indices[name]
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:count+1]
        rec_indices = [i[0] for i in sim_scores]
        recommendations_df = cleaned_df.iloc[rec_indices][['product', 'name']]
        return recommendations_df.to_dict(orient='records')
    except Exception as e:
        logging.error("GR000-1: Error in recommendation: %s", str(e))
        return [{"error": "Error generating recommendations"}]

# Định nghĩa các API endpoint

@app.route('/api/status', methods=['GET'])
def status():
    files_status, _ = check_data_files()
    status_detail = {
        **api_status,
        "file_paths": {"data_dir": str(DATA_DIR), "model_dir": str(MODEL_DIR)},
        "files": {"data_files": files_status, "model_file_exists": model_data_path.exists()}
    }
    return jsonify(status_detail)

@app.route('/api/reload', methods=['GET'])
def reload_data():
    success = load_data()
    return jsonify({"success": success, "status": api_status})

@app.route('/api/recommend/<product_name>', methods=['GET'])
def recommend(product_name):
    if api_status["status"] != "ready":
        return jsonify({
            "status": "error",
            "message": f"API not ready. Current status: {api_status['status']}",
            "details": api_status
        }), 503

    try:
        num_recommendations = request.args.get('count', default=10, type=int)
        recommendations = give_rec(product_name, num_recommendations)
        if isinstance(recommendations, dict) and "error" in recommendations:
            return jsonify({"status": "error", "message": recommendations["error"]}), 500
        return jsonify({
            "status": "success",
            "product_name": product_name,
            "recommendations": recommendations
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/recommend/<product_name>', methods=['GET'])
def simple_recommend(product_name):
    if api_status["status"] != "ready":
        return jsonify({"error": "API not ready", "status": api_status["status"]}), 503
    return jsonify(give_rec(product_name))

@app.route('/api/recommend-by-id/<int:product_id>', methods=['GET'])
def recommend_by_id(product_id):
    if api_status["status"] != "ready":
        return jsonify({
            "status": "error",
            "message": f"API not ready. Current status: {api_status['status']}",
            "details": api_status
        }), 503

    try:
        num_recommendations = request.args.get('count', default=5, type=int)
        product_row = cleaned_df[cleaned_df['id'] == product_id]
        if product_row.empty:
            return jsonify({"status": "error", "message": f"Product with ID {product_id} not found"}), 404
        product_name = product_row['name_y'].iloc[0]
        recommendations = give_rec(product_name, num_recommendations)
        return jsonify({
            "status": "success",
            "product_id": product_id,
            "product_name": product_name,
            "recommendations": recommendations
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/products', methods=['GET'])
def get_products():
    if api_status["status"] != "ready":
        return jsonify({
            "status": "error",
            "message": f"API not ready. Current status: {api_status['status']}",
            "details": api_status
        }), 503

    try:
        limit = request.args.get('limit', default=100, type=int)
        products = cleaned_df[['product', 'name']].head(limit).to_dict(orient='records')
        return jsonify({"status": "success", "count": len(products), "products": products})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    if load_data():
        logging.info("Data loaded successfully. Starting API server...")
    else:
        logging.warning("Starting API server without data loaded. Status will be 'error' until reload.")
    app.run(host='0.0.0.0', port=5000, debug=False)

# Load data khi khởi động ứng dụng
with app.app_context():
    load_data()
