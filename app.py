import logging
import traceback
import time
from pathlib import Path
import pickle

import pandas as pd
import psycopg2
from psycopg2 import sql
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
from flask_cors import CORS
from collaborative_filtering import (
    build_collaborative_model,
    item_based_recommendations,
    user_based_recommendations,
    hybrid_recommendations,
    record_user_action,
    snowflakeIdGenerator
)

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
    "data_info": None,
    "collaborative_model_loaded": False,
}

# Cấu hình thư mục mô hình
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Đường dẫn file mô hình
model_data_path = MODEL_DIR / 'model_data.pkl'

# Cấu hình database
DB_CONFIG = {
    # "host": "localhost",
    "host": "172.17.0.1",
    # "database": "AGRI_MARKET",
    "database": "agri_market",
    # "user": "postgres",
    "user": "vuvanhoang",
    "password": "Hoanglam06112003@",
    "port": 5432
}

# Biến lưu dữ liệu toàn cục
cleaned_df = None
indices = None
similarity_matrix = None


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


def get_db_connection():
    """Tạo kết nối đến PostgreSQL database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logging.error(f"Error connecting to database: {e}")
        raise


def load_data_from_db():
    """Load dữ liệu tối thiểu từ PostgreSQL database cho việc train model"""
    try:
        conn = get_db_connection()
        logging.info("Connected to PostgreSQL database")

        # Query để lấy chỉ các trường cần thiết cho việc train
        query = """
            SELECT p.id, p.name, p.category_id as category, c.name as category_name, d.uses
            FROM products p
            JOIN descriptions d ON p.id = d.product_id
            JOIN categories c ON p.category_id = c.id;
        """

        # Đọc dữ liệu vào DataFrame
        df = pd.read_sql(query, conn)

        print(df.head())  # Kiểm tra dữ liệu đọc từ database
        print(f"Dataframe shape: {df.shape}")  # Kiểm tra số dòng và cột

        logging.info(f"Loaded {len(df)} products with minimal fields from database")
        conn.close()

        return df
    except Exception as e:
        logging.error(f"Error loading data from database: {e}")
        raise


def load_data():
    """Load và xử lý dữ liệu, tạo mô hình và lưu lại nếu cần"""
    global cleaned_df, indices, similarity_matrix
    update_status("LD0001: loading")

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

    # Xử lý dữ liệu từ database nếu không load được mô hình đã lưu
    try:
        logging.info("Processing data from PostgreSQL database")
        try:
            df = load_data_from_db()
            logging.info("Data loaded successfully from database")
        except Exception as db_error:
            error_message = f"Error loading data from database: {db_error}"
            logging.error("LD000-2: " + error_message)
            update_status("error", model_loaded=False, error_message=error_message)
            return False

        # Tạo DataFrame đã được làm sạch
        cleaned_df = df.copy()

        # Đổi tên cột nếu cần
        if 'category_name' in cleaned_df.columns:
            cleaned_df.rename(columns={'category_name': 'category_name'}, inplace=True)

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

        try:
            conn = get_db_connection()
            collab_success = build_collaborative_model(conn)
            conn.close()
            api_status["collaborative_model_loaded"] = collab_success
        except Exception as e:
            api_status["collaborative_model_loaded"] = False
            logging.error("Failed to load collaborative model: %s", str(e))

        return True

    except Exception as e:
        logging.error("LD000-3: %s", traceback.format_exc())
        update_status("error", model_loaded=False, error_message=str(e))
        return False


def force_retrain():
    """Force retraining the model từ database với dữ liệu mới nhất"""
    global cleaned_df, indices, similarity_matrix

    logging.info("Force retraining model from database")
    update_status("retraining")

    try:
        # Xóa model hiện tại nếu tồn tại
        if model_data_path.exists():
            model_data_path.unlink()
            logging.info("Removed existing model file")

        # Load và xử lý dữ liệu từ database
        df = load_data_from_db()

        # Tạo DataFrame đã được làm sạch
        cleaned_df = df.copy()

        # Xử lý giá trị null
        if 'uses' in cleaned_df.columns:
            cleaned_df['uses'] = cleaned_df['uses'].fillna('')

        # Tạo stop words tiếng Việt
        my_stop_words = ["là", "của", "và", "nhưng", "hay", "hoặc", "tôi", "bạn", "mình",
                         "họ", "nó", "rất", "quá", "lắm", "không", "có", "làm", "được",
                         "tốt", "xấu"]

        # Tạo TF-IDF vectorizer và tính toán ma trận
        tfv = TfidfVectorizer(min_df=3, strip_accents='unicode', analyzer='word',
                              token_pattern=r'\w{1,}', ngram_range=(1, 3),
                              stop_words=my_stop_words)
        tfv_matrix = tfv.fit_transform(cleaned_df['uses'])

        # Tính toán similarity matrix sử dụng sigmoid kernel
        similarity_matrix = sigmoid_kernel(tfv_matrix, tfv_matrix)

        # Tạo mapping từ tên sản phẩm sang chỉ số
        cleaned_df.reset_index(drop=True, inplace=True)
        indices = pd.Series(cleaned_df.index, index=cleaned_df['name']).drop_duplicates()

        # Lưu dữ liệu mô hình để sử dụng lại
        data = {'cleaned_df': cleaned_df, 'indices': indices, 'cosine_sim': similarity_matrix}
        with model_data_path.open('wb') as f:
            pickle.dump(data, f)

        update_status("ready", model_loaded=True,
                      data_info={"products": len(cleaned_df), "last_model_update": time.ctime()})
        return True

    except Exception as e:
        logging.error("RETRAIN-ERROR: %s", traceback.format_exc())
        update_status("error", model_loaded=False, error_message=str(e))
        return False


def give_rec(name, count=10):
    """Hàm gợi ý sản phẩm dựa trên tên sản phẩm"""
    global indices, cleaned_df, similarity_matrix
    try:
        idx = indices[name]
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:count + 1]
        rec_indices = [i[0] for i in sim_scores]
        # recommendations_df = cleaned_df.iloc[rec_indices][['id', 'name', 'category_name']]
        recommendations_df = cleaned_df.iloc[rec_indices][['id']]
        return recommendations_df.to_dict(orient='records')
    except Exception as e:
        logging.error("GR000-1: Error in recommendation: %s", str(e))
        return [{"error": "Error generating recommendations"}]


# Định nghĩa các API endpoint

@app.route('/api/status', methods=['GET'])
def status():
    status_detail = {
        **api_status,
        "model_dir": str(MODEL_DIR),
        "model_file_exists": model_data_path.exists()
    }
    return jsonify(status_detail)


@app.route('/api/reload', methods=['GET'])
def reload_data():
    success = load_data()
    return jsonify({"success": success, "status": api_status})


@app.route('/api/retrain', methods=['GET'])
def retrain():
    """API endpoint để force retrain model từ database"""
    success = force_retrain()
    return jsonify({
        "success": success,
        "status": api_status,
        "message": "Model retrained successfully" if success else "Failed to retrain model"
    })


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
        num_recommendations = request.args.get('count', default=10, type=int)
        product_row = cleaned_df[cleaned_df['id'] == product_id]
        if product_row.empty:
            return jsonify({"status": "error", "message": f"Product with ID {product_id} not found"}), 404
        product_name = product_row['name'].iloc[0]
        recommendations = give_rec(product_name, num_recommendations)
        # return jsonify({
        #     "status": "success",
        #     "product_id": product_id,
        #     "product_name": product_name,
        #     "recommendations": recommendations
        # })
        return jsonify(give_rec(product_name, num_recommendations))
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
        products = cleaned_df[['id', 'name', 'category_name']].head(limit).to_dict(orient='records')
        return jsonify({"status": "success", "count": len(products), "products": products})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/test-db-connection', methods=['GET'])
def test_db_connection():
    """Test database connection"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        result = cur.fetchone()
        cur.close()
        conn.close()
        return jsonify({
            "status": "success",
            "message": "Database connection successful",
            "result": result
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Database connection failed: {str(e)}"
        }), 500

@app.route('/api/record-action', methods=['POST'])
def record_action():
    try:
        data = request.json
        user_id = data.get('user_id')
        session_id = data.get('session_id')
        product_id = data.get('product_id')
        action_type = data.get('action_type')
        action_value = data.get('action_value', '1')

        if not session_id and not user_id:
            return jsonify({"status": "error", "message": "Either user_id or session_id is required"}), 400
        if not product_id or not action_type:
            return jsonify({"status": "error", "message": "product_id and action_type are required"}), 400

        conn = get_db_connection()
        success = record_user_action(conn, user_id, session_id, product_id, action_type, action_value)
        conn.close()

        return jsonify({
            "status": "success" if success else "error",
            "message": "Action recorded" if success else "Failed to record"
        })
    except Exception as e:
        logging.error(f"Error in record_action: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/collaborative/user/<user_id>', methods=['GET'])
def recommend_for_user(user_id):
    if not api_status["collaborative_model_loaded"]:
        return jsonify({"status": "error", "message": "Collaborative model not loaded"}), 503
    count = request.args.get('count', default=10, type=int)
    recs = user_based_recommendations(user_id, count)
    return jsonify({"status": "success", "recommendations": recs})


@app.route('/api/collaborative/product/<int:product_id>', methods=['GET'])
def recommend_for_product(product_id):
    if not api_status["collaborative_model_loaded"]:
        return jsonify({"status": "error", "message": "Collaborative model not loaded"}), 503
    count = request.args.get('count', default=10, type=int)
    recs = item_based_recommendations(product_id, count)
    return jsonify({"status": "success", "recommendations": recs})


@app.route('/api/hybrid', methods=['GET'])
def hybrid():
    user_id = request.args.get('user_id')
    session_id = request.args.get('session_id')
    product_id = request.args.get('product_id', type=int)
    count = request.args.get('count', default=10, type=int)

    if not (api_status["collaborative_model_loaded"] or api_status["model_loaded"]):
        return jsonify({"status": "error", "message": "No model loaded"}), 503

    user_identifier = user_id if user_id else f"session_{session_id}"
    recs = hybrid_recommendations(user_identifier, product_id, count)
    return jsonify({"status": "success", "recommendations": recs})


if __name__ == '__main__':
    if load_data():
        logging.info("Data loaded successfully. Starting API server...")
    else:
        logging.warning("Starting API server without data loaded. Status will be 'error' until reload.")
    app.run(host='0.0.0.0', port=5000, debug=False)

def retrain_collaborative_model_periodically():
    try:
        conn = get_db_connection()
        from collaborative_filtering import build_collaborative_model
        build_collaborative_model(conn, force=True)
        conn.close()
        logging.info("Scheduled retraining of collaborative model completed.")
    except Exception as e:
        logging.error(f"Scheduled retraining failed: {str(e)}")

# Bắt đầu scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(retrain_collaborative_model_periodically, 'interval', minutes=1)  # chạy mỗi 10 phút
scheduler.start()


# Load data khi khởi động ứng dụng
with app.app_context():
    load_data()


