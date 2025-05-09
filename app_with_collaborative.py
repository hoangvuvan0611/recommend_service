import logging
import traceback
import time
from pathlib import Path
import pickle

import pandas as pd
import psycopg2
from psycopg2 import sql
from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
from flask_cors import CORS

# Import module collaborative filtering đã tạo
from collaborative_filtering import (
    build_collaborative_model,
    item_based_recommendations,
    user_based_recommendations,
    hybrid_recommendations,
    record_user_action
)

# Cấu hình Flask và CORS
app = Flask(__name__)
CORS(app)

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Trạng thái API toàn cục
api_status = {
    "status": "initializing",  # trạng thái: initializing, loading, ready, error
    "content_model_loaded": False,
    "collaborative_model_loaded": False,
    "last_updated": None,
    "error_message": None,
    "data_info": None
}

# Cấu hình thư mục mô hình
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Đường dẫn file mô hình content-based
model_data_path = MODEL_DIR / 'model_data.pkl'
# Đường dẫn file mô hình collaborative
collab_model_path = MODEL_DIR / 'collaborative_model.pkl'

# Cấu hình database
DB_CONFIG = {
    "host": "localhost",
    "database": "AGRI_MARKET",
    "user": "postgres",
    "password": "Hoanglam06112003@",
    "port": 5432
}

# Biến lưu dữ liệu toàn cục cho content-based
cleaned_df = None
indices = None
similarity_matrix = None


def update_status(status, content_model_loaded=None, collaborative_model_loaded=None, error_message=None,
                  data_info=None):
    """Cập nhật trạng thái API và log thông tin"""
    global api_status
    api_status["status"] = status
    api_status["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

    if content_model_loaded is not None:
        api_status["content_model_loaded"] = content_model_loaded

    if collaborative_model_loaded is not None:
        api_status["collaborative_model_loaded"] = collaborative_model_loaded

    if error_message is not None:
        api_status["error_message"] = error_message

    if data_info is not None:
        api_status["data_info"] = data_info

    logging.info(
        f"API Status: {status}, Content model: {api_status['content_model_loaded']}, Collaborative model: {api_status['collaborative_model_loaded']}")


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


def load_content_based_model():
    """Load và xử lý dữ liệu, tạo mô hình content-based và lưu lại nếu cần"""
    global cleaned_df, indices, similarity_matrix
    update_status("LD0001: loading content model")

    # Thử load mô hình đã được lưu
    if model_data_path.exists():
        try:
            logging.info("Loading pre-trained content-based model data...")
            with model_data_path.open('rb') as f:
                data = pickle.load(f)
                cleaned_df = data.get('cleaned_df')
                indices = data.get('indices')
                similarity_matrix = data.get('cosine_sim')
            update_status("content model loaded", content_model_loaded=True,
                          data_info={"products": len(cleaned_df),
                                     "last_model_update": time.ctime(model_data_path.stat().st_mtime)})
            return True
        except Exception as e:
            logging.error(f"Error loading pre-trained data: {e}")
            update_status("error", content_model_loaded=False, error_message=str(e))

    # Xử lý dữ liệu từ database nếu không load được mô hình đã lưu
    try:
        logging.info("Processing data from PostgreSQL database")
        try:
            df = load_data_from_db()
            logging.info("Data loaded successfully from database")
        except Exception as db_error:
            error_message = f"Error loading data from database: {db_error}"
            logging.error("LD000-2: " + error_message)
            update_status("error", content_model_loaded=False, error_message=error_message)
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
        logging.info("Content-based model data saved successfully.")

        update_status("content model loaded", content_model_loaded=True,
                      data_info={"products": len(cleaned_df), "last_model_update": time.ctime()})
        return True

    except Exception as e:
        logging.error("LD000-3: %s", traceback.format_exc())
        update_status("error", content_model_loaded=False, error_message=str(e))
        return False


def content_based_recommend(name, count=10):
    """Hàm gợi ý sản phẩm dựa trên tên sản phẩm (Content-based)"""
    global indices, cleaned_df, similarity_matrix
    try:
        idx = indices[name]
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:count + 1]
        rec_indices = [i[0] for i in sim_scores]
        recommendations_df = cleaned_df.iloc[rec_indices][['id']]
        return recommendations_df.to_dict(orient='records')
    except Exception as e:
        logging.error("GR000-1: Error in content-based recommendation: %s", str(e))
        return [{"error": "Error generating recommendations"}]


def load_models():
    """Load cả hai loại mô hình"""
    try:
        # Load content-based model
        content_success = load_content_based_model()

        # Tạo kết nối database cho collaborative model
        conn = get_db_connection()

        # Load collaborative model
        update_status("loading collaborative model")
        collab_success = build_collaborative_model(conn)
        conn.close()

        # Cập nhật trạng thái
        if content_success and collab_success:
            update_status("ready", content_model_loaded=True, collaborative_model_loaded=True)
            return True
        elif content_success:
            update_status("partial", content_model_loaded=True, collaborative_model_loaded=False,
                          error_message="Collaborative model failed to load")
            return True
        elif collab_success:
            update_status("partial", content_model_loaded=False, collaborative_model_loaded=True,
                          error_message="Content-based model failed to load")
            return True
        else:
            update_status("error", content_model_loaded=False, collaborative_model_loaded=False,
                          error_message="Both models failed to load")
            return False
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        update_status("error", error_message=str(e))
        return False


# API Endpoints

@app.route('/api/status', methods=['GET'])
def status():
    status_detail = {
        **api_status,
        "model_dir": str(MODEL_DIR),
        "content_model_exists": model_data_path.exists(),
        "collaborative_model_exists": collab_model_path.exists()
    }
    return jsonify(status_detail)


@app.route('/api/reload', methods=['GET'])
def reload_data():
    success = load_models()
    return jsonify({"success": success, "status": api_status})


@app.route('/api/retrain-content', methods=['GET'])
def retrain_content():
    """API endpoint để force retrain content-based model từ database"""
    if model_data_path.exists():
        model_data_path.unlink()
        logging.info("Removed existing content-based model file")

    success = load_content_based_model()
    return jsonify({
        "success": success,
        "status": api_status,
        "message": "Content-based model retrained successfully" if success else "Failed to retrain content-based model"
    })


@app.route('/api/retrain-collaborative', methods=['GET'])
def retrain_collaborative():
    """API endpoint để force retrain collaborative model từ database"""
    conn = get_db_connection()
    try:
        success = build_collaborative_model(conn, force=True)
        return jsonify({
            "success": success,
            "status": api_status,
            "message": "Collaborative model retrained successfully" if success else "Failed to retrain collaborative model"
        })
    finally:
        conn.close()


@app.route('/api/recommend/<product_name>', methods=['GET'])
def recommend(product_name):
    """API endpoint cho content-based recommendation"""
    if api_status["content_model_loaded"] != True:
        return jsonify({
            "status": "error",
            "message": f"Content-based model not ready. Current status: {api_status['status']}",
            "details": api_status
        }), 503

    try:
        num_recommendations = request.args.get('count', default=10, type=int)
        recommendations = content_based_recommend(product_name, num_recommendations)
        if isinstance(recommendations, dict) and "error" in recommendations:
            return jsonify({"status": "error", "message": recommendations["error"]}), 500
        return jsonify({
            "status": "success",
            "product_name": product_name,
            "recommendations": recommendations,
            "type": "content-based"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/recommend-by-id/<int:product_id>', methods=['GET'])
def recommend_by_id(product_id):
    """API endpoint cho content-based recommendation dựa trên ID sản phẩm"""
    if api_status["content_model_loaded"] != True:
        return jsonify({
            "status": "error",
            "message": f"Content-based model not ready. Current status: {api_status['status']}",
            "details": api_status
        }), 503

    try:
        num_recommendations = request.args.get('count', default=10, type=int)
        product_row = cleaned_df[cleaned_df['id'] == product_id]
        if product_row.empty:
            return jsonify({"status": "error", "message": f"Product with ID {product_id} not found"}), 404
        product_name = product_row['name'].iloc[0]
        recommendations = content_based_recommend(product_name, num_recommendations)
        return jsonify({
            "status": "success",
            "product_id": product_id,
            "product_name": product_name,
            "recommendations": recommendations,
            "type": "content-based"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/collaborative/recommend-by-product/<int:product_id>', methods=['GET'])
def collaborative_recommend_by_product(product_id):
    """API endpoint cho item-based collaborative recommendation"""
    if api_status["collaborative_model_loaded"] != True:
        return jsonify({
            "status": "error",
            "message": f"Collaborative model not ready. Current status: {api_status['status']}",
            "details": api_status
        }), 503

    try:
        num_recommendations = request.args.get('count', default=10, type=int)
        recommendations = item_based_recommendations(product_id, num_recommendations)

        if not recommendations:
            # Nếu không có đề xuất collaborative, sử dụng content-based
            if api_status["content_model_loaded"]:
                product_row = cleaned_df[cleaned_df['id'] == product_id]
                if not product_row.empty:
                    product_name = product_row['name'].iloc[0]
                    recommendations = content_based_recommend(product_name, num_recommendations)
                    return jsonify({
                        "status": "success",
                        "product_id": product_id,
                        "recommendations": recommendations,
                        "type": "content-based-fallback",
                        "message": "No collaborative data available, using content-based recommendations"
                    })

            return jsonify({
                "status": "warning",
                "message": f"No recommendations available for product ID {product_id}",
                "recommendations": []
            })

        return jsonify({
            "status": "success",
            "product_id": product_id,
            "recommendations": recommendations,
            "type": "item-based-collaborative"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/collaborative/recommend-for-user/<user_id>', methods=['GET'])
def collaborative_recommend_for_user(user_id):
    """API endpoint cho user-based collaborative recommendation"""
    if api_status["collaborative_model_loaded"] != True:
        return jsonify({
            "status": "error",
            "message": f"Collaborative model not ready. Current status: {api_status['status']}",
            "details": api_status
        }), 503

    try:
        num_recommendations = request.args.get('count', default=10, type=int)
        recommendations = user_based_recommendations(user_id, num_recommendations)

        if not recommendations:
            return jsonify({
                "status": "warning",
                "message": f"No user-based recommendations available for user ID {user_id}",
                "recommendations": []
            })

        return jsonify({
            "status": "success",
            "user_id": user_id,
            "recommendations": recommendations,
            "type": "user-based-collaborative"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/hybrid/recommend', methods=['GET'])
def hybrid_recommend():
    """API endpoint cho hybrid recommendation (kết hợp content-based và collaborative)"""
    if api_status["collaborative_model_loaded"] != True and api_status["content_model_loaded"] != True:
        return jsonify({
            "status": "error",
            "message": f"Neither collaborative nor content-based model is ready. Current status: {api_status['status']}",
            "details": api_status
        }), 503

    try:
        user_id = request.args.get('user_id')
        session_id = request.args.get('session_id')
        product_id = request.args.get('product_id', type=int)
        num_recommendations = request.args.get('count', default=10, type=int)

        # Nếu có user_id thì sử dụng, nếu không thì sử dụng session_id
        user_identifier = user_id if user_id and user_id != "null" else f"session_{session_id}"

        recommendations = []

        if product_id:
            # Hybrid approach: kết hợp item-based collaborative và user-based collaborative
            if api_status["collaborative_model_loaded"]:
                recommendations = hybrid_recommendations(user_identifier, product_id, num_recommendations)

            # Nếu không có đề xuất collaborative hoặc collaborative model không sẵn sàng
            if not recommendations and api_status["content_model_loaded"]:
                product_row = cleaned_df[cleaned_df['id'] == product_id]
                if not product_row.empty:
                    product_name = product_row['name'].iloc[0]
                    recommendations = content_based_recommend(product_name, num_recommendations)
                    return jsonify({
                        "status": "success",
                        "product_id": product_id,
                        "user_id": user_identifier,
                        "recommendations": recommendations,
                        "type": "content-based-fallback"
                    })
        elif user_identifier:
            # Chỉ user-based collaborative nếu không có product_id
            if api_status["collaborative_model_loaded"]:
                recommendations = user_based_recommendations(user_identifier, num_recommendations)

        if not recommendations:
            return jsonify({
                "status": "warning",
                "message": "No recommendations available with the provided parameters",
                "recommendations": []
            })

        return jsonify({
            "status": "success",
            "user_id": user_identifier,
            "product_id": product_id,
            "recommendations": recommendations,
            "type": "hybrid"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/record-action', methods=['POST'])
def record_action():
    """API endpoint để ghi lại hành động của người dùng"""
    try:
        data = request.json
        user_id = data.get('user_id')
        session_id = data.get('session_id')
        product_id = data.get('product_id')
        action_type = data.get('action_type')
        action_value = data.get('action_value', '1')

        if not session_id and not user_id:
            return jsonify({
                "status": "error",
                "message": "Either user_id or session_id must be provided"
            }), 400

        if not product_id:
            return jsonify({
                "status": "error",
                "message": "product_id is required"
            }), 400

        if not action_type:
            return jsonify({
                "status": "error",
                "message": "action_type is required"
            }), 400

        # Xác thực action_type
        valid_actions = ['view', 'click', 'add-to-cart', 'purchase', 'rating', 'favorite']
        if action_type not in valid_actions:
            return jsonify({
                "status": "error",
                "message": f"Invalid action_type. Valid actions are: {', '.join(valid_actions)}"
            }), 400

        # Kết nối database và ghi lại hành động
        conn = get_db_connection()
        try:
            success = record_user_action(conn, user_id, session_id, product_id, action_type, action_value)
            conn.commit()

            if success:
                return jsonify({
                    "status": "success",
                    "message": "User action recorded successfully"
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Failed to record user action"
                }), 500
        finally:
            conn.close()
    except Exception as e:
        logging.error(f"Error recording user action: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/products', methods=['GET'])
def get_products():
    if api_status["content_model_loaded"] != True:
        return jsonify({
            "status": "error",
            "message": f"Content-based model not ready. Current status: {api_status['status']}",
            "details": api_status
        }), 503

    try:
        limit = request.args.get('limit', default=100, type=int)
        products = cleaned_df[['id', 'name', 'category_name']].head(limit).to_dict(orient='records')
        return jsonify({"status": "success", "count": len(products), "products": products})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    if load_models():
        logging.info("Models loaded successfully. Starting API server...")
    else:
        logging.warning(
            "Starting API server without all models loaded. Status will be 'error' or 'partial' until reload.")
    app.run(host='0.0.0.0', port=5000, debug=False)

# Load models khi khởi động ứng dụng
with app.app_context():
    load_models()