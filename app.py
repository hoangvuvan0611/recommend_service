import traceback

import pandas as pd
import pickle
import os
import time
from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Cho phép truy cập từ các origin khác nhau

# Biến lưu trạng thái API
api_status = {
    "status": "initializing",  # Các trạng thái: initializing, loading, ready, error
    "model_loaded": False,
    "last_updated": None,
    "error_message": None,
    "data_info": None
}

# Thiết lập đường dẫn cố định đến thư mục dữ liệu
DATA_DIR = r"D:\data_train"  # Đường dẫn đến thư mục chứa file CSV
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# Đảm bảo thư mục mô hình tồn tại
os.makedirs(MODEL_DIR, exist_ok=True)

# Đường dẫn đến các file dữ liệu
df_product_path = os.path.join(DATA_DIR, 'Product.csv')
df_description_path = os.path.join(DATA_DIR, 'Description.csv')
df_category_path = os.path.join(DATA_DIR, 'Category.csv')

# Đường dẫn đến các file mô hình
tfidf_matrix_path = os.path.join(MODEL_DIR, 'tfidf_matrix.pkl')
cosine_sim_path = os.path.join(MODEL_DIR, 'cosine_sim.pkl')
model_data_path = os.path.join(MODEL_DIR, 'model_data.pkl')


def update_status(status, model_loaded=None, error_message=None, data_info=None):
    """Cập nhật trạng thái API"""
    global api_status
    api_status["status"] = status
    api_status["last_updated"] = time.time()

    if model_loaded is not None:
        api_status["model_loaded"] = model_loaded

    if error_message is not None:
        api_status["error_message"] = error_message

    if data_info is not None:
        api_status["data_info"] = data_info

    print(f"API Status: {status}, Model loaded: {api_status['model_loaded']}")


def check_data_files():
    """Kiểm tra các file dữ liệu có tồn tại không"""
    files_exist = {
        "product": os.path.exists(df_product_path),
        "description": os.path.exists(df_description_path),
        "category": os.path.exists(df_category_path)
    }

    return files_exist, all(files_exist.values())


def load_data():
    """Load và xử lý dữ liệu, lưu kết quả vào biến toàn cục"""
    global cleaned_df, indices, cosine_sim

    update_status("LD0001: loading")

    # Kiểm tra các file dữ liệu
    files_status, all_files_exist = check_data_files()
    if not all_files_exist:
        error_message = f"Missing data files: {files_status}"
        print("LD0002: " + error_message)
        update_status("error", model_loaded=False, error_message=error_message)
        return False

    try:
        # Kiểm tra xem có thể load mô hình đã lưu trước đó không
        if os.path.exists(model_data_path):
            print("Loading pre-trained model data...")
            with open(model_data_path, 'rb') as f:
                data = pickle.load(f)
                cleaned_df = data['cleaned_df']
                indices = data['indices']
                cosine_sim = data['cosine_sim']

            print("LD0003: check model data")
            update_status("ready", model_loaded=True,
                          data_info={"products": len(cleaned_df),
                                     "last_model_update": time.ctime(os.path.getmtime(model_data_path))})
            return True
    except Exception as e:
        print(f"Error loading pre-trained data: {e}")
        error_message = f"Error loading model data: {str(e)}"
        update_status("error", model_loaded=False, error_message=error_message)

    # Nếu không thể load, thực hiện xử lý dữ liệu từ đầu
    try:
        print("Processing data from CSV files at", DATA_DIR)
        # Load dữ liệu từ CSV
        try:
            df_product = pd.read_csv(df_product_path)
            df_description = pd.read_csv(df_description_path)
            df_category = pd.read_csv(df_category_path)
            print("LD000-1: Data loaded successfully from", DATA_DIR)
        except Exception as file_error:
            error_message = f"Error loading data from {DATA_DIR}: {str(file_error)}"
            print("LD000-2:" + error_message)
            update_status("error", model_loaded=False, error_message=error_message)
            return False

        # In thông tin debug
        print("df_product columns:", df_product.columns.tolist())
        print("df_description columns:", df_description.columns.tolist())
        print("df_category columns:", df_category.columns.tolist())

        # Merge thông tin bảng product và description
        try:
            merged_df = pd.merge(df_product, df_description, on='id', how='inner')
            print(f"Merged product & description: {merged_df.shape[0]} rows")

            # Làm sạch thông tin category
            category_cleaned_df = df_category.copy()
            if 'image' in category_cleaned_df.columns:
                category_cleaned_df = category_cleaned_df.drop(columns=['image', 'status', 'created_at'],
                                                               errors='ignore')
            category_cleaned_df.rename(columns={'id': 'category'}, inplace=True)

            # Merge với category
            merged_df2 = pd.merge(category_cleaned_df, merged_df, on='category', how='inner')
            print(f"Merged with category: {merged_df2.shape[0]} rows")

            # Lấy các cột cần thiết và đổi tên
            columns_to_drop = ['slug_x', 'id', 'slug_y', 'category', 'unit', 'original_price',
                               'sale_price', 'expiry_period', 'status', 'created_at',
                               'modified_at', 'image']

            # Chỉ drop những cột tồn tại
            existing_columns = [col for col in columns_to_drop if col in merged_df2.columns]
            cleaned_df = merged_df2.drop(columns=existing_columns, errors='ignore')

            # Đổi tên cột nếu chúng tồn tại
            cleaned_df.rename(columns={'name_x': 'category', 'name_y': 'name'}, inplace=True)

            # Đảm bảo dữ liệu không có giá trị null
            cleaned_df['uses'] = cleaned_df['uses'].fillna('')
            print(f"Cleaned dataframe: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns")
            print("Columns after cleaning:", cleaned_df.columns.tolist())

            # Tạo các stop words tiếng Việt để lọc
            my_stop_words = ["là", "của", "và", "nhưng", "hay", "hoặc", "tôi", "bạn", "mình",
                             "họ", "nó", "rất", "quá", "lắm", "không", "có", "làm", "được",
                             "tốt", "xấu"]

            # Tạo TF-IDF Vectorizer với tham số phù hợp cho tiếng Việt
            tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                                  analyzer='word', token_pattern=r'\w{1,}',
                                  ngram_range=(1, 3), stop_words=my_stop_words)

            # Tính toán ma trận TF-IDF dựa trên cột 'uses'
            print("Calculating TF-IDF matrix...")
            tfv_matrix = tfv.fit_transform(cleaned_df['uses'])
            print(f"TF-IDF matrix shape: {tfv_matrix.shape}")

            # Tính toán Sigmoid kernel thay vì Cosine similarity
            print("Calculating similarity matrix...")
            from sklearn.metrics.pairwise import sigmoid_kernel
            cosine_sim = sigmoid_kernel(tfv_matrix, tfv_matrix)
            print(f"Similarity matrix shape: {cosine_sim.shape}")

            # Tạo mapping từ tên sản phẩm đến chỉ mục
            cleaned_df = cleaned_df.reset_index(drop=True)  # Reset index để đảm bảo chỉ mục đúng
            indices = pd.Series(cleaned_df.index, index=cleaned_df['name']).drop_duplicates()
            print(f"Created indices mapping for {len(indices)} products")

            # Lưu mô hình để sử dụng lại
            data = {
                'cleaned_df': cleaned_df,
                'indices': indices,
                'cosine_sim': cosine_sim
            }

            print("Saving model data...")
            with open(model_data_path, 'wb') as f:
                pickle.dump(data, f)

            print("LD0004: Complete training model")
            update_status("ready", model_loaded=True,
                          data_info={"products": len(cleaned_df), "last_model_update": time.ctime()})
            return True

        except Exception as merge_error:
            traceback_info = traceback.format_exc()
            error_message = f"Error in data processing: {str(merge_error)}\n{traceback_info}"
            print("LD000-3: " + error_message)
            update_status("error", model_loaded=False, error_message=error_message)
            return False

    except Exception as e:
        traceback_info = traceback.format_exc()
        error_message = f"Error processing data: {str(e)}\n{traceback_info}"
        print("LD000-3: " + error_message)
        update_status("error", model_loaded=False, error_message=error_message)
        return False


# Hàm gợi ý sản phẩm
def give_rec(name, count=10):
    global indices, cleaned_df  # cosine_sim cũng là biến toàn cục
    try:
        # Lấy chỉ số của sản phẩm từ tên
        idx = indices[name]
        # Tính toán điểm tương đồng
        sig_scores = list(enumerate(cosine_sim[idx]))
        # Sắp xếp theo điểm tương đồng giảm dần
        sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
        # Bỏ qua sản phẩm đầu tiên (chính nó) và lấy count sản phẩm tiếp theo
        sig_scores = sig_scores[1:count+1]
        # Lấy chỉ số của các sản phẩm gợi ý
        rec_indices = [i[0] for i in sig_scores]
        # Lấy thông tin id và tên của các sản phẩm được gợi ý
        recommendations_df = cleaned_df.iloc[rec_indices][['product', 'name']]
        # Chuyển dataframe sang list các dict
        recommendations = recommendations_df.to_dict(orient='records')
        return recommendations
    except Exception as e:
        print(f"GR000-1: Error in recommendation: {str(e)}")
        return [{"error": "Error generating recommendations"}]


# API gọi để kiểm tra server
@app.route('/api/status', methods=['GET'])
def status():
    # Thêm thông tin chi tiết về đường dẫn và file
    file_status, _ = check_data_files()
    status_detail = {
        **api_status,
        "file_paths": {
            "data_dir": DATA_DIR,
            "model_dir": MODEL_DIR,
        },
        "files": {
            "data_files": file_status,
            "model_file_exists": os.path.exists(model_data_path)
        }
    }
    return jsonify(status_detail)


# API khởi tạo lại dữ liệu (tải lại model)
@app.route('/api/reload', methods=['GET'])
def reload_data():
    success = load_data()
    return jsonify({
        "success": success,
        "status": api_status
    })


# API gợi ý sản phẩm với tên - CHUẨN API
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
            return jsonify({
                "status": "error",
                "message": recommendations["error"]
            }), 500

        return jsonify({
            "status": "success",
            "product_name": product_name,
            "recommendations": recommendations
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# API gọi đơn giản - phiên bản từ đoạn code thứ hai
@app.route('/recommend/<product_name>', methods=['GET'])
def simple_recommend(product_name):
    if api_status["status"] != "ready":
        return jsonify({
            "error": "API not ready",
            "status": api_status["status"]
        }), 503

    recommendations = give_rec(product_name)
    return jsonify(recommendations)


# API gợi ý sản phẩm với id
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
        # Tìm tên sản phẩm từ id
        product_row = cleaned_df[cleaned_df['id'] == product_id]
        if product_row.empty:
            return jsonify({
                "status": "error",
                "message": f"Product with ID {product_id} not found"
            }), 404

        product_name = product_row['name_y'].iloc[0]
        recommendations = give_rec(product_name, num_recommendations)

        return jsonify({
            "status": "success",
            "product_id": product_id,
            "product_name": product_name,
            "recommendations": recommendations
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# API lấy danh sách tất cả sản phẩm
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
        return jsonify({
            "status": "success",
            "count": len(products),
            "products": products
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# Khởi tạo dữ liệu khi khởi động module
update_status("initializing")

if __name__ == '__main__':
    # Load dữ liệu khi khởi động
    load_success = load_data()

    if load_success:
        print("Data loaded successfully. Starting API server...")
        # Chạy API trên cổng 5000 và cho phép kết nối từ tất cả interfaces
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("WARNING: Starting API server without data loaded. Status will be 'error' until reload.")
        app.run(host='0.0.0.0', port=5000, debug=False)