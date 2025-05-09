import logging
import traceback
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import time
import random

class SnowflakeIdGenerator:
    def generate_id(self):
        return int(time.time() * 1000) + random.randint(0, 999)

snowflakeIdGenerator = SnowflakeIdGenerator()

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Thư mục lưu trữ mô hình
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Đường dẫn file mô hình collaborative filtering
collab_model_path = MODEL_DIR / 'collaborative_model.pkl'

# Biến lưu trữ dữ liệu toàn cục
user_item_matrix = None
item_similarity_matrix = None
user_similarity_matrix = None
product_mapping = None
user_mapping = None
reverse_product_mapping = None
reverse_user_mapping = None


def load_collaborative_data(conn):
    """Load dữ liệu từ bảng collaboratives để xây dựng collaborative filtering model"""
    try:
        logging.info("Loading collaborative data from database")

        # Query để lấy dữ liệu từ bảng collaboratives
        query = """
            SELECT user_id, session_id , product_id, action_type, action_value, created_at
            FROM collaboratives
            ORDER BY created_at DESC;
        """

        # Đọc dữ liệu vào DataFrame
        df = pd.read_sql(query, conn)

        logging.info(f"Loaded {len(df)} collaborative records from database")

        # Xử lý các loại hành động khác nhau (view, add-to-cart, purchase)
        # Gán trọng số cho từng loại hành động
        action_weights = {
            'view': 1.0,
            'click': 1.5,
            'add-to-cart': 3.0,
            'purchase': 5.0,
            'rating': 2.0,  # Sẽ nhân với giá trị rating
            'favorite': 2.5
        }

        # Tạo cột weight dựa trên action_type
        def calculate_weight(row):
            if row['action_type'] == 'rating':
                try:
                    return action_weights['rating'] * float(row['action_value'])
                except (ValueError, TypeError):
                    return action_weights['rating']
            else:
                return action_weights.get(row['action_type'], 1.0)

        df['weight'] = df.apply(calculate_weight, axis=1)

        return df
    except Exception as e:
        logging.error(f"Error loading collaborative data: {e}")
        raise


def preprocess_data(df):
    """Tiền xử lý dữ liệu collaborative"""
    try:
        # Xử lý giá trị NULL trong user_id nếu có (sử dụng sessionId thay thế)
        # Tạo định danh duy nhất cho cả user_id và sessionId
        # Tạo định danh người dùng duy nhất (ưu tiên user_id nếu có, nếu không thì dùng session_id)
        df['user_identifier'] = df.apply(
            lambda row: f"user_{int(row['user_id'])}" if pd.notnull(row['user_id']) else f"session_{row['session_id']}",
            axis=1
        )

        # Tạo ánh xạ user_identifier -> user_index và product_id -> product_index
        unique_users = df['user_identifier'].unique()
        unique_products = df['product_id'].unique()

        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        product_to_idx = {product: idx for idx, product in enumerate(unique_products)}

        idx_to_user = {idx: user for user, idx in user_to_idx.items()}
        idx_to_product = {idx: product for product, idx in product_to_idx.items()}

        # Tạo ma trận user-item
        user_item_matrix = np.zeros((len(unique_users), len(unique_products)))

        # Điền dữ liệu vào ma trận
        for _, row in df.iterrows():
            user_idx = user_to_idx[row['user_identifier']]
            product_idx = product_to_idx[row['product_id']]

            # Lấy trọng số cao nhất nếu có nhiều tương tác
            current_weight = user_item_matrix[user_idx, product_idx]
            new_weight = row['weight']

            if new_weight > current_weight:
                user_item_matrix[user_idx, product_idx] = new_weight

        return user_item_matrix, user_to_idx, product_to_idx, idx_to_user, idx_to_product
    except Exception as e:
        logging.error(f"Error preprocessing collaborative data: {e}")
        logging.error(traceback.format_exc())
        raise


def build_collaborative_model(conn, force=False):
    """Xây dựng mô hình collaborative filtering"""
    global user_item_matrix, item_similarity_matrix, user_similarity_matrix
    global product_mapping, user_mapping, reverse_product_mapping, reverse_user_mapping

    try:
        # Kiểm tra nếu mô hình đã tồn tại và không yêu cầu force retrain
        if collab_model_path.exists() and not force:
            logging.info("Loading pre-trained collaborative model...")
            with collab_model_path.open('rb') as f:
                model_data = pickle.load(f)
                user_item_matrix = model_data['user_item_matrix']
                item_similarity_matrix = model_data['item_similarity']
                user_similarity_matrix = model_data['user_similarity']
                product_mapping = model_data['product_mapping']
                user_mapping = model_data['user_mapping']
                reverse_product_mapping = model_data['reverse_product_mapping']
                reverse_user_mapping = model_data['reverse_user_mapping']

            logging.info("Collaborative model loaded successfully.")
            return True

        # Load dữ liệu và tiền xử lý
        df = load_collaborative_data(conn)

        if len(df) < 10:  # Quá ít dữ liệu để xây dựng mô hình
            logging.warning("Not enough collaborative data to build a model (less than 10 records)")
            return False

        # Tiền xử lý dữ liệu
        user_item_matrix, user_mapping, product_mapping, reverse_user_mapping, reverse_product_mapping = preprocess_data(
            df)

        # Tính toán ma trận tương đồng
        logging.info("Computing item-item similarity matrix")
        item_similarity_matrix = cosine_similarity(user_item_matrix.T)

        logging.info("Computing user-user similarity matrix")
        user_similarity_matrix = cosine_similarity(user_item_matrix)

        # Lưu mô hình
        model_data = {
            'user_item_matrix': user_item_matrix,
            'item_similarity': item_similarity_matrix,
            'user_similarity': user_similarity_matrix,
            'product_mapping': product_mapping,
            'user_mapping': user_mapping,
            'reverse_product_mapping': reverse_product_mapping,
            'reverse_user_mapping': reverse_user_mapping,
            'last_updated': time.ctime()
        }

        with collab_model_path.open('wb') as f:
            pickle.dump(model_data, f)

        logging.info("Collaborative model built and saved successfully.")
        return True
    except Exception as e:
        logging.error(f"Error building collaborative model: {e}")
        logging.error(traceback.format_exc())
        return False


def item_based_recommendations(item_id, num_recommendations=10):
    """Đề xuất sản phẩm dựa trên sản phẩm đã xem (Item-based CF)"""
    global item_similarity_matrix, product_mapping, reverse_product_mapping

    try:
        if item_similarity_matrix is None or product_mapping is None:
            logging.error("Collaborative model not loaded.")
            return []

        # Kiểm tra xem item_id có trong dữ liệu không
        if item_id not in product_mapping:
            logging.warning(f"Product ID {item_id} not found in collaborative data")
            return []

        # Lấy index của sản phẩm
        item_idx = product_mapping[item_id]

        # Lấy điểm tương đồng của sản phẩm với tất cả sản phẩm khác
        item_scores = list(enumerate(item_similarity_matrix[item_idx]))

        # Sắp xếp theo điểm tương đồng giảm dần và lấy top N sản phẩm
        item_scores = sorted(item_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]

        # Lấy ID của các sản phẩm được đề xuất
        recommended_items = []
        for idx, score in item_scores:
            product_id = reverse_product_mapping[idx]
            recommended_items.append({
                'id': int(product_id),
                'score': float(score)
            })

        return recommended_items
    except Exception as e:
        logging.error(f"Error in item-based recommendations: {e}")
        logging.error(traceback.format_exc())
        return []


def user_based_recommendations(user_identifier, num_recommendations=10):
    """Đề xuất sản phẩm dựa trên người dùng tương tự (User-based CF)"""
    global user_item_matrix, user_similarity_matrix, user_mapping, reverse_product_mapping

    try:
        if user_similarity_matrix is None or user_mapping is None:
            logging.error("Collaborative model not loaded.")
            return []

        # Kiểm tra xem user_identifier có trong dữ liệu không
        user_id_str = str(user_identifier)
        if user_id_str not in user_mapping:
            logging.warning(f"User {user_identifier} not found in collaborative data")
            return []

        # Lấy index của người dùng
        user_idx = user_mapping[user_id_str]

        # Lấy điểm tương đồng của người dùng với tất cả người dùng khác
        user_scores = list(enumerate(user_similarity_matrix[user_idx]))

        # Sắp xếp theo điểm tương đồng giảm dần
        user_scores = sorted(user_scores, key=lambda x: x[1], reverse=True)[1:11]  # Top 10 similar users

        # Lấy các sản phẩm mà người dùng tương tự đã thích mà người dùng hiện tại chưa tương tác
        user_items = set(np.where(user_item_matrix[user_idx] > 0)[0])
        recommendations = {}

        for similar_user_idx, similarity_score in user_scores:
            # Lấy các sản phẩm mà người dùng tương tự đã tương tác
            similar_user_items = np.where(user_item_matrix[similar_user_idx] > 0)[0]

            # Tính điểm cho mỗi sản phẩm
            for item_idx in similar_user_items:
                if item_idx not in user_items:  # Chỉ đề xuất sản phẩm mà người dùng chưa tương tác
                    if item_idx not in recommendations:
                        recommendations[item_idx] = 0

                    # Điểm đề xuất = độ tương đồng người dùng * trọng số tương tác
                    recommendations[item_idx] += similarity_score * user_item_matrix[similar_user_idx, item_idx]

        # Sắp xếp các đề xuất theo điểm giảm dần
        recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]

        # Lấy ID của các sản phẩm được đề xuất
        recommended_items = []
        for item_idx, score in recommendations:
            product_id = reverse_product_mapping[item_idx]
            recommended_items.append({
                'id': int(product_id),
                'score': float(score)
            })

        return recommended_items
    except Exception as e:
        logging.error(f"Error in user-based recommendations: {e}")
        logging.error(traceback.format_exc())
        return []


def hybrid_recommendations(user_identifier, item_id=None, num_recommendations=10):
    """Kết hợp item-based và user-based để đưa ra đề xuất (Hybrid approach)"""
    try:
        if item_id is not None:
            # Lấy đề xuất từ item-based
            item_recs = item_based_recommendations(item_id, num_recommendations)

            # Nếu user_identifier có tồn tại, kết hợp với user-based
            user_id_str = str(user_identifier)
            if user_mapping and user_id_str in user_mapping:
                user_recs = user_based_recommendations(user_identifier, num_recommendations)

                # Kết hợp kết quả, ưu tiên các sản phẩm xuất hiện ở cả hai phương pháp
                hybrid_scores = {}

                # Trọng số cho mỗi phương pháp
                item_weight = 0.7  # Trọng số cho item-based (thường ưu tiên hơn)
                user_weight = 0.3  # Trọng số cho user-based

                # Thêm item-based recommendations
                for rec in item_recs:
                    hybrid_scores[rec['id']] = rec['score'] * item_weight

                # Thêm/Cập nhật user-based recommendations
                for rec in user_recs:
                    if rec['id'] in hybrid_scores:
                        hybrid_scores[rec['id']] += rec['score'] * user_weight
                    else:
                        hybrid_scores[rec['id']] = rec['score'] * user_weight

                # Sắp xếp và lấy top N
                sorted_recs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]

                return [{'id': item_id, 'score': float(score)} for item_id, score in sorted_recs]

            return item_recs
        elif user_identifier:
            # Nếu chỉ có user_identifier, sử dụng user-based
            return user_based_recommendations(user_identifier, num_recommendations)
        else:
            logging.warning("Neither user_identifier nor item_id provided for recommendations")
            return []
    except Exception as e:
        logging.error(f"Error in hybrid recommendations: {e}")
        logging.error(traceback.format_exc())
        return []


def record_user_action(conn, user_id, session_id, product_id, action_type, action_value="1"):
    try:
        cursor = conn.cursor()

        # Dùng user_id nếu có, nếu không dùng session_id
        user_filter = user_id if user_id not in [None, "null", "undefined"] else None
        session_filter = session_id if user_filter is None else None

        # Kiểm tra bản ghi đã tồn tại
        check_query = """
            SELECT id FROM collaboratives 
            WHERE product_id = %s AND action_type = %s AND
                  (user_id = %s OR session_id = %s)
            ORDER BY created_at DESC LIMIT 1;
        """
        cursor.execute(check_query, (product_id, action_type, user_filter, session_filter))
        existing_record = cursor.fetchone()

        if existing_record:
            update_query = """
                UPDATE collaboratives
                SET action_value = %s, created_at = CURRENT_TIMESTAMP
                WHERE id = %s;
            """
            cursor.execute(update_query, (action_value, existing_record[0]))
        else:
            insert_query = """
                INSERT INTO collaboratives (id, user_id, session_id, product_id, action_type, action_value)
                VALUES (%s, %s, %s, %s, %s, %s);
            """
            generated_id = snowflakeIdGenerator.generate_id()
            cursor.execute(insert_query,
                           (generated_id, user_filter, session_filter, product_id, action_type, action_value))

        conn.commit()
        cursor.close()
        return True
    except Exception as e:
        logging.error(f"Error recording user action: {e}")
        conn.rollback()
        return False