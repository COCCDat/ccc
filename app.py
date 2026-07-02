import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# KHỐI MASTER & THIẾT LẬP TRANG
# ==========================================
st.set_page_config(
    page_title="Dự báo rủi ro khách hàng",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Danh sách biến từ Notebook
FEATURES = ['TC1', 'TC2', 'TC3', 'TC4', 'TC5', 'NL1', 'NL2', 'NL3', 'NL4', 
            'DK1', 'DK2', 'DK3', 'DK4', 'DK5', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 
            'TS1', 'TS2', 'TS3', 'TS4']
TARGET = 'PD'

@st.cache_data
def load_data(file):
    """Hàm nạp dữ liệu dùng chung"""
    df = pd.read_csv(file)
    return df

# Khởi tạo trạng thái ban đầu cho session_state nếu chưa có
if 'is_trained' not in st.session_state:
    st.session_state['is_trained'] = False

# ==========================================
# THÀNH PHẦN 1: SIDEBAR (CẤU HÌNH)
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # 1. Tải dữ liệu mẫu
    uploaded_file = st.file_uploader("Tải lên file dữ liệu (.csv)", type=["csv"], help="Tải lên tệp 5c.csv của bạn")
    
    st.divider()
    
    # 2. Thiết lập tham số mô hình
    st.subheader("Tham số mô hình AI")
    st.caption("Thuật toán: Logistic Regression")
    
    with st.expander("Cấu hình siêu tham số", expanded=True):
        test_size = st.slider("Tỷ lệ tập kiểm thử (%)", min_value=10, max_value=50, value=20, step=5, help="Tương đương test_size trong train_test_split") / 100
        random_state = st.number_input("Random State (Chia tập)", min_value=0, max_value=999, value=23, help="Đảm bảo kết quả chia tập ổn định và có thể tái lập")
        c_param = st.number_input("Tham số điều chuẩn (C)", min_value=0.01, max_value=10.0, value=1.0, help="Nghịch đảo của độ mạnh chuẩn hóa")
        max_iter = st.number_input("Số vòng lặp tối đa (max_iter)", min_value=100, max_value=1000, value=100)

    # 3. Kích hoạt huấn luyện
    st.divider()
    if st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.error("Vui lòng tải lên file dữ liệu trước!")
        else:
            with st.spinner("Đang huấn luyện mô hình..."):
                df = load_data(uploaded_file)
                
                # Kiểm tra tính toàn vẹn của các cột dữ liệu
                missing_cols = [col for col in FEATURES + [TARGET] if col not in df.columns]
                if missing_cols:
                    st.error(f"Dữ liệu thiếu các cột cần thiết: {missing_cols}")
                else:
                    X = df[FEATURES]
                    y = df[TARGET]
                    
                    # Phân chia tập dữ liệu train/test
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
                    
                    # Huấn luyện mô hình
                    model = LogisticRegression(C=c_param, max_iter=max_iter)
                    model.fit(X_train, y_train)
                    
                    # Dự đoán trên tập kiểm thử nhằm đánh giá hiệu năng
                    y_pred = model.predict(X_test)
                    
                    # Lưu trữ an toàn các kết quả vào session state
                    st.session_state['model'] = model
                    st.session_state['accuracy'] = accuracy_score(y_test, y_pred)
                    st.session_state['cm'] = confusion_matrix(y_test, y_pred)
                    st.session_state['is_trained'] = True
                    st.success("Huấn luyện thành công! Hãy xem kết quả ở các Tab bên phải.")

# ==========================================
# THÀNH PHẦN 2: HEADER (ĐỊNH HƯỚNG)
# ==========================================
st.title("🛡️ Ứng Dụng Dự Báo Rủi Ro Khách Hàng")
st.caption("Mô hình AI dự báo xác suất khách hàng có rủi ro (PD) dựa trên các biến khảo sát/đánh giá (TC, NL, DK, V, TS).")

if uploaded_file is None:
    st.info("👈 Vui lòng tải lên tệp dữ liệu mẫu (vd: `5c.csv`) ở thanh điều hướng bên trái để bắt đầu.")
    st.stop()

df = load_data(uploaded_file)
st.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}** | 📊 Số dòng: **{df.shape[0]}**")
st.divider()

# ==========================================
# KHỞI TẠO KHỐI TABS CHỨC NĂNG
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["📋 Tổng quan dữ liệu", "📈 Trực quan hóa", "🎯 Kết quả Mô hình", "🔮 Sử dụng Mô hình"])

# ==========================================
# THÀNH PHẦN 3: TAB TỔNG QUAN DỮ LIỆU
# ==========================================
with tab1:
    st.subheader("Thông tin tập dữ liệu")
    col1, col2, col3 = st.columns(3)
    col1.metric("Số dòng (Khách hàng)", df.shape[0])
    col2.metric("Số cột (Đặc trưng)", df.shape[1])
    col3.metric("Số lượng biến đưa vào mô hình", len(FEATURES))
    
    st.write("**Dữ liệu thô (5 dòng đầu):**")
    with st.container(height=300):
        st.dataframe(df.head(), use_container_width=True)
        
    st.write("**Thống kê mô tả các biến của mô hình:**")
    st.dataframe(df[FEATURES + [TARGET]].describe(), use_container_width=True)

# ==========================================
# THÀNH PHẦN 4: TAB TRỰC QUAN HÓA DỮ LIỆU
# ==========================================
with tab2:
    st.subheader("Phân phối dữ liệu")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Biểu đồ cột của biến mục tiêu (PD)
        fig_target = px.histogram(df, x=TARGET, color=TARGET, 
                                  title="Phân phối Biến mục tiêu (PD)",
                                  labels={TARGET: "0: Không rủi ro | 1: Có rủi ro"},
                                  text_auto=True, color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig_target, use_container_width=True)
        
    with col2:
        # Cho phép lựa chọn động biến độc lập để phân tích tương quan ngắn
        var_to_plot = st.selectbox("Chọn biến để xem phân phối so với Rủi ro:", FEATURES)
        fig_var = px.histogram(df, x=var_to_plot, color=TARGET, barmode='group',
                               title=f"Phân phối {var_to_plot} theo Rủi ro",
                               color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig_var, use_container_width=True)

# ==========================================
# THÀNH PHẦN 5: TAB KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH
# ==========================================
with tab3:
    # ĐOẠN SỬA LỖI: Kiểm tra đồng thời cả cờ trạng thái lẫn sự tồn tại thực tế của mô hình
    if not st.session_state.get('is_trained') or 'model' not in st.session_state:
        st.info("Vui lòng bấm nút **'🚀 Huấn luyện mô hình'** ở thanh bên trái để xem kết quả huấn luyện.")
    else:
        st.subheader("Kiểm định Mô hình: Logistic Regression")
        
        acc = st.session_state['accuracy']
        cm = st.session_state['cm']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(label="Độ chính xác (Accuracy)", value=f"{acc*100:.2f}%")
            st.write("""
            *Giải thích chỉ số:*
            * **Độ chính xác (Accuracy):** Biểu thị tỷ lệ phần trăm số mẫu dự đoán chính xác (cả ca rủi ro và không rủi ro) trên tổng số lượng mẫu thuộc tập kiểm thử.
            """)
            
        with col2:
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                               labels=dict(x="Dự báo (Predicted)", y="Thực tế (Actual)"),
                               x=['Không Rủi Ro (0)', 'Có Rủi Ro (1)'],
                               y=['Không Rủi Ro (0)', 'Có Rủi Ro (1)'],
                               title="Ma trận nhầm lẫn (Confusion Matrix)")
            st.plotly_chart(fig_cm, use_container_width=True)

# ==========================================
# THÀNH PHẦN 6: TAB SỬ DỤNG MÔ HÌNH ĐỂ DỰ BÁO
# ==========================================
with tab4:
    # ĐOẠN SỬA LỖI: Kiểm tra đồng thời cả cờ trạng thái lẫn sự tồn tại thực tế của mô hình
    if not st.session_state.get('is_trained') or 'model' not in st.session_state:
        st.info("Vui lòng bấm nút **'🚀 Huấn luyện mô hình'** ở thanh bên trái để có thể kích hoạt tính năng dự báo.")
    else:
        st.subheader("Dự báo dữ liệu mới")
        model = st.session_state['model']
        
        mode = st.radio("Chế độ dự báo:", ["Nhập thủ công (1 khách hàng)", "Tải lên file CSV (Nhiều khách hàng)"], horizontal=True)
        
        if mode == "Nhập thủ công (1 khách hàng)":
            st.write("Vui lòng nhập điểm số đánh giá (từ 1 đến 5) cho các tiêu chí dưới đây:")
            with st.form("predict_form"):
                cols = st.columns(5)
                input_data = {}
                
                # Chia lưới phân bổ 24 biến vào 5 cột giao diện cho gọn gàng
                chunk_size = len(FEATURES) // 5 + 1
                for i, feature in enumerate(FEATURES):
                    col_idx = i // chunk_size
                    with cols[col_idx]:
                        # Giá trị mặc định ban đầu dựa vào số trung vị (median) của file dữ liệu
                        default_val = int(df[feature].median()) if feature in df.columns else 3
                        input_data[feature] = st.number_input(feature, min_value=1, max_value=5, value=default_val)
                
                submit = st.form_submit_button("🔮 Chẩn đoán rủi ro", type="primary")
                
                if submit:
                    input_df = pd.DataFrame([input_data])
                    pred = model.predict(input_df)[0]
                    prob = model.predict_proba(input_df)[0]
                    
                    st.divider()
                    if pred == 0:
                        st.success(f"✅ **Kết quả chẩn đoán:** Khách hàng An Toàn / KHÔNG có rủi ro (Xác suất an toàn: {prob[0]*100:.2f}%)")
                    else:
                        st.error(f"⚠️ **Kết quả chẩn đoán:** Khách hàng Rủi Ro / CÓ rủi ro tiềm ẩn (Xác suất rủi ro: {prob[1]*100:.2f}%)")
                        
        else:
            batch_file = st.file_uploader("Tải lên file CSV chứa dữ liệu khách hàng mới (cần có đủ 24 cột cấu trúc tương đương)", type=['csv'], key="batch_upload")
            if batch_file:
                batch_df = pd.read_csv(batch_file)
                missing_batch = [col for col in FEATURES if col not in batch_df.columns]
                
                if missing_batch:
                    st.error(f"File tải lên không hợp lệ, thiếu các cột đặc trưng sau: {missing_batch}")
                else:
                    X_batch = batch_df[FEATURES]
                    batch_preds = model.predict(X_batch)
                    batch_probs = model.predict_proba(X_batch)[:, 1]
                    
                    res_df = batch_df.copy()
                    res_df['Dự báo PD'] = batch_preds
                    res_df['Xác suất có rủi ro (%)'] = (batch_probs * 100).round(2)
                    
                    st.write("**Bảng kết quả kết xuất hàng loạt:**")
                    with st.container(height=300):
                        st.dataframe(res_df, use_container_width=True)
                    
                    # Xuất file CSV đã chấm điểm rủi ro hỗ trợ tiếng Việt (utf-8-sig)
                    csv_data = res_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                    st.download_button("📥 Tải báo cáo kết quả (CSV)", data=csv_data, file_name="Ket_qua_du_bao_hang_loat.csv", mime="text/csv")
