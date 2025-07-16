import streamlit as st
import pandas as pd
import sqlite3
import os

# 设置默认存储路径
DEFAULT_STORAGE_PATH = "data"

def save_uploaded_file(uploaded_file, storage_path):
    """将上传的文件保存到指定路径"""
    file_path = os.path.join(storage_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def list_existing_files(storage_path):
    """列出存储路径中的数据库文件"""
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)  # 确保目录存在
    return [f for f in os.listdir(storage_path) if f.endswith(('.csv', '.db', '.sqlite'))]

def load_csv(file_path):
    """加载 CSV 文件"""
    df = pd.read_csv(file_path)
    st.subheader("数据预览")
    st.dataframe(df)
    
    st.subheader("描述性统计")
    st.write(df.describe())
    
    # 提供下载按钮
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 下载数据", csv, file_name="exported_data.csv", mime="text/csv")

def load_sqlite(file_path):
    """加载 SQLite 数据库并展示表"""
    conn = sqlite3.connect(file_path)
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables_df = pd.read_sql_query(query, conn)
    
    st.subheader("数据库中的表")
    st.dataframe(tables_df)
    
    if not tables_df.empty:
        table_name = st.selectbox("🔎 选择一个表进行预览", tables_df['name'].tolist())
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        st.subheader(f"表 {table_name} 的数据预览")
        st.dataframe(df)
        
        # 提供下载按钮
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(f"📥 下载 {table_name} 数据", csv, file_name=f"{table_name}.csv", mime="text/csv")

    conn.close()
