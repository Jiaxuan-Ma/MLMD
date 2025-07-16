import streamlit as st
import sqlite3
from passlib.hash import pbkdf2_sha256
import re
from MLMD import mlmd
from streamlit_extras.colored_header import colored_header
# 初始化数据库
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # 创建用户表
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  email TEXT UNIQUE,
                  password_hash TEXT,
                  full_name TEXT,
                  age INTEGER)''')
                  
    # 创建管理员表
    c.execute('''CREATE TABLE IF NOT EXISTS admins
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password_hash TEXT)''')
    
    # 添加默认管理员（初始密码为admin123）
    try:
        c.execute("INSERT INTO admins (username, password_hash) VALUES (?, ?)",
                  ('admin', pbkdf2_sha256.hash("admin123")))
    except sqlite3.IntegrityError:
        pass
    
    conn.commit()
    conn.close()

init_db()

# 自定义CSS样式
def local_css():
    st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            border: none;
        }
        .stTextInput>div>div>input {
            padding: 0.5rem;
            border-radius: 0.5rem;
        }
        .form-container {
            background: white;
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            max-width: 400px;
            margin: 2rem auto;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# 数据库操作函数
def create_user(username, email, password, full_name, age):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        password_hash = pbkdf2_sha256.hash(password)
        c.execute("INSERT INTO users (username, email, password_hash, full_name, age) VALUES (?, ?, ?, ?, ?)",
                (username, email, password_hash, full_name, age))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(identifier, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # 通过用户名或邮箱查找用户
    c.execute("SELECT password_hash FROM users WHERE username = ? OR email = ?", 
            (identifier, identifier))
    result = c.fetchone()
    conn.close()
    
    if result and pbkdf2_sha256.verify(password, result[0]):
        return True
    return False

def get_all_users():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id, username, email, full_name, age FROM users")
    users = c.fetchall()
    conn.close()
    return users

def delete_user(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

def verify_admin(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT password_hash FROM admins WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    
    if result and pbkdf2_sha256.verify(password, result[0]):
        return True
    return False

# 登录状态管理
def login_user(username):
    st.session_state.logged_in = True
    st.session_state.username = username

def logout():
    st.session_state.logged_in = False
    st.session_state.is_admin = False
    st.session_state.show_register = False
    st.session_state.show_profile = False
    st.session_state.username = None

# 输入验证函数
def validate_email(email):
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(pattern, email)

# 主程序
def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'is_admin' not in st.session_state:
        st.session_state.is_admin = False
    if 'show_register' not in st.session_state:
        st.session_state.show_register = False
    if 'show_profile' not in st.session_state:
        st.session_state.show_profile = False

    if st.session_state.logged_in:
        if st.session_state.is_admin:
            # 管理员界面
            st.title("管理员面板")
            users = get_all_users()
            
            if users:
                st.write("### 用户列表")
                for user in users:
                    cols = st.columns([2,3,3,2,2,2])
                    cols[0].write(user[1])  # 用户名
                    cols[1].write(user[2])  # 邮箱
                    cols[2].write(user[3])  # 姓名
                    cols[3].write(user[4])  # 年龄
                    if cols[5].button("删除", key=f"delete_{user[0]}"):
                        delete_user(user[0])
                        st.rerun()
            else:
                st.info("暂无用户信息")
            
            if st.button("退出管理员模式"):
                logout()
        else:
            # 用户功能界面
            # st.title(f"欢迎回来，{st.session_state.username}！")
            # st.write("这是您的个人中心")
            mlmd()
            with st.sidebar:
                if st.button("退出登录"):
                    logout()
    else:
        # 登录/注册界面
        st.image("logo.png")
        colored_header(label="力学结构智能设计平台",description="",color_name="violet-90")
        if st.session_state.show_register:
            # 注册表单
            with st.form("注册表单"):
                st.write("### 用户注册")
                new_username = st.text_input("用户名")
                new_email = st.text_input("邮箱")
                password = st.text_input("密码", type="password")
                confirm_password = st.text_input("确认密码", type="password")
                
                if st.form_submit_button("注册"):
                    if not all([new_username, new_email, password, confirm_password]):
                        st.error("请填写所有字段")
                    elif password != confirm_password:
                        st.error("密码不一致")
                    elif not validate_email(new_email):
                        st.error("请输入有效的邮箱地址")
                    else:
                        if create_user(new_username, new_email, password, "", 0):
                            st.session_state.show_profile = True
                            st.session_state.temp_username = new_username
                            st.session_state.show_register = False
                        else:
                            st.error("用户名或邮箱已被注册")

            if st.button("返回登录"):
                st.session_state.show_register = False

        elif st.session_state.show_profile:
            # 完善资料界面
            with st.form("资料表单"):
                st.write("### 请完善您的资料")
                full_name = st.text_input("真实姓名")
                age = st.number_input("年龄", min_value=0, max_value=150, value=0)
                
                if st.form_submit_button("提交"):
                    conn = sqlite3.connect('users.db')
                    c = conn.cursor()
                    c.execute("UPDATE users SET full_name = ?, age = ? WHERE username = ?",
                             (full_name, age, st.session_state.temp_username))
                    conn.commit()
                    conn.close()
                    st.session_state.show_profile = False
                    st.success("资料更新成功！")
                    st.rerun()

        else:
            # 登录表单
            with st.form("登录表单"):
                identifier = st.text_input("用户名/邮箱")
                password = st.text_input("密码", type="password")
                admin_mode = st.checkbox("管理员模式")
                
                if st.form_submit_button("登录"):
                    if admin_mode:
                        if verify_admin(identifier, password):
                            st.session_state.logged_in = True
                            st.session_state.is_admin = True
                        else:
                            st.error("管理员认证失败")
                    else:
                        if verify_user(identifier, password):
                            login_user(identifier)
                        else:
                            st.error("用户名/邮箱或密码错误")

            if st.button("前往注册"):
                st.session_state.show_register = True

if __name__ == "__main__":
    main()