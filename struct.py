import streamlit as st
import numpy as np
import plotly.graph_objects as go
from skimage import measure
import io
import tempfile
import os

# 尝试导入 numpy-stl 库
try:
    from stl import mesh as stl_mesh
except ImportError:
    st.error("请安装 numpy-stl 库：pip install numpy-stl")
    st.stop()

st.title("TPMS-GAN结构生成模块")

# ------------------------------
# 参数设置（侧边栏）
# ------------------------------
st.sidebar.header("参数设置")
# 每个方向的单胞数量，例如2表示2x2x2
num_cells = st.sidebar.number_input("单胞数量 (每个方向)", min_value=1, value=2, step=1)
# 单胞大小（默认2π，一个周期）
cell_size = st.sidebar.number_input("单胞大小 (默认2π)", min_value=1.0, value=2 * np.pi, step=0.1)
# 每个单胞内的网格分辨率（分辨率越高效果越精细，但计算量会增加）
grid_resolution_per_cell = st.sidebar.number_input("每个单胞的网格分辨率", min_value=10, value=50, step=5)
# 壁厚参数，值越大壁厚越厚（建议0~1之间）
thickness = st.sidebar.slider("壁厚参数", min_value=0.0, max_value=1.0, value=0.3, step=0.01)

# ------------------------------
# 网格生成
# ------------------------------
grid_size = int(grid_resolution_per_cell * num_cells)
# 构造整体空间，范围为 [-num_cells*cell_size/2, num_cells*cell_size/2]
x = np.linspace(-num_cells * cell_size / 2, num_cells * cell_size / 2, grid_size)
y = np.linspace(-num_cells * cell_size / 2, num_cells * cell_size / 2, grid_size)
z = np.linspace(-num_cells * cell_size / 2, num_cells * cell_size / 2, grid_size)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# ------------------------------
# 计算Gyroid函数
# ------------------------------
# 为保证函数周期与单胞大小一致，定义缩放因子：当 x 变化 cell_size 时，对应2π周期
scale_factor = 2 * np.pi / cell_size
# Gyroid的隐式函数
F = np.sin(scale_factor * X) * np.cos(scale_factor * Y) + \
    np.sin(scale_factor * Y) * np.cos(scale_factor * Z) + \
    np.sin(scale_factor * Z) * np.cos(scale_factor * X)
    

    
# 根据壁厚参数构造二值体数据（实体区域）
volume = np.abs(F) < thickness

# ------------------------------
# 提取TPMS结构表面（Marching Cubes算法）
# ------------------------------
verts, faces, normals, values = measure.marching_cubes(volume.astype(float), level=0.5)
# 将体素坐标映射为实际坐标
scale_x = x[1] - x[0]
scale_y = y[1] - y[0]
scale_z = z[1] - z[0]
verts_scaled = np.empty_like(verts)
verts_scaled[:, 0] = verts[:, 0] * scale_x + x[0]
verts_scaled[:, 1] = verts[:, 1] * scale_y + y[0]
verts_scaled[:, 2] = verts[:, 2] * scale_z + z[0]

# ------------------------------
# 生成交互式Plotly可视化图像
# ------------------------------
mesh3d = go.Mesh3d(
    x=verts_scaled[:, 0],
    y=verts_scaled[:, 1],
    z=verts_scaled[:, 2],
    i=faces[:, 0],
    j=faces[:, 1],
    k=faces[:, 2],
    opacity=0.5,
    color='lightblue',
    flatshading=True
)
fig = go.Figure(data=[mesh3d])
fig.update_layout(
    scene=dict(aspectmode='data'),
    title=f"带壁厚的Gyroid TPMS结构: {num_cells}x{num_cells}x{num_cells} 单胞, 单胞大小 = {cell_size:.2f}"
)
# 显示交互式图形，支持鼠标旋转、缩放等操作
st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# STL导出功能
# ------------------------------
def generate_stl(verts, faces):
    """使用numpy-stl将网格数据保存为STL格式，先写入临时文件，再读取其内容"""
    stl_mesh_data = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            stl_mesh_data.vectors[i][j] = verts[face[j]]
    # 使用临时文件保存STL
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
        tmp_name = tmp.name
    stl_mesh_data.save(tmp_name)
    with open(tmp_name, 'rb') as f:
        stl_bytes = f.read()
    os.remove(tmp_name)
    return stl_bytes

st.sidebar.header("导出STL文件")
if st.sidebar.button("生成并下载 STL 文件"):
    stl_data = generate_stl(verts_scaled, faces)
    st.download_button(
        label="点击下载 STL 文件",
        data=stl_data,
        file_name="gyroid_tpms.stl",
        mime="application/octet-stream"
    )
