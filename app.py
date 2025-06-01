import streamlit as st
import subprocess
import threading
import os
import time
import json
import zipfile
import requests
import shutil
import yaml
import platform
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import glob
import random
import base64
import sys

# 跨平台信号处理
try:
    import signal
except ImportError:
    signal = None  # Windows某些情况下可能不支持某些信号

# 状态文件
STATUS_FILE = "test_status.json"
OUTPUT_FILE = "test_output.txt"
DATASET_INFO_FILE = "dataset_info.json"
CONVERSION_OUTPUT_FILE = "conversion_output.txt"
MAPPING_FILE = "pt_dataset_mapping.json"  # 新增: 映射关系文件

# Docker镜像配置
REQUIRED_DOCKER_IMAGES = [
    "lintheyoung/yolov11-trainer:latest",  # 用于训练和ONNX转换
    "lintheyoung/tpuc_dev_env_build"       # 用于CviModel转换
]

# ==================== 跨平台兼容性工具函数 ====================

def get_platform_info():
    """获取平台信息"""
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "is_windows": platform.system() == "Windows",
        "is_linux": platform.system() == "Linux",
        "is_macos": platform.system() == "Darwin"
    }

def normalize_path_for_docker(local_path):
    """将本地路径转换为Docker挂载格式"""
    abs_path = os.path.abspath(local_path)
    
    if platform.system() == "Windows":
        # Windows: C:\path -> /c/path
        if len(abs_path) > 1 and abs_path[1] == ':':
            drive = abs_path[0].lower()
            path = abs_path[2:].replace('\\', '/')
            return f"/{drive}{path}"
    
    # Linux/Mac: 直接使用，但确保使用正斜杠
    return abs_path.replace(os.sep, "/")

def safe_chmod(file_path, mode=0o755):
    """安全的chmod操作，跨平台兼容"""
    if platform.system() != "Windows":
        try:
            os.chmod(file_path, mode)
            return True
        except OSError as e:
            print(f"⚠️ 设置文件权限失败: {e}")
            return False
    return True  # Windows下跳过chmod

def terminate_process_cross_platform(pid):
    """跨平台进程终止"""
    if not pid:
        return False
        
    try:
        if platform.system() == "Windows":
            # Windows使用taskkill，设置正确的编码
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['CHCP'] = '65001'
            
            result = subprocess.run(f"taskkill /F /PID {pid}", 
                                  shell=True, check=False, 
                                  capture_output=True, text=True,
                                  encoding='utf-8', errors='replace', env=env)
            return result.returncode == 0
        else:
            # Linux/Mac使用信号
            if signal is None:
                print("信号模块不可用，无法终止进程")
                return False
                
            try:
                os.kill(pid, signal.SIGTERM)  # 先尝试温和终止
                time.sleep(2)
                # 检查进程是否还存在
                os.kill(pid, 0)  # 检查进程是否存在
                os.kill(pid, signal.SIGKILL)  # 强制终止
            except ProcessLookupError:
                pass  # 进程已经终止
            return True
    except Exception as e:
        print(f"终止进程失败: {e}")
        return False

# ==================== 添加编码安全的subprocess包装函数 ====================

def run_subprocess_safe(cmd, timeout=30, shell=False, cwd=None):
    """安全的subprocess调用，处理编码问题"""
    try:
        # 设置环境变量强制UTF-8输出
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        if platform.system() == "Windows":
            env['CHCP'] = '65001'  # UTF-8 code page for Windows
        
        if isinstance(cmd, str):
            shell = True
        
        result = subprocess.run(
            cmd,
            shell=shell,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='replace',
            env=env,
            cwd=cwd
        )
        return result
    except Exception as e:
        print(f"subprocess执行失败: {e}")
        return None

def create_subprocess_safe(cmd, cwd=None):
    """创建安全的subprocess.Popen，处理编码问题"""
    try:
        # 设置环境变量强制UTF-8输出
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        if platform.system() == "Windows":
            env['CHCP'] = '65001'  # UTF-8 code page for Windows
        
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace',  # 遇到无法解码的字符时替换而不是报错
            bufsize=1,
            env=env,
            cwd=cwd
        )
        return process
    except Exception as e:
        print(f"创建subprocess失败: {e}")
        return None

def create_directory_safe(directory_path):
    """安全创建目录，处理权限问题"""
    try:
        os.makedirs(directory_path, exist_ok=True)
        # 在Linux下设置合适的权限
        if platform.system() != "Windows":
            try:
                os.chmod(directory_path, 0o755)
            except OSError:
                pass
        return True
    except Exception as e:
        print(f"创建目录失败 {directory_path}: {e}")
        return False

def get_temp_directory():
    """获取跨平台临时目录"""
    return tempfile.gettempdir()

# ==================== Docker环境检查函数 ====================

def check_docker_environment():
    """检查Docker环境是否可用"""
    try:
        print("🔍 检查Docker环境...")
        
        # 设置环境变量强制UTF-8输出（Windows兼容）
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        if platform.system() == "Windows":
            env['CHCP'] = '65001'  # UTF-8 code page for Windows
        
        # 检查Docker是否安装
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10,
                              encoding='utf-8', errors='replace', env=env)
        if result.returncode != 0:
            print("❌ Docker未安装或无法访问")
            print("请安装Docker: https://docs.docker.com/get-docker/")
            return False
        
        print(f"✅ Docker已安装: {result.stdout.strip()}")
        
        # 检查Docker是否运行
        result = subprocess.run(['docker', 'info'], 
                              capture_output=True, text=True, timeout=10,
                              encoding='utf-8', errors='replace', env=env)
        if result.returncode != 0:
            print("❌ Docker服务未运行")
            print("请启动Docker服务")
            return False
        
        print("✅ Docker服务正在运行")
        
        # 检查Docker权限（主要针对Linux）
        if not check_docker_permissions():
            return False
            
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Docker命令超时，请检查Docker是否正常运行")
        return False
    except FileNotFoundError:
        print("❌ 未找到Docker命令，请确认Docker已正确安装")
        return False
    except Exception as e:
        print(f"❌ 检查Docker环境时发生错误: {str(e)}")
        return False

def check_docker_permissions():
    """检查Docker权限（Linux特有问题）"""
    try:
        # 设置环境变量强制UTF-8输出
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        if platform.system() == "Windows":
            env['CHCP'] = '65001'
        
        result = subprocess.run(['docker', 'ps'], 
                              capture_output=True, text=True, timeout=10,
                              encoding='utf-8', errors='replace', env=env)
        if result.returncode != 0:
            if "permission denied" in result.stderr.lower():
                print("❌ Docker权限不足，请运行:")
                print("  sudo usermod -aG docker $USER")
                print("  然后重新登录或重启系统")
                print("  或者使用sudo运行此应用")
                return False
            else:
                print(f"❌ Docker命令执行失败: {result.stderr}")
                return False
        
        print("✅ Docker权限检查通过")
        return True
    except Exception as e:
        print(f"❌ 检查Docker权限时发生错误: {e}")
        return False

def check_nvidia_docker():
    """检查NVIDIA Docker支持"""
    try:
        print("🔍 检查NVIDIA Docker支持...")
        
        # 设置环境变量强制UTF-8输出
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        if platform.system() == "Windows":
            env['CHCP'] = '65001'
        
        result = subprocess.run([
            'docker', 'run', '--rm', '--gpus', 'all', 
            'nvidia/cuda:11.8-base-ubuntu20.04', 'nvidia-smi'
        ], capture_output=True, text=True, timeout=30,
           encoding='utf-8', errors='replace', env=env)
        
        if result.returncode == 0:
            print("✅ NVIDIA Docker支持正常")
            return True
        else:
            print("⚠️ NVIDIA Docker支持不可用，将使用CPU训练")
            return False
    except Exception as e:
        print(f"⚠️ 检查NVIDIA Docker时出错: {e}")
        return False

def check_docker_image_exists(image_name):
    """检查Docker镜像是否存在"""
    try:
        # 设置环境变量强制UTF-8输出
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        if platform.system() == "Windows":
            env['CHCP'] = '65001'
        
        result = subprocess.run(['docker', 'images', '-q', image_name], 
                              capture_output=True, text=True, timeout=30,
                              encoding='utf-8', errors='replace', env=env)
        return result.returncode == 0 and result.stdout.strip() != ""
    except Exception as e:
        print(f"❌ 检查镜像 {image_name} 时发生错误: {str(e)}")
        return False

def pull_docker_image(image_name):
    """拉取Docker镜像"""
    try:
        print(f"📥 正在下载Docker镜像: {image_name}")
        print("这可能需要几分钟时间，请耐心等待...")
        
        # 设置环境变量强制UTF-8输出
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        if platform.system() == "Windows":
            env['CHCP'] = '65001'
        
        result = subprocess.run(['docker', 'pull', image_name], 
                              capture_output=True, text=True, timeout=1800,
                              encoding='utf-8', errors='replace', env=env)  # 30分钟超时
        
        if result.returncode == 0:
            print(f"✅ 镜像下载成功: {image_name}")
            return True
        else:
            print(f"❌ 镜像下载失败: {image_name}")
            print(f"错误信息: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ 下载镜像 {image_name} 超时")
        return False
    except Exception as e:
        print(f"❌ 下载镜像 {image_name} 时发生错误: {str(e)}")
        return False

def check_and_pull_docker_images():
    """检查并下载所需的Docker镜像"""
    print("🔍 检查所需的Docker镜像...")
    
    missing_images = []
    
    for image in REQUIRED_DOCKER_IMAGES:
        if check_docker_image_exists(image):
            print(f"✅ 镜像已存在: {image}")
        else:
            print(f"⚠️  镜像不存在: {image}")
            missing_images.append(image)
    
    if missing_images:
        print(f"\n📥 需要下载 {len(missing_images)} 个镜像...")
        for image in missing_images:
            if not pull_docker_image(image):
                print(f"❌ 无法下载镜像: {image}")
                return False
    
    print("✅ 所有Docker镜像检查完成")
    return True

def initialize_environment():
    """初始化环境检查"""
    platform_info = get_platform_info()
    
    print("=" * 50)
    print("🚀 MaixCam YOLOv11训练平台 - 环境初始化")
    print("=" * 50)
    print(f"🖥️  操作系统: {platform_info['system']} ({platform_info['machine']})")
    
    # 创建必要的目录
    required_dirs = ["data", "models", "outputs", "transfer"]
    for dir_name in required_dirs:
        if not create_directory_safe(dir_name):
            print(f"❌ 创建目录失败: {dir_name}")
            return False
    
    # 检查Docker环境
    if not check_docker_environment():
        print("❌ Docker环境检查失败，程序可能无法正常运行")
        return False
    
    # Linux下检查NVIDIA Docker（可选）
    if platform_info['is_linux']:
        check_nvidia_docker()
    
    # 检查并下载Docker镜像
    if not check_and_pull_docker_images():
        print("❌ Docker镜像准备失败，程序可能无法正常运行")
        return False
    
    print("✅ 环境初始化完成，程序已准备就绪")
    print("=" * 50)
    return True

# ==================== 状态管理函数 ====================

def init_status():
    """初始化状态"""
    default_status = {
        "status": "idle",
        "pid": None,
        "timestamp": datetime.now().isoformat(),
        "current_run": None  # 添加当前运行的任务标识
    }
    
    if not os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(default_status, f)

def get_status():
    """获取状态"""
    try:
        with open(STATUS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        init_status()
        return get_status()

def set_status(status, pid=None, current_run=None):
    """设置状态"""
    status_data = get_status()
    status_data["status"] = status
    status_data["timestamp"] = datetime.now().isoformat()
    
    if pid is not None:
        status_data["pid"] = pid
        
    if current_run is not None:
        status_data["current_run"] = current_run
        
    with open(STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(status_data, f)

# ==================== 数据集管理函数 ====================

def save_dataset_info(info):
    """保存数据集信息"""
    with open(DATASET_INFO_FILE, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

def get_dataset_info():
    """获取数据集信息"""
    try:
        if os.path.exists(DATASET_INFO_FILE):
            with open(DATASET_INFO_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    except:
        return None

def save_pt_dataset_mapping(pt_file_path, dataset_path, run_name):
    """保存pt文件和数据集的映射关系"""
    try:
        # 读取现有映射
        mapping = {}
        if os.path.exists(MAPPING_FILE):
            with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
        
        # 添加新映射 - 适配新的数据集结构
        mapping[pt_file_path] = {
            "dataset_path": dataset_path,
            "run_name": run_name,
            "created_time": datetime.now().isoformat(),
            "images_path": os.path.join(dataset_path, "images")  # 直接指向images目录
        }
        
        # 保存映射
        with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
            
        return True
    except Exception as e:
        print(f"保存映射关系失败: {e}")
        return False

def get_pt_dataset_mapping(pt_file_path):
    """获取pt文件对应的数据集路径"""
    try:
        if os.path.exists(MAPPING_FILE):
            with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
                
                # 首先尝试直接匹配
                if pt_file_path in mapping:
                    return mapping[pt_file_path]
                
                # 如果直接匹配失败，尝试绝对路径匹配
                abs_path = os.path.abspath(pt_file_path)
                if abs_path in mapping:
                    return mapping[abs_path]
                
                # 如果还是失败，尝试规范化路径匹配
                normalized_path = os.path.normpath(abs_path)
                if normalized_path in mapping:
                    return mapping[normalized_path]
                
                # 最后尝试通过文件名匹配（如果路径分隔符不同）
                for key in mapping.keys():
                    if os.path.normpath(key) == normalized_path:
                        return mapping[key]
                
        return None
    except Exception as e:
        print(f"获取映射关系失败: {e}")
        return None

def get_dataset_labels():
    """从data.yaml中获取标签列表"""
    try:
        data_yaml_path = "data/data.yaml"
        if os.path.exists(data_yaml_path):
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    return data['names']
        return []
    except Exception as e:
        print(f"获取数据集标签失败: {e}")
        return []

# ==================== MUD文件和模型包处理函数 ====================

def create_mud_file(cvimodel_path, conversion_name):
    """创建MUD配置文件"""
    try:
        # 获取cvimodel文件的目录和文件名（不带扩展名）
        cvimodel_dir = os.path.dirname(cvimodel_path)
        cvimodel_filename = os.path.basename(cvimodel_path)
        cvimodel_basename = os.path.splitext(cvimodel_filename)[0]
        
        # 创建mud文件路径
        mud_filename = f"{cvimodel_basename}.mud"
        mud_path = os.path.join(cvimodel_dir, mud_filename)
        
        # 获取数据集标签
        labels = get_dataset_labels()
        labels_str = ", ".join(labels) if labels else "object"
        
        # MUD文件内容
        mud_content = f"""[basic]
type = cvimodel
model = {cvimodel_filename}

[extra]
model_type = yolo11
input_type = rgb
mean = 0, 0, 0
scale = 0.00392156862745098, 0.00392156862745098, 0.00392156862745098
anchors = 10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326
labels = {labels_str}
"""
        
        # 写入MUD文件
        with open(mud_path, 'w', encoding='utf-8') as f:
            f.write(mud_content)
        
        return mud_path, f"✅ 成功创建MUD配置文件: {mud_filename}"
        
    except Exception as e:
        return None, f"❌ 创建MUD文件失败: {str(e)}"

def create_model_package_zip(cvimodel_path, mud_path, conversion_name):
    """创建模型包ZIP文件"""
    try:
        # 获取文件所在目录
        model_dir = os.path.dirname(cvimodel_path)
        
        # 获取文件基础名称（不带扩展名）
        cvimodel_basename = os.path.splitext(os.path.basename(cvimodel_path))[0]
        zip_filename = f"{cvimodel_basename}.zip"
        zip_path = os.path.join(model_dir, zip_filename)
        
        # 创建ZIP文件
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 添加cvimodel文件
            zipf.write(cvimodel_path, os.path.basename(cvimodel_path))
            # 添加mud文件
            zipf.write(mud_path, os.path.basename(mud_path))
        
        # 获取文件大小
        zip_size = os.path.getsize(zip_path) / (1024 * 1024)  # MB
        
        return zip_path, f"✅ 成功创建模型包: {zip_filename} ({zip_size:.2f} MB)"
        
    except Exception as e:
        return None, f"❌ 创建模型包失败: {str(e)}"

# ==================== 图片处理函数 ====================

def collect_images_from_dataset(images_path, target_count=200):
    """从数据集的images目录中收集图片"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    all_images = []
    
    # 检查images目录是否存在
    if not os.path.exists(images_path):
        print(f"Images目录不存在: {images_path}")
        return []
    
    # 收集images文件夹中的所有图片
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(images_path, ext)))
        all_images.extend(glob.glob(os.path.join(images_path, ext.upper())))
    
    # 去重并随机打乱
    all_images = list(set(all_images))
    random.shuffle(all_images)
    
    print(f"在 {images_path} 中找到 {len(all_images)} 张图片")
    
    return all_images

def copy_images_to_transfer(images_list, target_dir, target_count=200):
    """复制图片到transfer目录"""
    try:
        # 创建images目录
        images_dir = os.path.join(target_dir, "images")
        create_directory_safe(images_dir)
        
        copied_images = []
        
        # 如果图片数量足够
        if len(images_list) >= target_count:
            selected_images = images_list[:target_count]
            for i, img_path in enumerate(selected_images):
                if os.path.exists(img_path):
                    file_ext = os.path.splitext(img_path)[1]
                    target_name = f"image_{i+1:03d}{file_ext}"
                    target_path = os.path.join(images_dir, target_name)
                    shutil.copy2(img_path, target_path)
                    copied_images.append(target_path)
        
        # 如果图片数量不够，重复复制并重命名
        else:
            available_count = len(images_list)
            if available_count == 0:
                return [], None
            
            for i in range(target_count):
                source_img = images_list[i % available_count]  # 循环使用现有图片
                if os.path.exists(source_img):
                    file_ext = os.path.splitext(source_img)[1]
                    target_name = f"image_{i+1:03d}{file_ext}"
                    target_path = os.path.join(images_dir, target_name)
                    shutil.copy2(source_img, target_path)
                    copied_images.append(target_path)
        
        # 复制一张图片作为test图片
        test_image = None
        if copied_images:
            test_source = copied_images[0]
            file_ext = os.path.splitext(test_source)[1]
            test_image = os.path.join(target_dir, f"test{file_ext}")
            shutil.copy2(test_source, test_image)
        
        return copied_images, test_image
        
    except Exception as e:
        print(f"复制图片失败: {e}")
        return [], None

# ==================== 数据集下载和处理函数 ====================

def download_file(url, local_filename, progress_placeholder=None):
    """下载文件"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    if progress_placeholder and total_size > 0:
                        progress = downloaded_size / total_size
                        progress_placeholder.progress(progress)
        
        return True
    except Exception as e:
        st.error(f"下载失败: {str(e)}")
        return False

def extract_zip(zip_path, extract_to):
    """解压ZIP文件"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        st.error(f"解压失败: {str(e)}")
        return False

def find_data_yaml(directory):
    """递归查找data.yaml文件"""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower() in ['data.yaml', 'data.yml']:
                return os.path.join(root, file)
    return None

def validate_dataset(data_yaml_path):
    """验证数据集格式"""
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 检查必要的字段
        required_fields = ['train', 'val', 'names']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return False, f"缺少必要字段: {missing_fields}"
        
        # 检查类别数量
        if 'nc' not in data:
            data['nc'] = len(data['names'])
        
        return True, data
    except Exception as e:
        return False, f"解析YAML文件失败: {str(e)}"

def process_uploaded_dataset(uploaded_file):
    """处理上传的数据集"""
    try:
        # 创建临时目录
        temp_dir = os.path.join(get_temp_directory(), "dataset_upload")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        create_directory_safe(temp_dir)
        
        # 保存上传的文件
        zip_path = os.path.join(temp_dir, "dataset.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 解压文件
        extract_dir = os.path.join(temp_dir, "extracted")
        if not extract_zip(zip_path, extract_dir):
            return False, "解压失败"
        
        # 查找data.yaml文件
        data_yaml_path = find_data_yaml(extract_dir)
        if not data_yaml_path:
            return False, "未找到data.yaml文件"
        
        # 验证数据集
        is_valid, result = validate_dataset(data_yaml_path)
        if not is_valid:
            return False, result
        
        # 移动到data目录
        data_dir = "data"
        if os.path.exists(data_dir):
            # 备份原有数据
            backup_dir = f"data_backup_{int(time.time())}"
            shutil.move(data_dir, backup_dir)
            st.info(f"原数据集已备份到: {backup_dir}")
        
        # 移动新数据集
        dataset_root = os.path.dirname(data_yaml_path)
        shutil.move(dataset_root, data_dir)
        
        # 保存数据集信息
        dataset_info = {
            "source": "upload",
            "filename": uploaded_file.name,
            "upload_time": datetime.now().isoformat(),
            "classes": result['names'],
            "num_classes": len(result['names'])
        }
        save_dataset_info(dataset_info)
        
        # 清理临时文件
        shutil.rmtree(temp_dir)
        
        return True, "数据集上传成功"
        
    except Exception as e:
        return False, f"处理上传文件失败: {str(e)}"

def process_url_dataset(url):
    """处理URL下载的数据集"""
    try:
        # 创建临时目录
        temp_dir = os.path.join(get_temp_directory(), "dataset_download")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        create_directory_safe(temp_dir)
        
        # 下载文件
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path) or "dataset.zip"
        zip_path = os.path.join(temp_dir, filename)
        
        # 显示下载进度
        progress_placeholder = st.empty()
        progress_placeholder.text("开始下载...")
        progress_bar = st.progress(0)
        
        # 下载
        response = requests.get(url, stream=True)
        response.raise_for_status()
        response.encoding = 'utf-8'  # 明确设置编码
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    if total_size > 0:
                        progress = downloaded_size / total_size
                        progress_bar.progress(progress)
                        progress_placeholder.text(f"下载中... {downloaded_size/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB")
        
        progress_placeholder.text("下载完成，开始解压...")
        
        # 解压文件
        extract_dir = os.path.join(temp_dir, "extracted")
        if not extract_zip(zip_path, extract_dir):
            return False, "解压失败"
        
        # 查找data.yaml文件
        data_yaml_path = find_data_yaml(extract_dir)
        if not data_yaml_path:
            return False, "未找到data.yaml文件"
        
        # 验证数据集
        is_valid, result = validate_dataset(data_yaml_path)
        if not is_valid:
            return False, result
        
        # 移动到data目录
        data_dir = "data"
        if os.path.exists(data_dir):
            # 备份原有数据
            backup_dir = f"data_backup_{int(time.time())}"
            shutil.move(data_dir, backup_dir)
            st.info(f"原数据集已备份到: {backup_dir}")
        
        # 移动新数据集
        dataset_root = os.path.dirname(data_yaml_path)
        shutil.move(dataset_root, data_dir)
        
        # 保存数据集信息
        dataset_info = {
            "source": "url",
            "url": url,
            "filename": filename,
            "download_time": datetime.now().isoformat(),
            "classes": result['names'],
            "num_classes": len(result['names'])
        }
        save_dataset_info(dataset_info)
        
        # 清理临时文件
        shutil.rmtree(temp_dir)
        progress_placeholder.empty()
        progress_bar.empty()
        
        return True, "数据集下载并配置成功"
        
    except Exception as e:
        return False, f"处理URL数据集失败: {str(e)}"

# ==================== 输出处理函数 ====================

def read_output():
    """读取输出"""
    try:
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        return ""
    except Exception as e:
        return f"读取输出失败: {str(e)}"

def read_conversion_output():
    """读取转换输出"""
    try:
        if os.path.exists(CONVERSION_OUTPUT_FILE):
            with open(CONVERSION_OUTPUT_FILE, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        return ""
    except Exception as e:
        return f"读取转换输出失败: {str(e)}"

def clear_output():
    """清空输出"""
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("")
    except Exception as e:
        print(f"清空输出文件失败: {e}")

def clear_conversion_output():
    """清空转换输出"""
    try:
        with open(CONVERSION_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("")
    except Exception as e:
        print(f"清空转换输出文件失败: {e}")

# ==================== CviModel处理函数 ====================

def find_and_move_cvimodel(transfer_dir, conversion_name, selected_model_name):
    """在workspace目录中查找.cvimodel文件并移动到顶层目录"""
    try:
        # 从模型文件名中提取基本名称（例如：best.pt -> best）
        model_base_name = os.path.splitext(os.path.basename(selected_model_name))[0]
        
        # 查找workspace目录
        workspace_dir = os.path.join(transfer_dir, "workspace")
        if not os.path.exists(workspace_dir):
            return None, None, None, "未找到workspace目录"
        
        # 查找.cvimodel文件
        cvimodel_files = []
        for file in os.listdir(workspace_dir):
            if file.endswith('.cvimodel'):
                cvimodel_files.append(file)
        
        if not cvimodel_files:
            return None, None, None, "未找到.cvimodel文件"
        
        # 寻找匹配的文件（优先查找包含模型基本名称的文件）
        target_cvimodel = None
        for file in cvimodel_files:
            if model_base_name in file:
                target_cvimodel = file
                break
        
        # 如果没有找到匹配的，使用第一个
        if not target_cvimodel:
            target_cvimodel = cvimodel_files[0]
        
        # 构造新的文件名：export_时间戳_int8.cvimodel
        new_filename = f"{conversion_name}_int8.cvimodel"
        
        # 源文件路径和目标文件路径
        source_path = os.path.join(workspace_dir, target_cvimodel)
        target_path = os.path.join(transfer_dir, new_filename)
        
        # 移动并重命名文件
        shutil.move(source_path, target_path)
        
        # 创建MUD文件
        mud_path, mud_message = create_mud_file(target_path, conversion_name)
        
        # 创建模型包ZIP文件
        zip_path = None
        zip_message = ""
        if mud_path:
            zip_path, zip_message = create_model_package_zip(target_path, mud_path, conversion_name)
        
        return target_path, mud_path, zip_path, f"✅ 成功移动并重命名: {target_cvimodel} -> {new_filename}\n{mud_message}\n{zip_message}"
        
    except Exception as e:
        return None, None, None, f"❌ 移动.cvimodel文件失败: {str(e)}"

# ==================== Docker命令构建函数 ====================

def build_docker_training_command(model, epochs, imgsz, run_name):
    """构建Docker训练命令"""
    # 获取当前目录的绝对路径
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, "data")
    models_path = os.path.join(current_dir, "models")
    outputs_path = os.path.join(current_dir, "outputs")
    
    # 确保目录存在
    for path in [data_path, models_path, outputs_path]:
        create_directory_safe(path)
    
    # 转换为Docker挂载格式
    docker_data_path = normalize_path_for_docker(data_path)
    docker_models_path = normalize_path_for_docker(models_path)
    docker_outputs_path = normalize_path_for_docker(outputs_path)
    
    # 构建Docker命令
    docker_command = f'''docker run --gpus all --name yolov11-{run_name} --rm --shm-size=4g -v "{docker_data_path}:/workspace/data" -v "{docker_models_path}:/workspace/models" -v "{docker_outputs_path}:/workspace/outputs" lintheyoung/yolov11-trainer:latest bash -c "cd /workspace/models && yolo train data=/workspace/data/data.yaml model={model} epochs={epochs} imgsz={imgsz} project=/workspace/outputs name={run_name}"'''
    
    return docker_command, data_path, models_path, outputs_path

def build_docker_conversion_command(model_path, format, imgsz_height, imgsz_width, opset, conversion_name):
    """构建Docker转换命令"""
    # 获取当前目录的绝对路径
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, "data")
    models_path = os.path.join(current_dir, "models")
    outputs_path = os.path.join(current_dir, "outputs")
    
    # 确保目录存在
    for path in [data_path, models_path, outputs_path]:
        create_directory_safe(path)
    
    # 转换模型路径为Docker容器内路径
    docker_model_path = model_path.replace(outputs_path, "/workspace/outputs")
    docker_model_path = docker_model_path.replace(os.sep, "/")
    
    # 转换为Docker挂载格式
    docker_data_path = normalize_path_for_docker(data_path)
    docker_models_path = normalize_path_for_docker(models_path)
    docker_outputs_path = normalize_path_for_docker(outputs_path)
    
    # 构建Docker命令
    docker_command = f'''docker run --gpus all --name yolo-export-{conversion_name} --rm --shm-size=4g -v "{docker_data_path}:/workspace/data" -v "{docker_models_path}:/workspace/models" -v "{docker_outputs_path}:/workspace/outputs" lintheyoung/yolov11-trainer:latest bash -c "yolo export model={docker_model_path} format={format} imgsz={imgsz_height},{imgsz_width} opset={opset} batch=1"'''
    
    return docker_command

def build_docker_cvimodel_command(transfer_dir):
    """构建CviModel转换命令"""
    # 获取transfer目录的绝对路径
    abs_transfer_dir = os.path.abspath(transfer_dir)
    docker_transfer_path = normalize_path_for_docker(abs_transfer_dir)
    
    # 构建Docker命令
    docker_command = f'''docker run --rm -it -v "{docker_transfer_path}:/workspace" lintheyoung/tpuc_dev_env_build bash -c "cd /workspace && ./convert_cvimodel.sh"'''
    
    return docker_command

# ==================== 训练和转换函数 ====================

def run_docker_training(model, epochs, imgsz):
    """运行Docker训练"""
    def training_task():
        try:
            # 获取当前时间戳（精确到秒）作为训练名称
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"train_{timestamp}"
            
            set_status("running", current_run=run_name)
            clear_output()
            
            # 构建Docker命令
            docker_command, data_path, models_path, outputs_path = build_docker_training_command(
                model, epochs, imgsz, run_name
            )
            
            # 建立映射关系 - 训练开始前就知道pt文件的最终位置
            future_weights_dir = os.path.join(outputs_path, run_name, "weights")
            future_best_pt = os.path.join(future_weights_dir, "best.pt")
            future_last_pt = os.path.join(future_weights_dir, "last.pt")
            
            # 保存映射关系
            save_pt_dataset_mapping(future_best_pt, data_path, run_name)
            save_pt_dataset_mapping(future_last_pt, data_path, run_name)
            
            # 启动进程 - 使用安全的subprocess创建函数
            process = create_subprocess_safe(docker_command)
            
            if process is None:
                set_status("failed")
                with open(OUTPUT_FILE, 'a', encoding='utf-8', errors='replace') as f:
                    f.write("\n❌ 无法启动Docker训练进程")
                return
            
            set_status("running", process.pid, run_name)
            
            # 实时读取输出
            with open(OUTPUT_FILE, 'w', encoding='utf-8', errors='replace') as f:
                f.write(f"开始执行命令:\n{docker_command}\n\n")
                f.write(f"已建立映射关系:\n")
                f.write(f"  - {future_best_pt} -> {data_path}\n")
                f.write(f"  - {future_last_pt} -> {data_path}\n\n")
                f.flush()
                
                for line in iter(process.stdout.readline, ''):
                    if line:
                        try:
                            f.write(line)
                            f.flush()
                        except UnicodeEncodeError:
                            # 如果遇到编码问题，尝试清理字符
                            clean_line = line.encode('utf-8', errors='replace').decode('utf-8')
                            f.write(clean_line)
                            f.flush()
            
            # 等待完成
            return_code = process.wait()
            
            if return_code == 0:
                set_status("completed", current_run=run_name)
                with open(OUTPUT_FILE, 'a', encoding='utf-8', errors='replace') as f:
                    f.write("\n✅ 训练完成!")
            else:
                set_status("failed", current_run=run_name)
                with open(OUTPUT_FILE, 'a', encoding='utf-8', errors='replace') as f:
                    f.write(f"\n❌ 训练失败，退出码: {return_code}")
                    
        except Exception as e:
            set_status("failed")
            with open(OUTPUT_FILE, 'a', encoding='utf-8', errors='replace') as f:
                f.write(f"\n❌ 执行出错: {str(e)}")
    
    # 后台线程运行
    thread = threading.Thread(target=training_task)
    thread.daemon = True
    thread.start()

def run_model_conversion(model_path, format="onnx", opset=18):
    """运行模型转换"""
    def conversion_task():
        try:
            # 获取当前时间戳作为转换名称
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            conversion_name = f"export_{timestamp}"
            
            # 设置状态为转换中
            set_status("converting", current_run=conversion_name)
            clear_conversion_output()
            
            # 获取pt文件的映射关系
            mapping_info = get_pt_dataset_mapping(model_path)
            
            # 创建transfer目录
            transfer_dir = os.path.join("transfer", conversion_name)
            create_directory_safe(transfer_dir)
            
            with open(CONVERSION_OUTPUT_FILE, 'w', encoding='utf-8', errors='replace') as f:
                f.write(f"开始模型转换流程 - {conversion_name}\n")
                f.write(f"模型文件: {model_path}\n")
                f.write(f"绝对路径: {os.path.abspath(model_path)}\n")
                f.write(f"Transfer目录: {transfer_dir}\n\n")
                
                # 调试映射关系查找过程
                f.write("=== 查找数据集映射关系 ===\n")
                f.write(f"查找路径: {model_path}\n")
                f.write(f"绝对路径: {os.path.abspath(model_path)}\n")
                
                # 显示所有映射关系
                if os.path.exists(MAPPING_FILE):
                    with open(MAPPING_FILE, 'r', encoding='utf-8') as map_f:
                        all_mappings = json.load(map_f)
                        f.write(f"映射文件中共有 {len(all_mappings)} 条记录:\n")
                        for key in all_mappings.keys():
                            f.write(f"  - {key}\n")
                else:
                    f.write("映射文件不存在\n")
                
                f.write(f"映射查找结果: {'找到' if mapping_info else '未找到'}\n\n")
                f.flush()
                
                # 如果找到映射关系，先复制数据集图片
                if mapping_info:
                    f.write("=== 数据集图片收集与复制 ===\n")
                    f.write(f"数据集路径: {mapping_info['dataset_path']}\n")
                    f.write(f"Images路径: {mapping_info['images_path']}\n\n")
                    f.flush()
                    
                    # 收集图片 - 使用新的函数调用方式
                    f.write("正在收集图片...\n")
                    f.flush()
                    all_images = collect_images_from_dataset(
                        mapping_info['images_path'], 
                        target_count=200
                    )
                    
                    f.write(f"找到 {len(all_images)} 张图片\n")
                    f.flush()
                    
                    if all_images:
                        # 复制图片
                        f.write("正在复制图片到transfer目录...\n")
                        f.flush()
                        copied_images, test_image = copy_images_to_transfer(all_images, transfer_dir, 200)
                        
                        f.write(f"成功复制 {len(copied_images)} 张图片到 images/ 文件夹\n")
                        if test_image:
                            f.write(f"创建测试图片: {os.path.basename(test_image)}\n")
                        f.write("图片复制完成!\n\n")
                        f.flush()
                    else:
                        f.write("⚠️ 未找到图片文件，跳过图片复制步骤\n\n")
                        f.flush()
                else:
                    f.write("⚠️ 未找到数据集映射关系，跳过图片复制步骤\n\n")
                    f.flush()
                
                # 开始ONNX转换
                f.write("=== ONNX模型转换 ===\n")
                f.flush()
            
            # 定义期望的图像尺寸（符合MaixCam的尺寸）
            imgsz_height = 224
            imgsz_width = 320
            
            # 构建Docker命令
            docker_command = build_docker_conversion_command(
                model_path, format, imgsz_height, imgsz_width, opset, conversion_name
            )
            
            # 启动进程 - 使用安全的subprocess创建函数
            process = create_subprocess_safe(docker_command)
            
            if process is None:
                set_status("failed", current_run=conversion_name)
                with open(CONVERSION_OUTPUT_FILE, 'a', encoding='utf-8', errors='replace') as f:
                    f.write("\n❌ 无法启动Docker转换进程")
                return
            
            # 记录进程ID
            set_status("converting", process.pid, conversion_name)
            
            # 实时读取输出
            with open(CONVERSION_OUTPUT_FILE, 'a', encoding='utf-8', errors='replace') as f:
                f.write(f"执行转换命令:\n{docker_command}\n\n")
                f.flush()
                
                for line in iter(process.stdout.readline, ''):
                    if line:
                        try:
                            f.write(line)
                            f.flush()
                        except UnicodeEncodeError:
                            clean_line = line.encode('utf-8', errors='replace').decode('utf-8')
                            f.write(clean_line)
                            f.flush()
            
            # 等待完成
            return_code = process.wait()
            
            # 转换完成后，复制ONNX模型到transfer目录
            with open(CONVERSION_OUTPUT_FILE, 'a', encoding='utf-8', errors='replace') as f:
                if return_code == 0:
                    f.write("\n=== ONNX转换成功，复制模型文件 ===\n")
                    
                    # 查找生成的ONNX文件
                    model_dir = os.path.dirname(model_path)
                    onnx_files = []
                    
                    if os.path.exists(model_dir):
                        for file in os.listdir(model_dir):
                            if file.endswith(".onnx"):
                                onnx_path = os.path.join(model_dir, file)
                                onnx_files.append(onnx_path)
                    
                    if onnx_files:
                        for onnx_file in onnx_files:
                            target_onnx = os.path.join(transfer_dir, os.path.basename(onnx_file))
                            shutil.copy2(onnx_file, target_onnx)
                            f.write(f"已复制ONNX模型: {os.path.basename(onnx_file)}\n")
                        
                        f.write(f"\n✅ ONNX转换和文件复制完成: {transfer_dir}\n")
                        
                        # 复制convert_cvimodel.sh文件
                        f.write("\n=== 复制转换脚本 ===\n")
                        f.flush()
                        
                        convert_script_path = "convert_cvimodel.sh"
                        if os.path.exists(convert_script_path):
                            target_script_path = os.path.join(transfer_dir, "convert_cvimodel.sh")
                            shutil.copy2(convert_script_path, target_script_path)
                            
                            # 设置执行权限
                            safe_chmod(target_script_path, 0o755)
                            
                            f.write(f"✅ 已复制转换脚本: {convert_script_path}\n")
                            f.flush()
                            
                            # 执行CviModel转换
                            f.write("\n=== 执行CviModel转换 ===\n")
                            f.write(f"切换到目录: {transfer_dir}\n")
                            f.flush()
                            
                            # 构建docker命令
                            cvi_docker_command = build_docker_cvimodel_command(transfer_dir)
                            
                            f.write(f"执行命令:\n{cvi_docker_command}\n\n")
                            f.flush()
                            
                            # 启动CviModel转换进程 - 使用安全的subprocess创建函数
                            cvi_process = create_subprocess_safe(cvi_docker_command, cwd=os.path.abspath(transfer_dir))
                            
                            if cvi_process is None:
                                f.write("❌ 无法启动CviModel转换进程\n")
                                f.write("ONNX模型仍可正常使用\n")
                            else:
                                # 实时读取CviModel转换输出
                                f.write("CviModel转换输出:\n")
                                f.write("-" * 50 + "\n")
                                f.flush()
                                
                                for line in iter(cvi_process.stdout.readline, ''):
                                    if line:
                                        try:
                                            f.write(line)
                                            f.flush()
                                        except UnicodeEncodeError:
                                            clean_line = line.encode('utf-8', errors='replace').decode('utf-8')
                                            f.write(clean_line)
                                            f.flush()
                                
                                # 等待CviModel转换完成
                                cvi_return_code = cvi_process.wait()
                                
                                f.write("-" * 50 + "\n")
                                if cvi_return_code == 0:
                                    f.write("✅ CviModel转换完成!\n")
                                    
                                    # 查找并移动.cvimodel文件，同时创建MUD文件和ZIP包
                                    f.write("\n=== 处理CviModel文件 ===\n")
                                    f.flush()
                                    
                                    # 从模型路径中提取文件名
                                    selected_model_name = os.path.basename(model_path)
                                    moved_file_path, mud_file_path, zip_file_path, move_message = find_and_move_cvimodel(
                                        transfer_dir, conversion_name, selected_model_name
                                    )
                                    
                                    f.write(f"{move_message}\n")
                                    
                                    if moved_file_path:
                                        f.write(f"CviModel文件路径: {moved_file_path}\n")
                                        f.write(f"CviModel文件大小: {os.path.getsize(moved_file_path) / (1024*1024):.2f} MB\n")
                                    
                                    if mud_file_path:
                                        f.write(f"MUD配置文件路径: {mud_file_path}\n")
                                        f.write(f"MUD文件大小: {os.path.getsize(mud_file_path) / 1024:.2f} KB\n")
                                    
                                    if zip_file_path:
                                        f.write(f"模型包ZIP路径: {zip_file_path}\n")
                                        f.write(f"ZIP文件大小: {os.path.getsize(zip_file_path) / (1024*1024):.2f} MB\n")
                                    
                                    f.write(f"\n🎉 完整的MaixCam模型包已创建: {transfer_dir}\n")
                                    f.write("包含内容:\n")
                                    f.write("  - images/ (200张训练图片)\n")
                                    f.write("  - test.png/jpg (测试图片)\n")
                                    f.write("  - *.onnx (ONNX模型)\n")
                                    f.write("  - convert_cvimodel.sh (转换脚本)\n")
                                    if moved_file_path:
                                        final_cvimodel_filename = os.path.basename(moved_file_path)
                                        f.write(f"  - {final_cvimodel_filename} (MaixCam优化模型) 🎯\n")
                                    if mud_file_path:
                                        final_mud_filename = os.path.basename(mud_file_path)
                                        f.write(f"  - {final_mud_filename} (MUD配置文件) 📋\n")
                                    if zip_file_path:
                                        final_zip_filename = os.path.basename(zip_file_path)
                                        f.write(f"  - {final_zip_filename} (完整模型包) 📦\n")
                                    
                                else:
                                    f.write(f"❌ CviModel转换失败，退出码: {cvi_return_code}\n")
                                    f.write("ONNX模型仍可正常使用\n")
                            
                        else:
                            f.write(f"⚠️ 未找到转换脚本: {convert_script_path}\n")
                            f.write("请确保convert_cvimodel.sh文件存在于应用根目录\n")
                            f.write("ONNX转换已完成，可手动进行CviModel转换\n")
                    else:
                        f.write("⚠️ 未找到生成的ONNX文件\n")
                    
                    set_status("completed", current_run=conversion_name)
                    f.write("\n✅ 模型转换和打包完成!")
                else:
                    set_status("failed", current_run=conversion_name)
                    f.write(f"\n❌ 模型转换失败，退出码: {return_code}")
                    
        except Exception as e:
            set_status("failed")
            with open(CONVERSION_OUTPUT_FILE, 'a', encoding='utf-8', errors='replace') as f:
                f.write(f"\n❌ 执行出错: {str(e)}")
    
    # 后台线程运行
    thread = threading.Thread(target=conversion_task)
    thread.daemon = True
    thread.start()

def stop_training():
    """停止训练"""
    status = get_status()
    pid = status.get("pid")
    
    if pid:
        try:
            success = terminate_process_cross_platform(pid)
            set_status("stopped")
            with open(OUTPUT_FILE, 'a', encoding='utf-8', errors='replace') as f:
                if success:
                    f.write(f"\n⏹️ 训练已手动停止 (PID: {pid})")
                else:
                    f.write(f"\n⚠️ 尝试停止训练 (PID: {pid})，请检查进程状态")
                
        except Exception as e:
            with open(OUTPUT_FILE, 'a', encoding='utf-8', errors='replace') as f:
                f.write(f"\n❌ 停止失败: {str(e)}")

def stop_conversion():
    """停止转换过程"""
    status = get_status()
    pid = status.get("pid")
    
    if pid:
        try:
            success = terminate_process_cross_platform(pid)
            set_status("stopped")
            with open(CONVERSION_OUTPUT_FILE, 'a', encoding='utf-8', errors='replace') as f:
                if success:
                    f.write(f"\n⏹️ 模型转换已手动停止 (PID: {pid})")
                else:
                    f.write(f"\n⚠️ 尝试停止模型转换 (PID: {pid})，请检查进程状态")
                
        except Exception as e:
            with open(CONVERSION_OUTPUT_FILE, 'a', encoding='utf-8', errors='replace') as f:
                f.write(f"\n❌ 停止失败: {str(e)}")

# ==================== 信息提取和显示函数 ====================

def extract_training_info(output_content):
    """提取训练关键信息"""
    lines = output_content.split('\n')
    info = {
        "current_epoch": None,
        "total_epochs": None,
        "latest_metrics": None,
        "progress_percentage": 0
    }
    
    # 从最新的几行中提取信息
    for line in reversed(lines[-20:]):
        # 提取epoch信息
        if "Epoch" in line and "/" in line:
            try:
                epoch_part = line.split("Epoch")[1].split()[0]
                if "/" in epoch_part:
                    current, total = epoch_part.split("/")
                    info["current_epoch"] = int(current)
                    info["total_epochs"] = int(total)
                    info["progress_percentage"] = (int(current) / int(total)) * 100
                    break
            except:
                pass
    
    # 提取最新的指标信息
    for line in reversed(lines[-10:]):
        if "mAP50" in line and "all" in line:
            info["latest_metrics"] = line.strip()
            break
    
    return info

def find_training_models():
    """查找训练产生的模型文件"""
    models = []
    outputs_dir = "./outputs"
    
    if os.path.exists(outputs_dir):
        for train_dir in os.listdir(outputs_dir):
            weights_dir = os.path.join(outputs_dir, train_dir, "weights")
            if os.path.isdir(weights_dir):
                # 查找best.pt和last.pt文件
                for weight_file in ["best.pt", "last.pt"]:
                    weight_path = os.path.join(weights_dir, weight_file)
                    if os.path.exists(weight_path):
                        # 添加模型信息
                        model_info = {
                            "name": f"{train_dir}/{weight_file}",
                            "path": weight_path,
                            "size": os.path.getsize(weight_path) / (1024 * 1024),  # MB
                            "time": datetime.fromtimestamp(os.path.getmtime(weight_path))
                        }
                        models.append(model_info)
    
    # 按修改时间排序，最新的在前面
    models.sort(key=lambda x: x["time"], reverse=True)
    return models

def find_converted_cvimodels():
    """查找转换完成的.cvimodel文件"""
    cvimodels = []
    transfer_dir = "transfer"
    
    if os.path.exists(transfer_dir):
        for export_dir in os.listdir(transfer_dir):
            if export_dir.startswith('export_'):
                export_path = os.path.join(transfer_dir, export_dir)
                if os.path.isdir(export_path):
                    # 查找.cvimodel文件
                    for file in os.listdir(export_path):
                        if file.endswith('.cvimodel'):
                            cvimodel_path = os.path.join(export_path, file)
                            if os.path.isfile(cvimodel_path):
                                # 添加模型信息
                                cvimodel_info = {
                                    "name": file,
                                    "path": cvimodel_path,
                                    "size": os.path.getsize(cvimodel_path) / (1024 * 1024),  # MB
                                    "time": datetime.fromtimestamp(os.path.getmtime(cvimodel_path)),
                                    "export_dir": export_dir
                                }
                                cvimodels.append(cvimodel_info)
    
    # 按修改时间排序，最新的在前面
    cvimodels.sort(key=lambda x: x["time"], reverse=True)
    return cvimodels

def find_model_packages():
    """查找模型包ZIP文件"""
    packages = []
    transfer_dir = "transfer"
    
    if os.path.exists(transfer_dir):
        for export_dir in os.listdir(transfer_dir):
            if export_dir.startswith('export_'):
                export_path = os.path.join(transfer_dir, export_dir)
                if os.path.isdir(export_path):
                    # 查找ZIP文件
                    for file in os.listdir(export_path):
                        if file.endswith('.zip') and '_int8.zip' in file:
                            zip_path = os.path.join(export_path, file)
                            if os.path.isfile(zip_path):
                                # 添加包信息
                                package_info = {
                                    "name": file,
                                    "path": zip_path,
                                    "size": os.path.getsize(zip_path) / (1024 * 1024),  # MB
                                    "time": datetime.fromtimestamp(os.path.getmtime(zip_path)),
                                    "export_dir": export_dir
                                }
                                packages.append(package_info)
    
    # 按修改时间排序，最新的在前面
    packages.sort(key=lambda x: x["time"], reverse=True)
    return packages

def display_results():
    """显示训练结果"""
    # 获取当前运行状态
    status = get_status()
    current_run = status.get("current_run")
    current_status = status.get("status")
    
    # 如果当前正在运行训练，则不显示结果
    if current_status == "running":
        st.info("🔄 训练进行中，完成后将显示结果")
        return
    
    # 如果没有当前运行的任务，则寻找最新的结果
    if not current_run and current_status != "completed":
        outputs_dir = "./outputs"
        if os.path.exists(outputs_dir):
            # 查找最新的训练结果
            train_dirs = []
            for item in os.listdir(outputs_dir):
                item_path = os.path.join(outputs_dir, item)
                if os.path.isdir(item_path) and item.startswith('train_'):
                    train_dirs.append(item)
            
            if train_dirs:
                # 按修改时间排序，获取最新的
                current_run = max(train_dirs, key=lambda x: os.path.getctime(os.path.join(outputs_dir, x)))
    
    # 如果有当前任务或找到了最新的结果
    if current_run:
        results_path = os.path.join("./outputs", current_run)
        
        if os.path.exists(results_path):
            st.subheader("📊 训练结果")
            st.write(f"结果目录: {results_path}")
            
            # 显示结果图片
            image_files = ['results.png', 'confusion_matrix.png', 'F1_curve.png', 'PR_curve.png']
            
            cols = st.columns(2)
            col_idx = 0
            
            for img_file in image_files:
                img_path = os.path.join(results_path, img_file)
                if os.path.exists(img_path):
                    with cols[col_idx % 2]:
                        st.image(img_path, caption=img_file.replace('.png', '').replace('_', ' ').title())
                    col_idx += 1
            
            # 显示权重文件
            weights_dir = os.path.join(results_path, 'weights')
            if os.path.exists(weights_dir):
                st.subheader("💾 模型权重")
                for weight_file in os.listdir(weights_dir):
                    weight_path = os.path.join(weights_dir, weight_file)
                    if os.path.isfile(weight_path):
                        file_size = os.path.getsize(weight_path) / (1024 * 1024)  # MB
                        st.write(f"📁 {weight_file} ({file_size:.1f} MB)")
        else:
            st.info("暂无训练结果（可以刷新一下）")
    else:
        st.info("暂无训练结果（可以刷新一下）")

# ==================== UI部分函数 ====================

def dataset_management_section():
    """数据集管理部分"""
    st.subheader("📦 数据集管理")
    
    # 显示当前数据集信息
    dataset_info = get_dataset_info()
    data_yaml_exists = os.path.exists("data/data.yaml")
    
    if data_yaml_exists and dataset_info:
        st.success("✅ 数据集已配置")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**来源:** {'文件上传' if dataset_info['source'] == 'upload' else 'URL下载'}")
            if dataset_info['source'] == 'upload':
                st.info(f"**文件名:** {dataset_info['filename']}")
                st.info(f"**上传时间:** {dataset_info['upload_time'][:19]}")
            else:
                st.info(f"**URL:** {dataset_info['url']}")
                st.info(f"**下载时间:** {dataset_info['download_time'][:19]}")
        
        with col2:
            st.info(f"**类别数量:** {dataset_info['num_classes']}")
            st.info(f"**类别名称:** {', '.join(dataset_info['classes'][:5])}{'...' if len(dataset_info['classes']) > 5 else ''}")
        
        # 显示数据集详细信息
        with st.expander("📋 查看详细信息"):
            try:
                with open("data/data.yaml", 'r', encoding='utf-8') as f:
                    yaml_content = f.read()
                st.code(yaml_content, language='yaml')
            except:
                st.error("无法读取data.yaml文件")
                
    elif data_yaml_exists:
        st.warning("⚠️ 发现数据集文件但无配置信息")
    else:
        st.warning("⚠️ 未配置数据集")
    
    # 数据集配置选项
    st.markdown("### 🔧 配置新数据集")
    
    # 选择数据集来源
    dataset_source = st.radio(
        "选择数据集来源:",
        ["📁 上传ZIP文件", "🌐 从URL下载"],
        horizontal=True
    )
    
    if dataset_source == "📁 上传ZIP文件":
        uploaded_file = st.file_uploader(
            "上传数据集ZIP文件",
            type=['zip'],
            help="请上传包含data.yaml配置文件的YOLO格式数据集"
        )
        
        if uploaded_file is not None:
            st.write(f"文件名: {uploaded_file.name}")
            st.write(f"文件大小: {uploaded_file.size / (1024*1024):.1f} MB")
            
            if st.button("🚀 处理上传的数据集", type="primary", key="process_uploaded_dataset_btn"):
                with st.spinner("处理中..."):
                    success, message = process_uploaded_dataset(uploaded_file)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
    
    else:  # URL下载
        dataset_url = st.text_input(
            "输入数据集下载URL:",
            placeholder="https://example.com/dataset.zip",
            help="请提供直接下载链接，文件应为包含data.yaml的ZIP格式"
        )
        
        if dataset_url:
            if st.button("🚀 下载并处理数据集", type="primary", key="process_url_dataset_btn"):
                with st.spinner("下载并处理中..."):
                    success, message = process_url_dataset(dataset_url)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

def extract_conversion_info(output_content):
    """提取转换关键信息"""
    lines = output_content.split('\n')
    info = {
        "current_step": None,
        "progress_percentage": 0,
        "latest_status": None,
        "conversion_name": None,
        "model_path": None,
        "images_collected": None,
        "images_copied": None,
        "onnx_conversion_status": None,
        "cvimodel_conversion_status": None,
        "mud_file_created": None,
        "zip_package_created": None
    }
    
    # 转换步骤映射
    steps_map = {
        "开始模型转换流程": (1, 10),
        "查找数据集映射关系": (2, 20),
        "数据集图片收集与复制": (3, 30),
        "ONNX模型转换": (4, 40),
        "ONNX转换成功": (5, 60),
        "复制转换脚本": (6, 70),
        "执行CviModel转换": (7, 80),
        "处理CviModel文件": (8, 90),
        "完整的MaixCam模型包已创建": (9, 100)
    }
    
    # 从日志中提取信息
    for line in lines:
        line = line.strip()
        
        # 提取转换名称
        if "开始模型转换流程" in line and "export_" in line:
            try:
                parts = line.split("export_")
                if len(parts) > 1:
                    info["conversion_name"] = "export_" + parts[1].split()[0]
            except:
                pass
        
        # 提取模型路径
        if "模型文件:" in line:
            try:
                info["model_path"] = line.split("模型文件:")[1].strip()
            except:
                pass
        
        # 检查步骤进度
        for step_text, (step_num, progress) in steps_map.items():
            if step_text in line:
                info["current_step"] = step_text
                info["progress_percentage"] = progress
                break
        
        # 提取图片收集信息
        if "找到" in line and "张图片" in line:
            try:
                import re
                numbers = re.findall(r'\d+', line)
                if numbers:
                    info["images_collected"] = int(numbers[0])
            except:
                pass
        
        # 提取图片复制信息
        if "成功复制" in line and "张图片" in line:
            try:
                import re
                numbers = re.findall(r'\d+', line)
                if numbers:
                    info["images_copied"] = int(numbers[0])
            except:
                pass
        
        # ONNX转换状态
        if "ONNX转换成功" in line:
            info["onnx_conversion_status"] = "成功"
        elif "ONNX转换失败" in line:
            info["onnx_conversion_status"] = "失败"
        
        # CviModel转换状态
        if "CviModel转换完成" in line:
            info["cvimodel_conversion_status"] = "成功"
        elif "CviModel转换失败" in line:
            info["cvimodel_conversion_status"] = "失败"
        
        # MUD文件创建状态
        if "成功创建MUD配置文件" in line:
            info["mud_file_created"] = "成功"
        elif "创建MUD文件失败" in line:
            info["mud_file_created"] = "失败"
        
        # ZIP包创建状态
        if "成功创建模型包" in line:
            info["zip_package_created"] = "成功"
        elif "创建模型包失败" in line:
            info["zip_package_created"] = "失败"
    
    # 提取最新状态
    for line in reversed(lines[-10:]):
        line = line.strip()
        if line and not line.startswith("="):
            info["latest_status"] = line
            break
    
    return info

def model_conversion_section():
    """模型转换部分 - 优化版本"""
    st.subheader("🔄 转换pt为MaixCam模型")
    
    # 获取当前状态
    status = get_status()
    current_status = status.get("status")
    
    # 显示状态
    status_icons = {
        "idle": "⚪ 待机中",
        "running": "🟢 训练中...",
        "converting": "🔄 转换中...",
        "completed": "✅ 完成",
        "failed": "❌ 失败",
        "stopped": "⏹️ 已停止"
    }
    
    status_text = status_icons.get(current_status, current_status)
    st.write(f"**当前状态:** {status_text}")
    
    # 查找可用的模型
    available_models = find_training_models()
    
    if not available_models:
        st.warning("⚠️ 未找到训练好的模型。请先完成模型训练。")
    else:
        st.success(f"✅ 发现 {len(available_models)} 个可用模型")
        
        # 创建模型选择下拉框
        model_options = [f"{model['name']} ({model['size']:.1f} MB, {model['time'].strftime('%Y-%m-%d %H:%M')})" for model in available_models]
        selected_model_idx = st.selectbox(
            "选择要转换的模型:",
            range(len(model_options)),
            format_func=lambda i: model_options[i],
            help="选择best.pt获得更好的精度，或选择last.pt获得最新的训练结果"
        )
        
        selected_model = available_models[selected_model_idx]
        st.info(f"已选择: **{selected_model['name']}**")
        st.info(f"模型路径: `{selected_model['path']}`")
        st.info(f"绝对路径: `{os.path.abspath(selected_model['path'])}`")
        
        # 显示映射关系信息
        mapping_info = get_pt_dataset_mapping(selected_model["path"])
        if mapping_info:
            st.success("✅ 找到数据集映射关系")
            with st.expander("📋 查看映射信息"):
                st.json(mapping_info)
        else:
            st.warning("⚠️ 未找到数据集映射关系，将跳过图片复制步骤")
            
            # 调试信息
            with st.expander("🔍 调试映射关系"):
                st.write("**查找的路径:**")
                st.code(selected_model["path"])
                st.write("**绝对路径:**")
                st.code(os.path.abspath(selected_model["path"]))
                
                if os.path.exists(MAPPING_FILE):
                    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
                        all_mappings = json.load(f)
                        st.write("**映射文件中的所有路径:**")
                        for key in all_mappings.keys():
                            st.code(key)
                else:
                    st.error("映射文件不存在")
        
        # 检查convert_cvimodel.sh文件是否存在
        convert_script_exists = os.path.exists("convert_cvimodel.sh")
        if convert_script_exists:
            st.success("✅ 找到转换脚本: convert_cvimodel.sh")
        else:
            st.warning("⚠️ 未找到转换脚本: convert_cvimodel.sh")
            st.info("请确保convert_cvimodel.sh文件存在于应用根目录，否则将跳过CviModel转换步骤")
        
        # 显示数据集标签预览
        labels = get_dataset_labels()
        if labels:
            st.success(f"✅ 检测到数据集标签: {', '.join(labels[:5])}{'...' if len(labels) > 5 else ''}")
        else:
            st.warning("⚠️ 未找到数据集标签，MUD文件将使用默认标签")
        
        # ONNX相关参数设置
        st.markdown("### ⚙️ ONNX转换参数")
        
        # ONNX Opset版本（固定）
        opset_version = 18  # 固定值
        st.info(f"**ONNX Opset版本:** {opset_version} (固定参数，专为MaixCam优化)")
        
        # 显示高级参数
        with st.expander("高级参数设置"):
            st.markdown("ONNX转换的高级参数")
            
            # 这些参数暂时不会实际使用，但保留UI元素供未来扩展
            st.markdown("以下参数当前固定:")
            st.code("""
batch=1                  # 批次大小固定为1，适合设备推理
include=['onnx']         # 仅导出ONNX格式
half=True                # 使用FP16半精度
int8=False               # 不使用INT8量化
device=0                 # 使用第一个GPU设备
""", language="bash")
        
        # 转换按钮
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if current_status not in ["converting", "running"]:
                if st.button("🚀 开始转换", type="primary", key="start_conversion_btn"):
                    # 执行转换
                    run_model_conversion(
                        model_path=selected_model["path"], 
                        format="onnx", 
                        opset=opset_version
                    )
                    st.success("模型转换已开始!")
                    st.rerun()
            else:
                st.button("🚀 开始转换", disabled=True, key="start_conversion_btn_disabled")
        
        with col2:
            if current_status == "converting":
                if st.button("⏹️ 停止转换", type="secondary", key="stop_conversion_btn"):
                    stop_conversion()
                    st.rerun()
            else:
                st.button("⏹️ 停止转换", disabled=True, key="stop_conversion_btn_disabled")
        
        with col3:
            if st.button("🔄 刷新状态", key="refresh_conversion_status_btn"):
                st.rerun()
        
        # ===== 优化后的转换输出日志显示部分 =====
        st.markdown("### 📄 转换输出日志")
        conversion_output = read_conversion_output()
        
        if conversion_output:
            # 提取转换关键信息
            conversion_info = extract_conversion_info(conversion_output)
            
            # 显示转换进度摘要
            if conversion_info["current_step"]:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🔄 当前步骤", conversion_info["current_step"])
                with col2:
                    st.metric("📈 转换进度", f"{conversion_info['progress_percentage']:.0f}%")
                with col3:
                    if conversion_info["conversion_name"]:
                        st.metric("📦 转换任务", conversion_info["conversion_name"])
                
                # 显示进度条
                progress_bar = st.progress(conversion_info['progress_percentage'] / 100)
            
            # 显示转换状态摘要
            if any([conversion_info["images_collected"], conversion_info["onnx_conversion_status"], 
                   conversion_info["cvimodel_conversion_status"], conversion_info["mud_file_created"]]):
                
                st.markdown("**🎯 转换状态摘要:**")
                
                status_cols = st.columns(4)
                
                with status_cols[0]:
                    if conversion_info["images_collected"]:
                        st.info(f"📸 图片收集: {conversion_info['images_collected']}张")
                    if conversion_info["images_copied"]:
                        st.info(f"📋 图片复制: {conversion_info['images_copied']}张")
                
                with status_cols[1]:
                    if conversion_info["onnx_conversion_status"]:
                        if conversion_info["onnx_conversion_status"] == "成功":
                            st.success(f"🔄 ONNX: {conversion_info['onnx_conversion_status']}")
                        else:
                            st.error(f"🔄 ONNX: {conversion_info['onnx_conversion_status']}")
                
                with status_cols[2]:
                    if conversion_info["cvimodel_conversion_status"]:
                        if conversion_info["cvimodel_conversion_status"] == "成功":
                            st.success(f"🎯 CviModel: {conversion_info['cvimodel_conversion_status']}")
                        else:
                            st.error(f"🎯 CviModel: {conversion_info['cvimodel_conversion_status']}")
                
                with status_cols[3]:
                    if conversion_info["mud_file_created"]:
                        if conversion_info["mud_file_created"] == "成功":
                            st.success(f"📋 MUD: {conversion_info['mud_file_created']}")
                        else:
                            st.error(f"📋 MUD: {conversion_info['mud_file_created']}")
                    if conversion_info["zip_package_created"]:
                        if conversion_info["zip_package_created"] == "成功":
                            st.success(f"📦 ZIP: {conversion_info['zip_package_created']}")
                        else:
                            st.error(f"📦 ZIP: {conversion_info['zip_package_created']}")
            
            # 显示最新的几行日志（置顶显示）
            st.markdown("**🔥 最新日志:**")
            lines = conversion_output.split('\n')
            
            # 过滤掉空行和分隔线，取最后10行有内容的日志
            non_empty_lines = [line for line in lines if line.strip() and not line.strip().startswith('=') and not line.strip().startswith('-')]
            recent_lines = non_empty_lines[-10:] if len(non_empty_lines) >= 10 else non_empty_lines
            
            # 反转显示顺序，最新的在上面
            recent_lines_reversed = list(reversed(recent_lines))
            recent_content = '\n'.join(recent_lines_reversed)
            
            # 使用代码块显示最新日志
            log_container = st.container()
            with log_container:
                st.code(recent_content, language=None)
            
            # 显示选项
            col1, col2 = st.columns(2)
            with col1:
                show_full_conversion_log = st.checkbox("显示完整转换日志", value=False, key="show_full_conversion_logs")
            with col2:
                auto_refresh_conversion = st.checkbox("自动刷新", value=True, key="auto_refresh_conversion_logs")
            
            # 显示完整日志（可选）
            if show_full_conversion_log:
                st.markdown("**📋 完整转换日志:**")
                st.text_area(
                    "转换日志:",
                    value=conversion_output,
                    height=400,
                    key="conversion_output_area"
                )
            
            # 显示日志统计
            total_lines = len([line for line in lines if line.strip()])
            st.caption(f"📊 总计 {total_lines} 行有效日志 | 🕒 最后更新: {datetime.now().strftime('%H:%M:%S')}")
            
            # 如果正在转换，自动刷新
            if current_status == "converting" and auto_refresh_conversion:
                time.sleep(2)  # 每2秒刷新一次
                st.rerun()
                
        else:
            st.info("暂无转换日志")
        
        # ===== 新增：显示转换结果和下载功能 =====
        st.markdown("### 📦 转换结果和下载")
        
        # 查找已转换的模型包
        converted_packages = find_model_packages()
        converted_cvimodels = find_converted_cvimodels()
        
        if converted_packages:
            st.success(f"✅ 发现 {len(converted_packages)} 个完整模型包")
            
            # 显示模型包列表
            for i, package in enumerate(converted_packages):
                with st.expander(f"📦 {package['name']} ({package['size']:.2f} MB) - {package['time'].strftime('%Y-%m-%d %H:%M:%S')}", expanded=(i==0)):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.info(f"**文件路径:** `{package['path']}`")
                        st.info(f"**转换任务:** {package['export_dir']}")
                        st.info(f"**文件大小:** {package['size']:.2f} MB")
                        st.info(f"**创建时间:** {package['time'].strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    with col2:
                        # 使用Streamlit的download_button提供下载功能
                        try:
                            with open(package['path'], "rb") as file:
                                file_bytes = file.read()
                            
                            st.download_button(
                                label="📥 下载完整模型包",
                                data=file_bytes,
                                file_name=package['name'],
                                mime="application/zip",
                                key=f"download_package_{i}",
                                type="primary"
                            )
                        except Exception as e:
                            st.error(f"准备下载失败: {str(e)}")
                    
                    with col3:
                        # 显示包含内容预览
                        if st.button(f"🔍 查看内容", key=f"show_content_{i}"):
                            try:
                                import zipfile
                                with zipfile.ZipFile(package['path'], 'r') as zip_ref:
                                    file_list = zip_ref.namelist()
                                    st.markdown("**ZIP包内容:**")
                                    for file in file_list:
                                        st.text(f"📄 {file}")
                            except Exception as e:
                                st.error(f"读取ZIP内容失败: {str(e)}")
        
        elif converted_cvimodels:
            st.warning("⚠️ 发现CviModel文件但无完整模型包")
            
            # 显示CviModel文件列表
            for i, cvimodel in enumerate(converted_cvimodels):
                with st.expander(f"🎯 {cvimodel['name']} ({cvimodel['size']:.2f} MB) - {cvimodel['time'].strftime('%Y-%m-%d %H:%M:%S')}", expanded=(i==0)):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.info(f"**文件路径:** `{cvimodel['path']}`")
                        st.info(f"**转换任务:** {cvimodel['export_dir']}")
                        st.info(f"**文件大小:** {cvimodel['size']:.2f} MB")
                        st.info(f"**创建时间:** {cvimodel['time'].strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    with col2:
                        # 下载单独的CviModel文件
                        try:
                            with open(cvimodel['path'], "rb") as file:
                                file_bytes = file.read()
                            
                            st.download_button(
                                label="📥 下载CviModel",
                                data=file_bytes,
                                file_name=cvimodel['name'],
                                mime="application/octet-stream",
                                key=f"download_cvimodel_{i}",
                                type="secondary"
                            )
                        except Exception as e:
                            st.error(f"准备下载失败: {str(e)}")
                    
                    # 检查是否有对应的MUD文件
                    mud_file_path = cvimodel['path'].replace('.cvimodel', '.mud')
                    if os.path.exists(mud_file_path):
                        try:
                            with open(mud_file_path, "rb") as file:
                                mud_bytes = file.read()
                            
                            st.download_button(
                                label="📋 下载MUD配置",
                                data=mud_bytes,
                                file_name=os.path.basename(mud_file_path),
                                mime="text/plain",
                                key=f"download_mud_{i}",
                                type="secondary"
                            )
                        except Exception as e:
                            st.error(f"准备MUD下载失败: {str(e)}")
        
        else:
            st.info("💡 暂无转换完成的模型包。完成模型转换后，下载按钮将在此处显示。")
            
            # 显示transfer目录的所有内容（用于调试）
            transfer_dir = "transfer"
            if os.path.exists(transfer_dir):
                with st.expander("🔍 调试：查看transfer目录内容"):
                    for item in os.listdir(transfer_dir):
                        item_path = os.path.join(transfer_dir, item)
                        if os.path.isdir(item_path):
                            st.write(f"📁 {item}/")
                            # 显示子目录内容
                            for subitem in os.listdir(item_path):
                                subitem_path = os.path.join(item_path, subitem)
                                if os.path.isfile(subitem_path):
                                    size_mb = os.path.getsize(subitem_path) / (1024 * 1024)
                                    st.write(f"   📄 {subitem} ({size_mb:.2f} MB)")
                                else:
                                    st.write(f"   📁 {subitem}/")
            else:
                st.info("transfer目录不存在")

def main():
    """主应用程序"""
    st.set_page_config(
        page_title="MaixCam的YOLOv11训练平台",
        page_icon="🧪",
        layout="wide"
    )
    
    st.title("🧪 MaixCam的YOLOv11训练平台")
    st.markdown("支持数据集上传/下载、参数设置、模型转换和MaixCam CviModel生成的增强版训练平台")
    
    # 显示平台信息
    platform_info = get_platform_info()
    st.sidebar.markdown(f"**系统信息:**")
    st.sidebar.info(f"操作系统: {platform_info['system']}")
    st.sidebar.info(f"架构: {platform_info['machine']}")
    
    # 初始化
    init_status()
    current_status = get_status()
    
    # 主要内容区域
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📦 数据集管理", "🚀 训练控制", "📺 实时输出", "📊 训练结果", "📤 转换pt为MaixCam模型"])
    
    with tab1:
        dataset_management_section()
    
    with tab2:
        st.subheader("🚀 训练控制")
        
        # 状态显示
        status_icons = {
            "idle": "⚪ 待机中",
            "running": "🟢 训练中...",
            "converting": "🔄 转换中...",
            "completed": "✅ 训练完成",
            "failed": "❌ 训练失败",
            "stopped": "⏹️ 已停止"
        }
        
        status_text = status_icons.get(current_status["status"], current_status["status"])
        st.write(f"**当前状态:** {status_text}")
        
        # 添加训练参数设置
        st.markdown("### ⚙️ 训练参数设置")
        
        # 模型选择
        # "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"，暂时只是支持yolo11n
        model_options = ["yolo11n.pt"]
        selected_model = st.selectbox(
            "选择模型:",
            model_options,
            index=0,
            help="选择YOLOv11模型版本，n(nano)最小，x(xlarge)最大"
        )
        
        # Epoch设置
        epochs = st.slider(
            "训练轮数 (Epochs):",
            min_value=5,
            max_value=300,
            value=20,
            step=5,
            help="训练循环的总轮数，更多的轮数可能获得更好的结果，但训练时间更长"
        )
        
        # 图片尺寸设置
        img_size_options = [320, 416, 512, 640, 768, 896, 1024, 1280]
        selected_img_size = st.select_slider(
            "图片尺寸 (Image Size):",
            options=img_size_options,
            value=640,
            help="训练图片尺寸，更大的尺寸可能提高准确率，但会增加显存需求和训练时间"
        )
        
        # 高级参数
        with st.expander("高级参数设置"):
            st.markdown("以下是当前固定的高级参数，将在未来版本中开放设置")
            st.code("""
batch=16                 # 批次大小
patience=50              # 早停耐心值
optimizer='auto'         # 优化器
lr0=0.01                 # 初始学习率
cos_lr=True              # 是否使用余弦学习率调度
weight_decay=0.0005      # 权重衰减
dropout=0.0              # 丢弃率
label_smoothing=0.0      # 标签平滑
""", language="bash")
        
        # 控制按钮
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if current_status["status"] in ["idle", "completed", "failed", "stopped"]:
                if st.button("🚀 开始训练", type="primary", key="start_training_btn"):
                    # 检查基本文件
                    if not os.path.exists("data/data.yaml"):
                        st.error("❌ 未找到 data/data.yaml 文件!")
                        st.info("请先在'数据集管理'标签页配置数据集")
                    else:
                        run_docker_training(selected_model, epochs, selected_img_size)
                        st.success("训练已开始!")
                        st.rerun()
            else:
                st.button("🚀 开始训练", disabled=True, key="start_training_btn_disabled")
        
        with col2:
            if current_status["status"] == "running":
                if st.button("⏹️ 停止训练", type="secondary", key="stop_training_btn"):
                    stop_training()
                    st.rerun()
            else:
                st.button("⏹️ 停止训练", disabled=True, key="stop_training_btn_disabled")
        
        with col3:
            if st.button("🔄 刷新状态", key="refresh_training_status_btn"):
                st.rerun()
        
        with col4:
            if st.button("🧹 清空日志", key="clear_logs_btn"):
                clear_output()
                st.success("日志已清空")
                st.rerun()
        
        # 显示Docker命令
        with st.expander("🔍 查看执行的Docker命令"):
            # 使用当前选择的参数生成命令预览
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"train_{timestamp}"
            
            docker_cmd, _, _, _ = build_docker_training_command(
                selected_model, epochs, selected_img_size, run_name
            )
            st.code(docker_cmd, language='bash')
    
    with tab3:
        st.subheader("📺 训练输出")
        
        output_content = read_output()
        if output_content:
            # 提取训练关键信息
            training_info = extract_training_info(output_content)
            
            # 显示训练进度摘要
            if training_info["current_epoch"]:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 当前Epoch", f"{training_info['current_epoch']}/{training_info['total_epochs']}")
                with col2:
                    st.metric("📈 训练进度", f"{training_info['progress_percentage']:.1f}%")
                with col3:
                    if training_info["latest_metrics"]:
                        # 简化显示最新指标
                        if "mAP50-95" in training_info["latest_metrics"]:
                            try:
                                map_value = training_info["latest_metrics"].split()[-1]
                                st.metric("🎯 mAP50-95", map_value)
                            except:
                                st.metric("🎯 最新指标", "计算中...")
                
                # 显示进度条
                progress_bar = st.progress(training_info['progress_percentage'] / 100)
            
            # 显示最新的几行日志（置顶显示）
            st.markdown("**🔥 最新日志:**")
            lines = output_content.split('\n')
            
            # 过滤掉空行，取最后10行有内容的日志
            non_empty_lines = [line for line in lines if line.strip()]
            recent_lines = non_empty_lines[-10:] if len(non_empty_lines) >= 10 else non_empty_lines
            
            # 反转显示顺序，最新的在上面
            recent_lines_reversed = list(reversed(recent_lines))
            recent_content = '\n'.join(recent_lines_reversed)
            
            # 使用不同的显示方式
            log_container = st.container()
            with log_container:
                st.code(recent_content, language=None)
            
            # 显示选项
            col1, col2 = st.columns(2)
            with col1:
                show_full_log = st.checkbox("显示完整日志", value=False, key="show_full_logs_checkbox")
            with col2:
                auto_scroll = st.checkbox("自动刷新", value=True, key="auto_refresh_logs_checkbox")
            
            # 显示完整日志（可选）
            if show_full_log:
                st.markdown("**📋 完整训练日志:**")
                st.text_area(
                    "所有日志内容:",
                    value=output_content,
                    height=300,
                    key="full_output_area"
                )
            
            # 显示日志统计
            total_lines = len([line for line in lines if line.strip()])
            st.caption(f"📊 总计 {total_lines} 行有效日志 | 🕒 最后更新: {datetime.now().strftime('%H:%M:%S')}")
            
        else:
            st.info("暂无输出内容（可以浏览器刷新一下）")
        
        # 如果正在运行，自动刷新（默认开启）
        if current_status["status"] == "running" and 'auto_scroll' in locals() and auto_scroll:
            time.sleep(2)  # 每2秒刷新一次
            st.rerun()
    
    with tab4:
        display_results()
    
    with tab5:
        model_conversion_section()


if __name__ == "__main__":
    # Windows系统编码设置
    if platform.system() == "Windows":
        # 设置控制台编码为UTF-8
        os.system('chcp 65001 >nul 2>&1')
        # 设置环境变量
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['CHCP'] = '65001'
    
    # 在应用启动时进行环境初始化
    print("正在启动MaixCam YOLOv11训练平台...")
    
    platform_info = get_platform_info()
    print(f"检测到操作系统: {platform_info['system']} ({platform_info['machine']})")
    
    # 在应用启动时进行环境初始化（在后台线程中运行，避免阻塞Streamlit启动）
    def background_env_check():
        """后台环境检查"""
        try:
            initialize_environment()
        except Exception as e:
            print(f"环境初始化失败: {e}")
            print("程序仍将启动，但某些功能可能无法正常工作")
    
    # 创建后台线程进行环境检查
    env_check_thread = threading.Thread(target=background_env_check)
    env_check_thread.daemon = True
    env_check_thread.start()
    
    print("🚀 启动Streamlit应用...")
    
    # 启动Streamlit应用
    main()