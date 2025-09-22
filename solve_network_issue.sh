#!/bin/bash
# IndexTTS2 网络问题解决方案脚本

echo "=== IndexTTS2 网络问题解决方案 ==="
echo

# 函数：检测网络连接
check_network() {
    echo "检测网络连接..."
    if curl -s --connect-timeout 5 https://huggingface.co > /dev/null; then
        echo "✓ HuggingFace 可以访问"
        return 0
    else
        echo "✗ HuggingFace 无法访问"
        return 1
    fi
}

# 函数：设置HuggingFace镜像
setup_hf_mirror() {
    echo "设置HuggingFace镜像源..."
    export HF_ENDPOINT=https://hf-mirror.com
    echo "已设置 HF_ENDPOINT=https://hf-mirror.com"
    
    # 测试镜像连接
    if curl -s --connect-timeout 5 https://hf-mirror.com > /dev/null; then
        echo "✓ HuggingFace镜像可以访问"
        return 0
    else
        echo "✗ HuggingFace镜像无法访问"
        return 1
    fi
}

# 函数：设置网络超时
setup_network_timeout() {
    echo "设置网络超时参数..."
    export HF_HUB_DOWNLOAD_TIMEOUT=300  # 5分钟超时
    export TRANSFORMERS_TIMEOUT=300
    echo "已设置较长的网络超时时间"
}

# 函数：预下载模型
predownload_models() {
    echo "开始预下载模型..."
    python predownload_models.py
    return $?
}

# 函数：启动离线模式
setup_offline_mode() {
    echo "设置离线模式..."
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    echo "已启用离线模式"
    echo "注意：离线模式需要事先下载好所有模型"
}

# 函数：使用CPU模式
setup_cpu_mode() {
    echo "设置CPU模式（绕过GPU内存问题）..."
    export CUDA_VISIBLE_DEVICES=""
    echo "已禁用GPU，将使用CPU模式"
    echo "注意：CPU模式运行较慢"
}

# 主菜单
show_menu() {
    echo "请选择解决方案："
    echo "1. 使用HuggingFace镜像源"
    echo "2. 预下载所有模型"
    echo "3. 设置离线模式"
    echo "4. 增加网络超时时间"
    echo "5. 使用CPU模式（绕过GPU问题）"
    echo "6. 网络诊断"
    echo "7. 直接启动（跳过预下载）"
    echo "8. 退出"
    echo
}

# 主逻辑
main() {
    # 首先检测网络
    if ! check_network; then
        echo
        echo "检测到网络问题，建议使用以下解决方案："
        echo
    fi
    
    while true; do
        show_menu
        read -p "请输入选项 (1-8): " choice
        echo
        
        case $choice in
            1)
                setup_hf_mirror
                if [ $? -eq 0 ]; then
                    echo "镜像设置成功，可以尝试选项2预下载模型"
                fi
                echo
                ;;
            2)
                predownload_models
                if [ $? -eq 0 ]; then
                    echo "模型预下载成功！现在可以启动API服务器"
                    read -p "是否立即启动API服务器？(y/n): " start_api
                    if [ "$start_api" = "y" ] || [ "$start_api" = "Y" ]; then
                        echo "启动API服务器..."
                        ./start_api.sh
                        break
                    fi
                else
                    echo "模型预下载失败，请尝试其他解决方案"
                fi
                echo
                ;;
            3)
                setup_offline_mode
                echo "离线模式已启用，现在启动API服务器..."
                ./start_api.sh
                break
                ;;
            4)
                setup_network_timeout
                echo "网络超时已设置，现在可以尝试启动API服务器"
                read -p "是否立即启动API服务器？(y/n): " start_api
                if [ "$start_api" = "y" ] || [ "$start_api" = "Y" ]; then
                    ./start_api.sh
                    break
                fi
                echo
                ;;
            5)
                setup_cpu_mode
                echo "CPU模式已设置，现在启动API服务器..."
                ./start_api.sh
                break
                ;;
            6)
                echo "运行网络诊断..."
                python network_debug.py
                echo
                ;;
            7)
                echo "直接启动API服务器（可能会在网络下载时卡住）..."
                ./start_api.sh
                break
                ;;
            8)
                echo "退出"
                break
                ;;
            *)
                echo "无效选项，请重新选择"
                echo
                ;;
        esac
    done
}

# 显示使用说明
echo "这个脚本提供多种解决IndexTTS2网络卡顿问题的方案"
echo "根据你的网络环境选择合适的解决方案"
echo

# 检查必要文件
if [ ! -f "predownload_models.py" ]; then
    echo "警告: predownload_models.py 不存在"
fi

if [ ! -f "network_debug.py" ]; then
    echo "警告: network_debug.py 不存在"
fi

if [ ! -f "start_api.sh" ]; then
    echo "错误: start_api.sh 不存在"
    exit 1
fi

# 运行主逻辑
main 