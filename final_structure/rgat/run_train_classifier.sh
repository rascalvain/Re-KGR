#!/bin/bash

# ============================================================================
# RGAT分类器训练脚本
# 功能：后台运行训练，输出写入日志文件
# 使用：bash run_train_classifier.sh
# ============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="train_classifier.py"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_classifier_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/train_classifier.pid"

# 创建日志目录
mkdir -p "${LOG_DIR}"

# ============================================================================
# 函数定义
# ============================================================================

print_header() {
    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${BLUE}  RGAT分类器训练 - 后台执行${NC}"
    echo -e "${BLUE}============================================================================${NC}"
}

check_python() {
    echo -e "${YELLOW}[1/5] 检查Python环境...${NC}"

    if ! command -v python &> /dev/null; then
        echo -e "${RED}❌ 错误: 未找到Python${NC}"
        exit 1
    fi

    PYTHON_VERSION=$(python --version 2>&1)
    echo -e "${GREEN}✓ Python版本: ${PYTHON_VERSION}${NC}"
}

check_script() {
    echo -e "${YELLOW}[2/5] 检查训练脚本...${NC}"

    if [ ! -f "${SCRIPT_DIR}/${PYTHON_SCRIPT}" ]; then
        echo -e "${RED}❌ 错误: 未找到 ${PYTHON_SCRIPT}${NC}"
        echo -e "${RED}   请确保脚本在当前目录: ${SCRIPT_DIR}${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ 训练脚本: ${PYTHON_SCRIPT}${NC}"
}

check_existing_process() {
    echo -e "${YELLOW}[3/5] 检查现有进程...${NC}"

    if [ -f "${PID_FILE}" ]; then
        OLD_PID=$(cat "${PID_FILE}")

        if ps -p "${OLD_PID}" > /dev/null 2>&1; then
            echo -e "${YELLOW}⚠️  发现正在运行的训练进程 (PID: ${OLD_PID})${NC}"
            echo -e "${YELLOW}   是否停止现有进程？ (y/n)${NC}"
            read -r response

            if [[ "$response" =~ ^[Yy]$ ]]; then
                echo -e "${YELLOW}   正在停止进程 ${OLD_PID}...${NC}"
                kill "${OLD_PID}"
                sleep 2

                if ps -p "${OLD_PID}" > /dev/null 2>&1; then
                    echo -e "${RED}   强制停止进程...${NC}"
                    kill -9 "${OLD_PID}"
                fi

                echo -e "${GREEN}✓ 进程已停止${NC}"
            else
                echo -e "${RED}   取消启动新进程${NC}"
                exit 1
            fi
        else
            echo -e "${YELLOW}   清理过期的PID文件${NC}"
            rm -f "${PID_FILE}"
        fi
    fi

    echo -e "${GREEN}✓ 无冲突进程${NC}"
}

start_training() {
    echo -e "${YELLOW}[4/5] 启动训练进程...${NC}"

    # 切换到脚本目录
    cd "${SCRIPT_DIR}" || exit 1

    # 使用nohup后台运行，输出重定向到日志文件
    nohup python -u "${PYTHON_SCRIPT}" > "${LOG_FILE}" 2>&1 &

    # 获取进程ID
    TRAIN_PID=$!

    # 保存PID到文件
    echo "${TRAIN_PID}" > "${PID_FILE}"

    # 等待一下确保进程启动
    sleep 2

    # 验证进程是否还在运行
    if ps -p "${TRAIN_PID}" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ 训练进程已启动${NC}"
        echo -e "${GREEN}  进程ID: ${TRAIN_PID}${NC}"
        echo -e "${GREEN}  日志文件: ${LOG_FILE}${NC}"
    else
        echo -e "${RED}❌ 进程启动失败${NC}"
        echo -e "${RED}   查看日志: ${LOG_FILE}${NC}"
        exit 1
    fi
}

show_info() {
    echo -e "${YELLOW}[5/5] 训练信息${NC}"
    echo ""
    echo -e "${BLUE}训练已在后台运行！${NC}"
    echo ""
    echo -e "${GREEN}进程信息:${NC}"
    echo -e "  PID: ${TRAIN_PID}"
    echo -e "  PID文件: ${PID_FILE}"
    echo ""
    echo -e "${GREEN}日志信息:${NC}"
    echo -e "  日志文件: ${LOG_FILE}"
    echo -e "  日志目录: ${LOG_DIR}"
    echo ""
    echo -e "${GREEN}有用的命令:${NC}"
    echo -e "  查看日志: tail -f ${LOG_FILE}"
    echo -e "  查看进程: ps -p ${TRAIN_PID}"
    echo -e "  停止训练: kill ${TRAIN_PID}"
    echo -e "  查看GPU: nvidia-smi"
    echo ""
    echo -e "${BLUE}============================================================================${NC}"
    echo ""

    # 显示前几行日志
    echo -e "${YELLOW}日志预览（前20行）:${NC}"
    echo -e "${BLUE}----------------------------------------------------------------------------${NC}"
    head -n 20 "${LOG_FILE}" 2>/dev/null || echo "日志还未生成..."
    echo -e "${BLUE}----------------------------------------------------------------------------${NC}"
    echo ""
    echo -e "${GREEN}继续查看日志: tail -f ${LOG_FILE}${NC}"
}

# ============================================================================
# 主流程
# ============================================================================

main() {
    print_header
    check_python
    check_script
    check_existing_process
    start_training
    show_info
}

# 执行主流程
main