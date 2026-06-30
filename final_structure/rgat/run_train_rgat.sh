#!/bin/bash

# ============================================================================
# RGAT训练脚本
# 功能：后台运行RGAT训练，每次在独立的时间戳文件夹中保存输出
# 使用：bash run_train_rgat.sh
# ============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="train_rgat_hotpotqa.py"
LOG_DIR="${SCRIPT_DIR}/logs"

# 🔥 生成时间戳（用于创建独立的训练文件夹）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="run_${TIMESTAMP}"

LOG_FILE="${LOG_DIR}/train_rgat_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/train_rgat_${TIMESTAMP}.pid"

# 创建日志目录
mkdir -p "${LOG_DIR}"

# ============================================================================
# 函数定义
# ============================================================================

print_header() {
    echo -e "${MAGENTA}============================================================================${NC}"
    echo -e "${MAGENTA}  RGAT (关系图注意力网络) 训练 - 后台执行${NC}"
    echo -e "${MAGENTA}  训练运行名称: ${RUN_NAME}${NC}"
    echo -e "${MAGENTA}============================================================================${NC}"
}

check_python() {
    echo -e "${YELLOW}[1/6] 检查Python环境...${NC}"

    if ! command -v python &> /dev/null; then
        echo -e "${RED}❌ 错误: 未找到Python${NC}"
        exit 1
    fi

    PYTHON_VERSION=$(python --version 2>&1)
    echo -e "${GREEN}✓ Python版本: ${PYTHON_VERSION}${NC}"

    # 检查PyTorch
    if python -c "import torch" &> /dev/null; then
        TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
        echo -e "${GREEN}✓ PyTorch版本: ${TORCH_VERSION}${NC}"

        # 检查CUDA
        if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" &> /dev/null; then
            CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
            GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
            echo -e "${GREEN}✓ CUDA版本: ${CUDA_VERSION}${NC}"
            echo -e "${GREEN}✓ 可用GPU数量: ${GPU_COUNT}${NC}"
        else
            echo -e "${YELLOW}⚠️  未检测到CUDA，将使用CPU训练${NC}"
        fi
    else
        echo -e "${RED}❌ 错误: 未安装PyTorch${NC}"
        exit 1
    fi
}

check_script() {
    echo -e "${YELLOW}[2/6] 检查训练脚本...${NC}"

    if [ ! -f "${SCRIPT_DIR}/${PYTHON_SCRIPT}" ]; then
        echo -e "${RED}❌ 错误: 未找到 ${PYTHON_SCRIPT}${NC}"
        echo -e "${RED}   请确保脚本在当前目录: ${SCRIPT_DIR}${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ 训练脚本: ${PYTHON_SCRIPT}${NC}"

    # 检查依赖文件
    REQUIRED_FILES=(
        "siamese_rgat_improved.py"
        "config_hotpotqa_rgat.py"
    )

    for file in "${REQUIRED_FILES[@]}"; do
        if [ -f "${SCRIPT_DIR}/${file}" ]; then
            echo -e "${GREEN}✓ 依赖文件: ${file}${NC}"
        else
            echo -e "${YELLOW}⚠️  未找到: ${file}${NC}"
        fi
    done
}

check_data_files() {
    echo -e "${YELLOW}[3/6] 检查数据和嵌入文件...${NC}"

    # 从配置文件读取路径
    DATA_DIR=$(python -c "from config_hotpotqa_rgat import Config; print(Config.DATA_DIR)" 2>/dev/null)

    if [ -z "$DATA_DIR" ]; then
        echo -e "${YELLOW}⚠️  无法读取数据路径配置${NC}"
    else
        echo -e "${CYAN}  数据目录: ${DATA_DIR}${NC}"

        # 检查关键文件（不影响执行）
        if [ -d "${DATA_DIR}/final_hybrid_embeddings" ]; then
            echo -e "${GREEN}✓ 嵌入目录存在${NC}"
        else
            echo -e "${YELLOW}⚠️  嵌入目录不存在，训练可能会失败${NC}"
        fi

        # 🔥 显示输出目录信息
        OUTPUT_BASE=$(python -c "from config_hotpotqa_rgat import Config; import os; print(os.path.join(Config.DATA_DIR, 'rgat_output'))" 2>/dev/null)
        if [ -n "$OUTPUT_BASE" ]; then
            echo -e "${CYAN}  输出基础目录: ${OUTPUT_BASE}${NC}"
            echo -e "${CYAN}  本次训练目录: ${OUTPUT_BASE}/${RUN_NAME}${NC}"
        fi
    fi
}

check_existing_process() {
    echo -e "${YELLOW}[4/6] 检查现有RGAT训练进程...${NC}"

    # 🔥 检查所有RGAT训练进程（不仅仅是当前的）
    ACTIVE_PIDS=$(find "${LOG_DIR}" -name "train_rgat_*.pid" -type f 2>/dev/null)

    if [ -n "$ACTIVE_PIDS" ]; then
        echo -e "${CYAN}  发现以下PID文件:${NC}"
        for pid_file in $ACTIVE_PIDS; do
            if [ -f "$pid_file" ]; then
                OLD_PID=$(cat "$pid_file")
                if ps -p "${OLD_PID}" > /dev/null 2>&1; then
                    echo -e "${YELLOW}    - PID ${OLD_PID}: 运行中 ($(basename $pid_file))${NC}"
                else
                    echo -e "${CYAN}    - PID ${OLD_PID}: 已结束 (清理 $(basename $pid_file))${NC}"
                    rm -f "$pid_file"
                fi
            fi
        done

        # 询问是否继续
        RUNNING_COUNT=$(find "${LOG_DIR}" -name "train_rgat_*.pid" -type f 2>/dev/null | wc -l)
        if [ "$RUNNING_COUNT" -gt 0 ]; then
            echo -e "${YELLOW}⚠️  发现 ${RUNNING_COUNT} 个正在运行的训练进程${NC}"
            echo -e "${YELLOW}   是否继续启动新的训练？ (y/n)${NC}"
            read -r response

            if [[ ! "$response" =~ ^[Yy]$ ]]; then
                echo -e "${RED}   取消启动新训练${NC}"
                exit 1
            fi
        fi
    fi

    echo -e "${GREEN}✓ 准备启动新训练${NC}"
}

start_training() {
    echo -e "${YELLOW}[5/6] 启动RGAT训练进程...${NC}"

    # 切换到脚本目录
    cd "${SCRIPT_DIR}" || exit 1

    # 🔥 使用nohup后台运行，传递时间戳参数
    # --run_name 参数会被Python脚本接收，用于创建独立的输出目录
    nohup python -u "${PYTHON_SCRIPT}" --run_name "${RUN_NAME}" > "${LOG_FILE}" 2>&1 &

    # 获取进程ID
    TRAIN_PID=$!

    # 保存PID到文件
    echo "${TRAIN_PID}" > "${PID_FILE}"

    # 等待一下确保进程启动
    sleep 3

    # 验证进程是否还在运行
    if ps -p "${TRAIN_PID}" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ RGAT训练进程已启动${NC}"
        echo -e "${GREEN}  进程ID: ${TRAIN_PID}${NC}"
        echo -e "${GREEN}  运行名称: ${RUN_NAME}${NC}"
        echo -e "${GREEN}  日志文件: ${LOG_FILE}${NC}"
    else
        echo -e "${RED}❌ 进程启动失败${NC}"
        echo -e "${RED}   查看日志: ${LOG_FILE}${NC}"
        exit 1
    fi
}

show_info() {
    echo -e "${YELLOW}[6/6] RGAT训练信息${NC}"
    echo ""
    echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║                    RGAT训练已在后台运行！                              ║${NC}"
    echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}📊 进程信息:${NC}"
    echo -e "  运行名称: ${RUN_NAME}"
    echo -e "  PID: ${TRAIN_PID}"
    echo -e "  PID文件: ${PID_FILE}"
    echo ""
    echo -e "${CYAN}📁 输出信息:${NC}"
    OUTPUT_DIR=$(python -c "from config_hotpotqa_rgat import Config; import os; print(os.path.join(Config.DATA_DIR, 'rgat_output', '${RUN_NAME}'))" 2>/dev/null)
    if [ -n "$OUTPUT_DIR" ]; then
        echo -e "  输出目录: ${OUTPUT_DIR}"
        echo -e "  模型检查点: ${OUTPUT_DIR}/checkpoints/"
        echo -e "  训练曲线: ${OUTPUT_DIR}/rgat_training_curves.png"
        echo -e "  训练历史: ${OUTPUT_DIR}/rgat_training_history.json"
    fi
    echo ""
    echo -e "${CYAN}📄 日志信息:${NC}"
    echo -e "  日志文件: ${LOG_FILE}"
    echo -e "  日志目录: ${LOG_DIR}"
    echo ""
    echo -e "${CYAN}🔧 有用的命令:${NC}"
    echo -e "  ${GREEN}查看日志:${NC} tail -f ${LOG_FILE}"
    echo -e "  ${GREEN}查看进程:${NC} ps -p ${TRAIN_PID}"
    echo -e "  ${GREEN}停止训练:${NC} kill ${TRAIN_PID}"
    echo -e "  ${GREEN}查看GPU:${NC}  nvidia-smi"
    echo -e "  ${GREEN}GPU监控:${NC}  watch -n 1 nvidia-smi"
    echo -e "  ${GREEN}查看所有训练:${NC} ls -lht ${LOG_DIR}"
    echo ""
    echo -e "${MAGENTA}============================================================================${NC}"
    echo ""

    # 显示前几行日志
    echo -e "${YELLOW}📝 日志预览（前30行）:${NC}"
    echo -e "${BLUE}----------------------------------------------------------------------------${NC}"
    head -n 30 "${LOG_FILE}" 2>/dev/null || echo "日志还未生成..."
    echo -e "${BLUE}----------------------------------------------------------------------------${NC}"
    echo ""
    echo -e "${GREEN}💡 提示: 继续查看实时日志，请运行:${NC}"
    echo -e "   ${CYAN}tail -f ${LOG_FILE}${NC}"
    echo ""
}

# ============================================================================
# 主流程
# ============================================================================

main() {
    print_header
    check_python
    check_script
    check_data_files
    check_existing_process
    start_training
    show_info
}

# 执行主流程
main