事件相机模拟器 - 详细使用指南
📚 目录
1. 安装与设置
2. 快速开始
3. 命令行使用
4. 制作演示视频
5. 参数调优指南
6. 常见问题

---
安装与设置
1. 克隆/下载代码
# 确保所有文件在同一目录
event_simulator/
├── pixel.py
├── sensor.py
├── visualization.py
├── main.py
├── demo.py
├── test_simulator.py
├── requirements.txt
└── README.md
2. 安装依赖
pip install -r requirements.txt
或手动安装：
pip install numpy scipy opencv-python tqdm matplotlib
3. 验证安装
# 运行快速测试
python test_simulator.py

# 应该生成 test_videos/ 目录和测试结果

---
快速开始
最简单的工作流程
# 步骤1: 准备输入视频
# 将你的视频文件放在工作目录（例如: my_video.mp4）

# 步骤2: 运行模拟器并生成可视化
python main.py --input my_video.mp4 --visualize

# 步骤3: 查看结果
# 输出在 output/sim_YYYYMMDD_HHMMSS/ 目录
运行预设演示
# 快速演示（30秒内完成）
python demo.py quick

# 完整演示（生成所有场景，约5分钟）
python demo.py

---
命令行使用
基础命令
python main.py --input VIDEO_PATH [选项]
常用场景
场景1: 只生成事件数据（不可视化）
python main.py --input video.mp4 --output my_results
场景2: 生成事件 + 可视化视频
python main.py --input video.mp4 --visualize
场景3: 三栏对比视频
python main.py --input video.mp4 --visualize --comparison
场景4: 自定义参数
python main.py \
    --input video.mp4 \
    --threshold 0.15 \
    --time-resolution 0.0005 \
    --accumulation-time 0.05 \
    --visualize \
    --viz-style green_red
场景5: 高性能模式（大视频）
python main.py \
    --input large_video.mp4 \
    --multiprocessing \
    --workers 8 \
    --max-frames 300

---
制作演示视频
方法1: 使用演示脚本（推荐）
# 生成所有演示素材
python demo.py

# 输出目录: demo_output/
#   - scene1_*.mp4 (多种风格)
#   - scene2_visualization.mp4
#   - scene3_visualization.mp4
#   - scene1_comparison.mp4 (三栏对比)
#   - demo_report.txt (统计报告)
方法2: 使用自己的视频
# 准备高质量输入视频
# - 建议帧率: 30-60 FPS
# - 建议分辨率: 480p-720p
# - 内容: 清晰的运动物体

# 生成可视化
python main.py \
    --input your_video.mp4 \
    --visualize \
    --comparison \
    --viz-style red_blue \
    --accumulation-time 0.033 \
    --output demo_output
演示视频结构建议
15分钟演示视频大纲：
1. 开场 (1分钟)
  - 自我介绍
  - 项目背景
  - 演示概览
2. 事件相机原理 (2-3分钟)
  - 什么是事件相机
  - 与传统相机的区别
  - 对数亮度变化检测原理
  - 展示公式: ΔL ≥ C → Event
3. 系统架构 (2分钟)
  - 三层设计：像素→阵列→可视化
  - 关键算法流程图
  - 代码组织结构
4. 功能演示 (6-7分钟)
  - 场景1: 简单运动（移动方块） 
    - 播放原始视频
    - 播放事件可视化
    - 播放三栏对比
  - 场景2: 复杂运动（旋转/多物体）
  - 不同可视化风格对比
  - 参数影响演示（阈值对比）
5. 技术亮点 (2分钟)
  - 多进程加速
  - 噪声模型
  - 可配置参数
  - 性能统计
6. 总结 (1分钟)
  - 实现的功能
  - 遇到的挑战
  - 未来改进方向
  - Q&A
录制工具推荐
- 屏幕录制: OBS Studio, QuickTime, Camtasia
- 视频编辑: DaVinci Resolve, iMovie, Premiere
- 幻灯片: PowerPoint, Keynote, Google Slides

---
参数调优指南
对比度阈值 (--threshold)
作用: 控制事件触发的灵敏度
暂时无法在飞书文档外展示此内容
# 低对比度视频
python main.py -i video.mp4 --threshold 0.15 --visualize

# 高对比度视频
python main.py -i video.mp4 --threshold 0.35 --visualize
累积时间 (--accumulation-time)
作用: 控制可视化中事件的显示时长
暂时无法在飞书文档外展示此内容
# 快速运动
python main.py -i video.mp4 --visualize --accumulation-time 0.02

# 慢速运动
python main.py -i video.mp4 --visualize --accumulation-time 0.08
可视化风格 (--viz-style)
# 经典风格（科研论文常用）
python main.py -i video.mp4 --visualize --viz-style red_blue

# 清新风格
python main.py -i video.mp4 --visualize --viz-style green_red

# 热图风格（密度可视化）
python main.py -i video.mp4 --visualize --viz-style heatmap
性能优化
# 小视频（<640x480）
python main.py -i small.mp4 --visualize

# 中等视频（720p）
python main.py -i medium.mp4 --visualize --multiprocessing

# 大视频（1080p+）
python main.py -i large.mp4 \
    --multiprocessing \
    --workers 8 \
    --max-frames 500 \
    --time-resolution 0.001

---
常见问题
Q1: 没有生成任何事件？
Q1: 没有生成任何事件？
可能原因:
- 对比度阈值太高
- 视频内容静止或变化太小
- 输入视频有问题
解决方案:
# 降低阈值
python main.py -i video.mp4 --threshold 0.1 --visualize

# 检查视频是否正确加载
python -c "import cv2; cap = cv2.VideoCapture('video.mp4'); print(cap.isOpened())"
Q2: 生成的事件太多，性能慢？
解决方案:
# 1. 限制处理帧数（测试用）
python main.py -i video.mp4 --max-frames 100 --visualize

# 2. 增大时间分辨率（降低精度换速度）
python main.py -i video.mp4 --time-resolution 0.001 --visualize

# 3. 使用多进程
python main.py -i video.mp4 --multiprocessing --workers 4 --visualize

# 4. 提高阈值（减少事件数）
python main.py -i video.mp4 --threshold 0.4 --visualize
Q3: 可视化视频看不清事件？
解决方案:
# 1. 增加累积时间
python main.py -i video.mp4 --visualize --accumulation-time 0.08

# 2. 换用更明显的颜色风格
python main.py -i video.mp4 --visualize --viz-style green_red

# 3. 使用对比视频模式
python main.py -i video.mp4 --visualize --comparison
Q4: 内存不足错误？
解决方案:
# 1. 限制帧数
python main.py -i video.mp4 --max-frames 200

# 2. 降低视频分辨率
# 使用 ffmpeg 预处理:
ffmpeg -i input.mp4 -vf scale=640:480 output_480p.mp4
python main.py -i output_480p.mp4 --visualize

# 3. 增大时间分辨率（减少插值点）
python main.py -i video.mp4 --time-resolution 0.002
Q5: 如何调试单个像素？
# 创建测试脚本 test_pixel.py
from pixel import PixelEventGenerator
import numpy as np

# 创建测试数据
timestamps = np.linspace(0, 1, 30)
pixel_values = (128 + 100 * np.sin(2*np.pi*timestamps)).astype(np.uint8)

# 测试
gen = PixelEventGenerator(contrast_threshold=0.2)
events = gen.process(pixel_values, timestamps, x=0, y=0)

print(f"生成事件数: {len(events)}")
for e in events[:5]:
    print(e)
Q6: 如何导出事件供其他工具使用？
# 导出为 NumPy 格式（Python友好）
python main.py -i video.mp4 --format npy

# 加载事件数据:
import numpy as np
events = np.load('output/sim_*/events.npy')
print(events['t'])  # 时间戳
print(events['x'])  # X坐标
print(events['y'])  # Y坐标
print(events['polarity'])  # 极性
# 导出为文本格式（通用）
python main.py -i video.mp4 --format txt

# 可用任何工具读取 events.txt
Q7: 如何比较不同参数的效果？
# 创建批处理脚本 batch_test.sh
for threshold in 0.1 0.2 0.3; do
    python main.py \
        --input test.mp4 \
        --threshold $threshold \
        --visualize \
        --output results_${threshold}
done

---
高级用法
在 Python 代码中使用
from sensor import EventCameraSensor
from visualization import EventVisualizer
import numpy as np
import cv2

# 加载视频
cap = cv2.VideoCapture('video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(gray)
cap.release()
frames = np.array(frames)

# 创建传感器
sensor = EventCameraSensor(
    contrast_threshold=0.2,
    timestamp_resolution=0.001
)

# 生成事件
events = sensor.simulate(frames, fps)

# 可视化
height, width = frames[0].shape
viz = EventVisualizer(width, height, style='red_blue')
viz.create_video(events, frames, fps, 'output.mp4')

# 分析事件
on_events = [e for e in events if e['polarity'] == 1]
off_events = [e for e in events if e['polarity'] == -1]
print(f"ON: {len(on_events)}, OFF: {len(off_events)}")
自定义可视化
from visualization import EventVisualizer
import numpy as np

class CustomVisualizer(EventVisualizer):
    def _render_overlay(self, events, event_layer):
        # 自定义渲染逻辑
        for event in events:
            x, y = event['x'], event['y']
            # 绘制圆圈而不是点
            cv2.circle(event_layer, (x, y), 2, (0, 255, 0), -1)
        return event_layer

# 使用自定义可视化器
viz = CustomVisualizer(width, height)
viz.create_video(events, frames, fps, 'custom.mp4')
事件数据分析
import numpy as np
import matplotlib.pyplot as plt

# 加载事件
events = np.load('events.npy')

# 时间分布
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.hist(events['t'], bins=50)
plt.xlabel('时间 (s)')
plt.ylabel('事件数')
plt.title('时间分布')

# 空间分布
plt.subplot(132)
plt.scatter(events['x'], events['y'], s=1, alpha=0.3)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('空间分布')

# 极性分布
plt.subplot(133)
polarity_counts = np.bincount(events['polarity'] + 1)
plt.bar(['OFF', 'ON'], polarity_counts)
plt.ylabel('事件数')
plt.title('极性分布')

plt.tight_layout()
plt.savefig('event_analysis.png')

---
性能基准
测试环境
- CPU: Intel i7 (8核)
- RAM: 16GB
- Python: 3.8+
性能数据
暂时无法在飞书文档外展示此内容
提示:
- 小视频（<VGA）用单进程足够快
- 中等视频（VGA-720p）推荐多进程
- 大视频（1080p+）建议降采样或限制帧数

---
幻灯片内容建议
幻灯片1: 标题页
事件相机模拟器
Event Camera Simulator

姓名 | 学号
日期
幻灯片2: 项目背景
- 什么是事件相机？
- 应用场景
- 项目目标
幻灯片3: 工作原理
- 对数亮度变化检测
- 公式: ΔL = log(I_new) - log(I_ref) ≥ C
- 示意图
幻灯片4: 系统架构
输入视频 → 像素处理 → 传感器阵列 → 事件流 → 可视化
幻灯片5-7: 演示场景
- 每个场景：原始视频 + 事件可视化
- 使用 demo.py 生成的视频
幻灯片8: 技术特性
- 多种可视化风格
- 参数可配置
- 多进程加速
- 噪声模型
幻灯片9: 实现细节
- 代码结构
- 关键算法
- 性能优化
幻灯片10: 总结与展望
- 完成功能
- 挑战与解决
- 未来改进

---
提交清单
必需文件
- [ ] 源代码（所有 .py 文件）
- [ ] requirements.txt
- [ ] README.md
- [ ] 演示视频（15分钟）
- [ ] 演示幻灯片（PDF）
- [ ] 贡献权重表
可选文件
- [ ] 示例输入视频
- [ ] 生成的事件数据
- [ ] 可视化视频样例
- [ ] 测试结果
打包建议
# 创建提交包
mkdir submission
cp *.py submission/
cp requirements.txt README.md submission/
cp presentation.pdf submission/
cp demo_video.mp4 submission/

# 压缩
zip -r event_simulator_submission.zip submission/

---
联系与支持
如有问题，请检查：
1. 本文档的常见问题部分
2. README.md
3. 代码注释

---
祝项目顺利！🎉