import json
import os
import re
import base64
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

from .asg_generator import getQwenClient

# 常量定义
BASE_DIR = os.path.normpath("src/static/data/md")  # 根目录
INFO_DIR = os.path.normpath("src/static/data/info")  # 存放 JSON 结果的目录

# 加载 PyTorch EfficientNet 训练好的 3 类分类模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(pretrained=False)

# 修改最后一层，适应 3 类（flowchart, non-flowchart, other）
num_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_features, 3)  # 3 类
model.load_state_dict(torch.load("flowchart_classifier.pth", map_location=device))
model.to(device)  # 确保模型移动到正确的设备
model.eval()

# 预处理图片
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def analyze_flowchart_with_vlm(image_path):
    """
    用 VLM 分析图片：确认是否是有意义的图表，并生成 caption。
    返回 (is_chart: bool, caption: str)
    """
    try:
        client = getQwenClient()
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        # 根据文件扩展名确定 MIME 类型
        ext = os.path.splitext(image_path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"

        vlm_model = os.environ.get("VLM_MODEL") or os.environ.get("MODEL")
        if not vlm_model:
            raise ValueError("VLM_MODEL or MODEL environment variable must be set")
        print(f"🔧 Using VLM model: {vlm_model}")

        response = client.chat.completions.create(
            model=vlm_model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"}
                    },
                    {
                        "type": "text",
                        "text": (
                            "Is this image a meaningful chart, diagram, flowchart, or architecture figure "
                            "from an academic paper? If YES, provide a concise one-sentence academic caption "
                            "describing what this chart illustrates. If NO (e.g. it's a logo, decorative image, "
                            "table, screenshot, or unreadable), just say NO.\n\n"
                            "Respond in EXACTLY this format:\n"
                            "IS_CHART: YES or NO\n"
                            "CAPTION: <one-sentence caption or empty>"
                        )
                    }
                ]
            }],
            max_tokens=1024,
            temperature=0.1,
        )

        result_text = response.choices[0].message.content
        if result_text is None:
            print(f"⚠️ VLM returned None for {os.path.basename(image_path)}, skipping")
            return False, ""
        result_text = result_text.strip()
        print(f"🤖 VLM response for {os.path.basename(image_path)}: {result_text}")

        # 解析响应
        is_chart = False
        caption = ""
        for line in result_text.splitlines():
            line = line.strip()
            if line.upper().startswith("IS_CHART:"):
                is_chart = "YES" in line.upper()
            elif line.upper().startswith("CAPTION:"):
                caption = line.split(":", 1)[1].strip()

        return is_chart, caption

    except Exception as e:
        print(f"⚠️ VLM analysis failed for {image_path}: {e}")
        # VLM 失败时保守处理：不插入该图片
        return False, ""

def _process_single_pdf_images(pdf_folder, pdf_folder_path):
    """
    处理单个 PDF 文件夹中的所有图片，返回 (pdf_folder, result) 或 None
    result 是 {"path": image_path, "caption": caption} 或 None
    """
    print(f"🔍 处理 PDF 文件夹: {pdf_folder}")

    candidates = []  # 收集该 PDF 的所有候选图片

    # 遍历所有 `xxx/auto/images` 目录
    for root, dirs, files in os.walk(pdf_folder_path):
        if "auto/images" in root.replace("\\", "/"):  # 兼容 Windows 和 Linux
            for filename in sorted(files):  # 按文件名排序
                if not filename.lower().endswith(".jpg"):  # 只处理 JPG
                    continue

                image_path = os.path.join(root, filename)

                try:
                    img = Image.open(image_path).convert("RGB")  # 打开图片并转换为 RGB

                    # 预处理图片并转换为张量
                    img_tensor = transform(img).unsqueeze(0).to(device)

                    # 运行分类模型
                    with torch.no_grad():
                        output = model(img_tensor)
                        predicted_class = torch.argmax(output).item()

                    # **确保 predicted_class == 2 表示 flowchart**
                    if predicted_class == 2:  # `2` 代表 Flowchart 类别
                        print(f"🔎 Classifier detected flowchart candidate: {image_path}")
                        candidates.append(image_path)
                except Exception as e:
                    print(f"⚠️ Error processing image {image_path}: {e}")
                    continue

    # 用 VLM 逐个验证候选图片（按文件名顺序）
    for image_path in candidates:
        is_chart, caption = analyze_flowchart_with_vlm(image_path)
        if is_chart and caption:  # 必须有 caption 才算有效
            print(f"✅ VLM confirmed flowchart: {image_path}")
            return (pdf_folder, {"path": image_path, "caption": caption})
        else:
            print(f"❌ VLM rejected as non-chart: {image_path}")

    return None  # 该 PDF 没有有效 flowchart


def detect_flowcharts(survey_id, max_workers=4):
    """
    在指定 survey_id 目录下查找 flowchart，并保存 JSON 结果
    使用多线程并行处理多个 PDF 文件夹
    """
    survey_path = os.path.join(BASE_DIR, survey_id)  # 该 survey_id 的目录
    if not os.path.exists(survey_path):
        print(f"❌ 目录 {survey_path} 不存在！")
        return

    flowchart_dict = {}  # 存储 flowchart 结果
    pdf_folders = []  # 收集所有 PDF 文件夹

    # 收集所有 PDF 文件夹
    for pdf_folder in os.listdir(survey_path):
        pdf_folder_path = os.path.join(survey_path, pdf_folder)
        if os.path.isdir(pdf_folder_path):
            pdf_folders.append((pdf_folder, pdf_folder_path))

    if not pdf_folders:
        print(f"⚠️ 没有找到 PDF 文件夹")
        return

    print(f"🚀 开始并行处理 {len(pdf_folders)} 个 PDF 文件夹 (max_workers={max_workers})")

    # 并行处理所有 PDF 文件夹
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdf = {
            executor.submit(_process_single_pdf_images, pdf_folder, pdf_folder_path): pdf_folder
            for pdf_folder, pdf_folder_path in pdf_folders
        }

        for future in as_completed(future_to_pdf):
            result = future.result()
            if result:
                pdf_folder, data = result
                flowchart_dict[pdf_folder] = data

    # 只有检测到 Flowchart 时才保存 JSON
    if flowchart_dict:
        os.makedirs(os.path.join(INFO_DIR, survey_id), exist_ok=True)  # 确保目录存在
        json_path = os.path.join(INFO_DIR, survey_id, "flowchart_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(flowchart_dict, f, indent=4, ensure_ascii=False)

        print(f"📁 Flowchart 结果已保存: {json_path} (共 {len(flowchart_dict)} 个)")
    else:
        print(f"⚠️ 没有检测到 Flowchart，未生成 JSON")

# 示例调用
# survey_id = "test"  # 例如 "test"
# detect_flowcharts(survey_id)

def insert_ref_images(json_path, ref_names, text):
    """
    参数:
      json_path: JSON 文件路径，其内容格式例如：
                 {
                   "Accelerating federated learning with data and model parallelism in edge computing":
                     "src/static/data/md/test/Accelerating federated learning with data and model parallelism in edge computing/auto/images/xxx.jpg",
                   ... 
                 }
      ref_names: 引用名称列表，其中第 1 个元素对应 [1]，第 2 个对应 [2]，以此类推。
      text: 包含类似 [1]、[2] 等引用的 Markdown 文本。

    返回:
      修改后的文本字符串。在每个引用标记首次出现行的下方插入对应的 HTML 代码块，
      格式如下：
      
      <div style="text-align:center">
          <img src="image_path" alt="the flow chart of [ref_name]" style="width:50%;"/>
      </div>
      <div style="text-align:center">
          Fig [ref_num]: The flow chart of [ref_name]
      </div>
      
      其中 [ref_num] 为引用编号（ref_names 中的 1-based index），[ref_name] 为引用名称。

    说明：
      1. JSON 中存储的路径已是目标路径，但可能混合了正斜杠和反斜杠。
      2. 代码将先拆分路径字符串，再利用 os.path.join 拼接生成当前系统的标准路径，
         最后转换为统一的正斜杠格式并进行 URL 编码，以适配所有系统。
    """
    # 加载 JSON 文件内容
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            img_mapping = json.load(f)
    except Exception as e:
        raise Exception(f"加载 JSON 文件出错: {e}")

    inserted_refs = {}  # 记录每个引用标记是否已插入图片
    lines = text.splitlines()
    new_lines = []
    # 匹配类似 [1]、[2] 的引用标记
    ref_pattern = re.compile(r'\[(\d+)\]')
    img_index = 2
    for line in lines:
        new_lines.append(line)
        matches = ref_pattern.findall(line)
        for ref_num_str in matches:
            try:
                ref_num = int(ref_num_str)
            except ValueError:
                continue

            # 仅在引用标记首次出现时插入 HTML 块
            if ref_num not in inserted_refs:
                inserted_refs[ref_num] = True

                if 1 <= ref_num <= len(ref_names):
                    ref_name = ref_names[ref_num - 1]
                    entry = img_mapping.get(ref_name, "")
                else:
                    ref_name = f"ref_{ref_num}"
                    entry = ""

                # 兼容新旧 JSON 格式
                if isinstance(entry, dict):
                    jpg_path = entry.get("path", "")
                    caption = entry.get("caption", "")
                elif isinstance(entry, str):
                    jpg_path = entry
                    caption = ""
                else:
                    jpg_path = ""
                    caption = ""

                if not caption:
                    caption = f"Chart from '{ref_name}'"

                if jpg_path:
                    # 将路径中可能混合的正斜杠和反斜杠拆分为多个部分
                    parts = re.split(r'[\\/]+', jpg_path)
                    # 使用 os.path.join 拼接成当前系统的规范路径
                    normalized_jpg_path = os.path.join(*parts)
                    # 转换为适用于 HTML 的路径格式（全部替换为正斜杠）
                    normalized_jpg_path = normalized_jpg_path.replace(os.sep, '/')
                    # 对路径进行 URL 编码（保留斜杠）
                    normalized_jpg_path_url = quote(normalized_jpg_path, safe="/")

                    html_block = (
                        f"<div style=\"text-align:center\">\n"
                        f"    <img src=\"{normalized_jpg_path_url}\" alt=\"{caption}\" style=\"width:60%;\"/>\n"
                        f"</div>\n"
                        f"<div style=\"text-align:center;font-size:smaller;\">\n"
                        f"    Fig {img_index}: {caption}\n"
                        f"</div>"
                    )
                    new_lines.append(html_block)
                    new_lines.append("")  # 增加一个空行分隔
                    img_index += 1

    return "\n".join(new_lines)

def insert_tex_images(json_path, ref_names, text):
    """
    将 Markdown 文本中出现的数字引用（例如 [1], \[1], \[1\]）替换为 LaTeX figure 环境。
    仅在每个引用编号第一次出现时插入对应图片，后续出现同编号不再重复插入。

    参数:
        json_path: JSON 文件路径，其内容格式例如：
            {
              "Accelerating federated learning with data and model parallelism in edge computing":
                "src/static/data/md/test/Accelerating federated learning with data and model parallelism in edge computing/auto/images/xxx.jpg",
              ...
            }
        ref_names: 引用名称列表。其中第 1 个元素对应 [1]，第 2 个对应 [2]，以此类推。
        text: 包含类似 [1]、\[1]、\[1\] 等形式的 Markdown 文本。

    返回:
        修改后的文本字符串。在每个引用标记首次出现行的下方插入对应的 LaTeX figure 环境：

        \begin{figure}[htbp]
          \centering
          \includegraphics[width=0.6\textwidth]{image_path}
          \caption{Fig 2: Chart from 'ref_name'}
        \end{figure}

    说明：
      1. JSON 中存储的路径可能含正反斜杠。
      2. 我们按系统拼接路径，再统一转为正斜杠并进行 URL 编码。
      3. figure 的计数从 1 开始（可根据需求调整）。
      4. 若某引用编号未在 JSON 中匹配到图片，则不插入 figure。
    """

    # 读取 JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            img_mapping = json.load(f)
    except Exception as e:
        raise Exception(f"加载 JSON 文件出错: {e}")

    # 用于记录某个编号是否已插入过
    inserted_refs = {}

    # 按行处理文本
    lines = text.splitlines()
    new_lines = []

    # --------------------------
    # 匹配 [1], \[1], \[1\] 等数字引用
    # --------------------------
    # 含义:
    #   (?:\\)?    -> 可选的反斜杠 0或1次
    #   \[         -> 文字 '[' (在正则中需转义)
    #   (\d+)      -> 捕获一个或多个数字
    #   (?:\\)?    -> 可选的反斜杠 0或1次
    #   \]         -> 文字 ']' (需转义)
    # 整体匹配可能出现以下形式:
    #   [1], \[1], \[1\], [12], \[12] 等
    ref_pattern = re.compile(r'(?:\\)?\[(\d+)(?:\\)?\]')

    # figure 计数
    figure_index = 1

    for line in lines:
        new_lines.append(line)  # 先把此行内容写入新文本

        # 查找本行中所有符合模式的引用
        matches = ref_pattern.findall(line)
        for ref_num_str in matches:
            try:
                ref_num = int(ref_num_str)
            except ValueError:
                continue

            # 若该引用编号尚未插入过图片，则执行插入
            if ref_num not in inserted_refs:
                inserted_refs[ref_num] = True

                # 判断这个编号是否在 ref_names 范围内
                if 1 <= ref_num <= len(ref_names):
                    ref_name = ref_names[ref_num - 1]
                    entry = img_mapping.get(ref_name, "")
                else:
                    ref_name = f"ref_{ref_num}"
                    entry = ""

                # 兼容新旧 JSON 格式
                if isinstance(entry, dict):
                    jpg_path = entry.get("path", "")
                    caption = entry.get("caption", "")
                elif isinstance(entry, str):
                    jpg_path = entry
                    caption = ""
                else:
                    jpg_path = ""
                    caption = ""

                if not caption:
                    caption = f"Chart from '{ref_name}'"

                if jpg_path:
                    # 规范化路径
                    parts = re.split(r'[\\/]+', jpg_path)
                    normalized_jpg_path = os.path.join(*parts)
                    normalized_jpg_path = normalized_jpg_path.replace(os.sep, '/')
                    # URL 编码（保留 '/')
                    # normalized_jpg_path_url = quote(normalized_jpg_path, safe="/")
                    normalized_jpg_path_url = normalized_jpg_path

                    # 构建 LaTeX figure 块
                    # 转义 caption 中的 LaTeX 特殊字符
                    tex_caption = caption.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')
                    tex_block = (
                        r"\begin{figure}[htbp]" "\n"
                        r"  \centering" "\n"
                        f"  \\includegraphics[width=0.5\\textwidth]{{{normalized_jpg_path_url}}}\n"
                        f"  \\caption{{{tex_caption}}}\n"
                        r"\end{figure}"
                    )

                    # 插到新文本中，再加个空行分隔
                    new_lines.append(tex_block)
                    new_lines.append("")
                    figure_index += 1

    return "\n".join(new_lines)


# 示例用法
if __name__ == "__main__":
    # Markdown 文件路径
    md_file_path = "src/static/data/info/test/survey_test_processed.md"
    # JSON 文件路径
    json_file_path = "src/static/data/info/test/flowchart_results.json"

    try:
        with open(md_file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"错误: Markdown 文件 {md_file_path} 未找到！")
        text = ""

    ref_names = [
        "An explainable federated learning and blockchain based secure credit modeling method",
        "Bafl a blockchain based asynchronous",
        "Biscotti a blockchain system for private and secure federated learning",
        "Blockdfl a blockchain based fully decentralized peer to peer",
        "Accelerating blockchain enabled federated learning with clustered clients",
        "A fast blockchain based federated learning framework with compressed communications"
    ]

    result = insert_ref_images(json_file_path, ref_names, text)
    print("修改后的文本为：\n")
    print(result)