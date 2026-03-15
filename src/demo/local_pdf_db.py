"""
本地 PDF 数据库模块
用于扫描本地 PDF 文件夹并提供类似 arXiv 的搜索接口
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Optional
import hashlib

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


def extract_pdf_metadata(pdf_path: str) -> Dict:
    """
    从 PDF 文件中提取元数据（标题、摘要等）

    Args:
        pdf_path: PDF 文件路径

    Returns:
        包含标题、摘要等信息的字典
    """
    metadata = {
        "title": "",
        "summary": "",
        "pdf_link": pdf_path,
        "arxiv_id": "",
        "source": "local"
    }

    # 1. 尝试从文件名提取标题
    filename = os.path.basename(pdf_path)
    name_without_ext = os.path.splitext(filename)[0]
    # 将下划线、连字符替换为空格，作为标题
    title_from_filename = name_without_ext.replace("_", " ").replace("-", " ").strip()
    metadata["title"] = title_from_filename

    # 2. 如果 PyMuPDF 可用，尝试从 PDF 元数据中提取
    if PYMUPDF_AVAILABLE:
        try:
            with fitz.open(pdf_path) as doc:
                # 获取 PDF 元数据
                pdf_metadata = doc.metadata
                if pdf_metadata.get("title"):
                    metadata["title"] = pdf_metadata["title"]

                # 尝试从第一页提取文本作为摘要
                if len(doc) > 0:
                    first_page = doc[0]
                    text = first_page.get_text()
                    # 取前 500 字符作为摘要预览
                    metadata["summary"] = text[:500].strip() if text else ""
        except Exception as e:
            print(f"Error extracting metadata from {pdf_path}: {e}")

    # 3. 生成唯一的本地 ID
    metadata["arxiv_id"] = f"local_{hashlib.md5(pdf_path.encode()).hexdigest()[:12]}"

    return metadata


def scan_local_pdf_database(root_folder: str) -> List[Dict]:
    """
    递归扫描本地 PDF 数据库文件夹

    Args:
        root_folder: 根文件夹路径，支持子文件夹结构

    Returns:
        包含所有 PDF 文件信息的列表
    """
    papers = []
    root_path = Path(root_folder)

    if not root_path.exists():
        print(f"Local PDF database folder does not exist: {root_folder}")
        return papers

    # 递归查找所有 PDF 文件
    pdf_files = list(root_path.rglob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files in {root_folder}")

    for pdf_file in pdf_files:
        pdf_path = str(pdf_file.absolute())
        try:
            metadata = extract_pdf_metadata(pdf_path)
            papers.append(metadata)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")

    return papers


def search_local_pdfs(topic: str, root_folder: str, max_results: int = 50) -> List[Dict]:
    """
    在本地 PDF 数据库中搜索相关论文

    目前的实现：返回所有本地 PDF（因为本地数据库通常是为了特定主题准备的）
    可以根据需要扩展为基于关键词的过滤

    Args:
        topic: 搜索主题
        root_folder: 本地 PDF 数据库根文件夹
        max_results: 最大返回结果数

    Returns:
        匹配的论文列表
    """
    all_papers = scan_local_pdf_database(root_folder)

    # 简单关键词匹配（可选）
    if topic and topic.strip():
        topic_lower = topic.lower()
        topic_words = set(re.findall(r'\w+', topic_lower))

        # 根据标题与主题的匹配程度排序
        def relevance_score(paper):
            title_lower = paper.get("title", "").lower()
            # 计算匹配的关键词数量
            matches = sum(1 for word in topic_words if word in title_lower)
            return matches

        # 按相关性排序
        all_papers.sort(key=relevance_score, reverse=True)

    return all_papers[:max_results]


def get_local_pdf_path(arxiv_id: str, root_folder: str) -> Optional[str]:
    """
    根据本地 ID 获取 PDF 文件路径

    Args:
        arxiv_id: 本地生成的 ID
        root_folder: 本地 PDF 数据库根文件夹

    Returns:
        PDF 文件路径，如果找不到则返回 None
    """
    if not arxiv_id.startswith("local_"):
        return None

    # 重新扫描找到匹配的 PDF
    papers = scan_local_pdf_database(root_folder)
    for paper in papers:
        if paper.get("arxiv_id") == arxiv_id:
            return paper.get("pdf_link")
    return None


# 配置：本地 PDF 数据库根文件夹
# 可以从环境变量或配置文件读取
DEFAULT_LOCAL_DB_PATH = os.environ.get("LOCAL_PDF_DB_PATH", "")


def set_local_db_path(path: str):
    """设置本地 PDF 数据库路径"""
    global DEFAULT_LOCAL_DB_PATH
    DEFAULT_LOCAL_DB_PATH = path


def get_local_db_path() -> str:
    """获取当前设置的本地 PDF 数据库路径"""
    return DEFAULT_LOCAL_DB_PATH
