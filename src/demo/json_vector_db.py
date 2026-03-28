"""
JSON 论文数据库的向量检索模块
用于从 JSON 文件加载论文标题并建立向量索引
JSON 格式: {"title": "pdf_path"}
"""
import os
import json
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

# 尝试导入 embedding 库
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Please install: pip install sentence-transformers")

# 尝试导入 FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not available. Please install: pip install faiss-cpu")


class JSONVectorDB:
    """
    JSON 文件向量数据库

    用法:
        db = JSONVectorDB(json_folder="src/static/local_pdfs")
        db.build_index()  # 首次使用需要构建索引

        # 搜索
        results = db.search("optical fiber communication", k=50)
        # 返回: [{"title": "...", "pdf_path": "...", "score": 0.95, ...}, ...]
    """

    def __init__(
        self,
        json_folder: str = "src/static/local_pdfs",
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None
    ):
        """
        初始化向量数据库

        Args:
            json_folder: 存放 JSON 文件的文件夹路径
            model_name: embedding 模型名称
            cache_dir: 索引缓存文件夹
        """
        self.json_folder = Path(json_folder)
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else self.json_folder / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 数据存储
        self.titles: List[str] = []  # 论文标题列表
        self.pdf_paths: List[str] = []  # PDF 文件路径列表
        self.json_sources: List[str] = []  # 来源 JSON 文件名
        self.paper_ids: List[str] = []  # 论文唯一 ID

        # FAISS 索引
        self.index = None
        self.embedding_model = None

        # 检查依赖
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required. Install: pip install sentence-transformers")
        if not FAISS_AVAILABLE:
            raise ImportError("faiss is required. Install: pip install faiss-cpu")

    def _get_cache_paths(self) -> Tuple[Path, Path]:
        """获取缓存文件路径"""
        index_cache = self.cache_dir / f"faiss_index_{self.model_name.replace('/', '_')}.bin"
        data_cache = self.cache_dir / f"papers_data_{self.model_name.replace('/', '_')}.pkl"
        return index_cache, data_cache

    def _get_json_files(self) -> List[Path]:
        """Return JSON files in a stable order."""
        if not self.json_folder.exists():
            return []
        return sorted(self.json_folder.glob("*.json"), key=lambda path: path.name.lower())

    def _compute_json_signature(self, json_files: List[Path]) -> str:
        """Fingerprint the current JSON set so stale caches can be invalidated."""
        hasher = hashlib.md5()
        for json_file in json_files:
            stat = json_file.stat()
            hasher.update(str(json_file.resolve()).encode("utf-8"))
            hasher.update(str(stat.st_size).encode("utf-8"))
            hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
        return hasher.hexdigest()

    def _load_json_files(self) -> Dict[str, dict]:
        """
        加载所有 JSON 文件

        Returns:
            {title: {pdf_path, json_source}} 的字典
            JSON 格式: {"title": "pdf_path"}
        """
        all_papers = {}

        if not self.json_folder.exists():
            print(f"JSON folder does not exist: {self.json_folder}")
            return all_papers

        # 查找所有 JSON 文件
        json_files = self._get_json_files()
        print(f"Found {len(json_files)} JSON files in {self.json_folder}")

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # JSON 格式是 {title: pdf_path}
                if isinstance(data, dict):
                    for title, pdf_path in data.items():
                        # 存储来源信息
                        all_papers[title] = {
                            "pdf_path": pdf_path,
                            "json_source": json_file.name
                        }
                else:
                    print(f"Warning: {json_file} is not a dictionary, skipping")

            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        print(f"Loaded {len(all_papers)} papers from JSON files")
        return all_papers

    def build_index(self, force_rebuild: bool = False) -> bool:
        """
        构建向量索引

        Args:
            force_rebuild: 是否强制重新构建索引

        Returns:
            是否成功构建
        """
        index_cache, data_cache = self._get_cache_paths()
        json_files = self._get_json_files()
        json_signature = self._compute_json_signature(json_files)

        # 检查缓存是否存在
        if not force_rebuild and index_cache.exists() and data_cache.exists():
            print("Loading cached index...")
            try:
                self._load_cache(index_cache, data_cache)
                if getattr(self, "json_signature", None) == json_signature:
                    return True
                print("JSON files changed since cache build, rebuilding index...")
            except Exception as e:
                print(f"Cache loading failed: {e}, rebuilding...")

        # 加载 JSON 数据
        papers_dict = self._load_json_files()
        if not papers_dict:
            print("No papers found in JSON files")
            return False

        # 准备数据
        self.titles = []
        self.pdf_paths = []
        self.json_sources = []
        self.paper_ids = []

        for title, info in papers_dict.items():
            self.titles.append(title)
            self.pdf_paths.append(info["pdf_path"])
            self.json_sources.append(info["json_source"])
            # 生成唯一 ID
            paper_id = hashlib.md5(f"{title}_{info['json_source']}".encode()).hexdigest()[:16]
            self.paper_ids.append(f"json_{paper_id}")

        print(f"Building embeddings for {len(self.titles)} papers...")

        # 加载 embedding 模型
        print(f"Loading embedding model: {self.model_name}")
        self.embedding_model = SentenceTransformer(self.model_name)

        # 生成 embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            self.titles,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # 归一化 embeddings（用于余弦相似度）
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # 创建 FAISS 索引
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        # 使用内积索引（归一化后的内积 = 余弦相似度）
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))

        print(f"Index built successfully. Total papers: {len(self.titles)}")
        self.json_signature = json_signature

        # 保存缓存
        self._save_cache(index_cache, data_cache)

        return True

    def _save_cache(self, index_path: Path, data_path: Path):
        """保存索引和数据到缓存"""
        try:
            # 保存 FAISS 索引
            faiss.write_index(self.index, str(index_path))

            # 保存论文数据
            data = {
                "titles": self.titles,
                "pdf_paths": self.pdf_paths,
                "json_sources": self.json_sources,
                "paper_ids": self.paper_ids,
                "json_signature": self._compute_json_signature(self._get_json_files())
            }
            with open(data_path, 'wb') as f:
                pickle.dump(data, f)

            print(f"Cache saved to {self.cache_dir}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    def _load_cache(self, index_path: Path, data_path: Path):
        """从缓存加载索引和数据"""
        # 加载 FAISS 索引
        self.index = faiss.read_index(str(index_path))

        # 加载论文数据
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.titles = data["titles"]
        self.pdf_paths = data["pdf_paths"]
        self.json_sources = data["json_sources"]
        self.paper_ids = data["paper_ids"]
        self.json_signature = data.get("json_signature")

        # 加载 embedding 模型
        self.embedding_model = SentenceTransformer(self.model_name)

        print(f"Cache loaded. Total papers: {len(self.titles)}")

    def search(self, query: str, k: int = 50) -> List[Dict]:
        """
        搜索最相似的论文

        Args:
            query: 搜索查询
            k: 返回结果数量

        Returns:
            论文列表，按相似度排序
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        if not query or not query.strip():
            return []

        # 生成查询的 embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        # 搜索
        scores, indices = self.index.search(query_embedding.astype('float32'), min(k, len(self.titles)))

        # 构建结果
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS 返回 -1 表示没有更多结果
                break
            results.append({
                "title": self.titles[idx],
                "pdf_path": self.pdf_paths[idx],
                "json_source": self.json_sources[idx],
                "paper_id": self.paper_ids[idx],
                "score": float(score),
                "rank": i + 1,
                "source": "json_vec"
            })

        return results

    def get_paper_by_id(self, paper_id: str) -> Optional[Dict]:
        """根据 ID 获取论文信息"""
        try:
            idx = self.paper_ids.index(paper_id)
            return {
                "title": self.titles[idx],
                "pdf_path": self.pdf_paths[idx],
                "json_source": self.json_sources[idx],
                "paper_id": paper_id,
                "source": "json_vec"
            }
        except ValueError:
            return None

    def get_stats(self) -> Dict:
        """获取数据库统计信息"""
        return {
            "total_papers": len(self.titles),
            "json_files": len(set(self.json_sources)) if self.json_sources else 0,
            "model": self.model_name,
            "index_built": self.index is not None
        }


# 全局实例（单例模式）
_db_instance: Optional[JSONVectorDB] = None

def get_db_instance(json_folder: str = "src/static/local_pdfs") -> JSONVectorDB:
    """获取全局数据库实例"""
    global _db_instance
    if _db_instance is None or _db_instance.json_folder != Path(json_folder):
        _db_instance = JSONVectorDB(json_folder=json_folder)
    return _db_instance


def search_json_papers(query: str, json_folder: str = "src/static/local_pdfs", k: int = 50) -> List[Dict]:
    """
    便捷的搜索函数

    Args:
        query: 搜索查询
        json_folder: JSON 文件夹路径
        k: 返回结果数量

    Returns:
        论文列表
    """
    db = get_db_instance(json_folder)

    if db.index is None:
        success = db.build_index()
        if not success:
            return []

    return db.search(query, k=k)


# 测试代码
if __name__ == "__main__":
    # 测试
    db = JSONVectorDB(json_folder="src/static/local_pdfs")
    success = db.build_index()

    if success:
        print("\nDatabase stats:", db.get_stats())

        # 测试搜索
        query = "optical fiber communication"
        print(f"\nSearching for: {query}")
        results = db.search(query, k=10)

        for r in results:
            print(f"\n[{r['rank']}] Score: {r['score']:.4f}")
            print(f"Title: {r['title']}")
            print(f"PDF: {r['pdf_path']}")
            print(f"Source: {r['json_source']}")
