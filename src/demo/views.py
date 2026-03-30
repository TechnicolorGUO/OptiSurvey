from __future__ import unicode_literals
import sys
from langchain_huggingface import HuggingFaceEmbeddings
# 禁用所有遥测功能
import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['DISABLE_TELEMETRY'] = '1'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'
# os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_DISABLED'] = 'true'
os.environ['COMET_DISABLE_AUTO_LOGGING'] = '1'
# 添加 Hugging Face 离线模式和缓存设置
# os.environ['HF_DATASETS_OFFLINE'] = '1'  
# os.environ['TRANSFORMERS_CACHE'] = './models/transformers_cache'
# os.environ['HF_HOME'] = './models/huggingface_cache'
# os.environ['HF_HUB_CACHE'] = './models/huggingface_hub_cache'

from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.uploadedfile import InMemoryUploadedFile
import os
import json
import requests
import time
import pandas as pd
import shutil
import traceback
from io import BytesIO

import hashlib
import re
import os
import csv
import xml.etree.ElementTree as ET
import urllib.parse

from django.http import JsonResponse
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage

# from .parse import DocumentLoading
from .asg_retriever import legal_pdf, process_pdf, query_embeddings_new_new, cleanup_retriever
from .asg_generator import generate, generate_sentence_patterns, normalize_query_list, cleanup_openai_client, getQwenClient, generateResponse
from .asg_outline import OutlineGenerator,generateOutlineHTML_qwen, generateSurvey_qwen_new
from .asg_clustername import generate_cluster_name_new
from .postprocess import generate_references_section
from .asg_query import generate_generic_query_qwen, generate_query_qwen
from .asg_add_flowchart import insert_ref_images, detect_flowcharts
from .asg_mindmap import generate_graphviz_png, insert_outline_image
from .asg_latex import tex_to_pdf, insert_figures, md_to_tex, preprocess_md
from .local_pdf_db import search_local_pdfs, scan_local_pdf_database, get_local_pdf_path, set_local_db_path
from .library_nebula import get_library_nebula_payload
# from .survey_generator_api import ensure_all_papers_cited
import glob

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from functools import wraps

load_dotenv()
# # 打印所有环境变量（可选，调试时使用）
# print("所有环境变量:", os.environ)

# # 获取特定环境变量
# openai_api_key = os.getenv("OPENAI_API_KEY")
# openai_api_base = os.getenv("OPENAI_API_BASE")

# # 打印获取到的值
# print(f"OPENAI_API_KEY: {openai_api_key}")
# print(f"OPENAI_API_BASE: {openai_api_base}")

# 导入异步任务支持
from .middleware import async_task, task_manager

import os
from pathlib import Path
from markdown_pdf import MarkdownPdf, Section

DATA_PATH = './src/static/data/pdf/'
TXT_PATH = './src/static/data/txt/'
TSV_PATH = './src/static/data/tsv/'
MD_PATH = './src/static/data/md/'
INFO_PATH = './src/static/data/info/'
IMG_PATH = './src/static/img/'

OPTIRESEARCH_DEV_MODE = True
OPTIRESEARCH_DEV_PDF_RELATIVE_PATH = os.path.join('src', 'static', 'data', 'results', 'survey_fa56a18080_latex.pdf')
OPTIRESEARCH_DEV_FALLBACK_TO_LATEST = True

paths = [DATA_PATH, TXT_PATH, TSV_PATH, MD_PATH, INFO_PATH, IMG_PATH]

for path in paths:
    path_obj = Path(path)
    if not path_obj.exists():
        path_obj.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")



Survey_dict = {
    '2742488' : 'Energy Efficiency in Cloud Computing',
    '2830555' : 'Cache Management for Real-Time Systems',
    '2907070' : 'Predictive Modeling on Imbalanced Data',
    '3073559' : 'Malware Detection with Data Mining',
    '3274658' : 'Analysis of Handwritten Signature'
}



Survey_Topic_dict = {
    '2742488' : ['energy'],
    '2830555' : ['cache'],
    '2907070' : ['imbalanced'],
    '3073559' : ['malware', 'detection'],
    '3274658' : ['handwritten', 'signature']
}


Survey_n_clusters = {
    '2742488' : 3,
    '2830555' : 3,
    '2907070' : 3,
    '3073559' : 3,
    '3274658' : 2
}

Global_survey_id = ""
Global_survey_title=""
Global_ref_list = []
Global_category_description = []
Global_category_label = []
Global_df_selected = ""
Global_test_flag = False
Global_collection_names = []
Global_collection_names_clustered = []
Global_file_names=[]
Global_description_list = []
Global_cluster_names = []
Global_citation_data = []
Global_cluster_num = 4


# 创建模型缓存目录
import os
from pathlib import Path

def ensure_cache_dirs():
    """确保缓存目录存在"""
    cache_dirs = [
        './models/transformers_cache',
        './models/huggingface_cache', 
        './models/huggingface_hub_cache'
    ]
    for cache_dir in cache_dirs:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

# def init_embedder_with_retry():
#     """初始化embedder，带重试和错误处理"""
#     ensure_cache_dirs()
    
#     try:
#         print("正在初始化 SentenceTransformer embeddings...")
#         # 尝试初始化embedder
#         model = SentenceTransformer(
#             'sentence-transformers/all-MiniLM-L6-v2',
#             cache_folder='./models/transformers_cache'
#         )
#         print("SentenceTransformer embeddings 初始化成功")
#         return model
        
#     except Exception as e:
#         print(f"初始化 SentenceTransformer embeddings 失败: {e}")
#         print("尝试使用本地缓存或替代方案...")
        
#         try:
#             # 尝试使用本地缓存
#             model = SentenceTransformer(
#                 'sentence-transformers/all-MiniLM-L6-v2',
#                 cache_folder='./models/transformers_cache',
#                 local_files_only=True
#             )
#             print("使用本地缓存成功")
#             return model
#         except Exception as e2:
#             print(f"使用本地缓存也失败: {e2}")
#             print("警告: 将使用空的 embedder，某些功能可能不可用")
#             return None

# 初始化embedder
embedder = None  # 延迟初始化

def get_embedder():
    """获取embedder实例，如果未初始化则进行初始化"""
    global embedder
    if embedder is None:
        try:
            print("正在初始化 embedder...")
            embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            print("embedder 初始化完成")
        except Exception as e:
            print(f"embedder 初始化失败: {e}")
            return None
    return embedder

from demo.category_and_tsne import clustering

# 添加超时装饰器
def timeout_handler(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                # 线程仍在运行，说明超时了
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        return wrapper
    return decorator

# 添加进度跟踪
progress_tracker = {}

def update_progress(operation_id, progress, message="", meta=None):
    """更新操作进度"""
    progress_tracker[operation_id] = {
        'progress': progress,
        'message': message,
        'timestamp': time.time()
    }
    if isinstance(meta, dict):
        progress_tracker[operation_id].update(meta)
    print(f"[{operation_id}] {progress}% - {message}")

def get_progress(operation_id):
    """获取操作进度"""
    return progress_tracker.get(operation_id, {'progress': 0, 'message': 'Starting...', 'timestamp': time.time()})

def no_cache_json_response(data, status=200):
    response = JsonResponse(data, status=status)
    response['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    return response

# 添加进度查询端点
@csrf_exempt
def get_operation_progress(request):
    """获取操作进度的API端点，支持异步任务结果"""
    if request.method == 'GET':
        operation_id = request.GET.get('operation_id')
        if operation_id:
            print(f"[DEBUG] Checking progress for operation_id: {operation_id}")
            
            # 首先检查异步任务状态
            task_status = task_manager.get_task_status(operation_id)
            print(f"[DEBUG] Task status: {task_status}")
            
            if task_status['status'] == 'completed':
                # 任务完成，返回结果
                print(f"[DEBUG] Task {operation_id} completed, returning result")
                result = task_status.get('result')
                progress_info = get_progress(operation_id)
                
                # 检查result是否是HttpResponse对象（旧的PDF生成方式）
                if hasattr(result, 'content'):
                    try:
                        import json
                        content = json.loads(result.content.decode('utf-8'))
                        return no_cache_json_response({
                            **progress_info,
                            'progress': 100,
                            'message': 'Task completed successfully!',
                            'status': 'completed',
                            'result': content
                        })
                    except Exception as e:
                        print(f"[DEBUG] Error parsing HttpResponse content: {e}")
                        # 对于PDF等二进制文件，我们不解析内容，只返回完成状态
                        return no_cache_json_response({
                            **progress_info,
                            'progress': 100,
                            'message': 'Task completed successfully!',
                            'status': 'completed',
                            'result': {'message': 'Binary file generated successfully'}
                        })
                # 检查result是否是Django JsonResponse对象
                elif hasattr(result, 'content') and hasattr(result, 'status_code'):
                    try:
                        import json
                        content = json.loads(result.content.decode('utf-8'))
                        return no_cache_json_response({
                            **progress_info,
                            'progress': 100,
                            'message': 'Task completed successfully!',
                            'status': 'completed',
                            'result': content
                        })
                    except Exception as e:
                        print(f"[DEBUG] Error parsing JsonResponse content: {e}")
                        return no_cache_json_response({
                            **progress_info,
                            'progress': 100,
                            'message': 'Task completed successfully!',
                            'status': 'completed'
                        })
                else:
                    # 普通的结果对象
                    return no_cache_json_response({
                        **progress_info,
                        'progress': 100,
                        'message': 'Task completed successfully!',
                        'status': 'completed',
                        'result': result
                    })
            elif task_status['status'] == 'failed':
                # 任务失败
                print(f"[DEBUG] Task {operation_id} failed: {task_status.get('error')}")
                progress_info = get_progress(operation_id)
                return no_cache_json_response({
                    **progress_info,
                    'progress': -1,
                    'message': f"Task failed: {task_status.get('error', 'Unknown error')}",
                    'status': 'failed',
                    'error': task_status.get('error')
                })
            elif task_status['status'] == 'running':
                # 任务正在运行，返回当前进度
                print(f"[DEBUG] Task {operation_id} is running, checking progress")
                progress_info = get_progress(operation_id)
                print(f"[DEBUG] Progress info: {progress_info}")
                return no_cache_json_response({
                    **progress_info,
                    'status': 'running'
                })
            else:
                # 任务未找到，返回默认进度
                print(f"[DEBUG] Task {operation_id} not found, returning default progress")
                progress_info = get_progress(operation_id)
                return no_cache_json_response({
                    **progress_info,
                    'status': 'not_found'
                })
        return no_cache_json_response({'error': 'operation_id is required'}, status=400)
    return no_cache_json_response({'error': 'Invalid request method'}, status=405)

class reference_collection(object):
    def __init__(
            self,
            input_df
    ):
        self.input_df = input_df

    def full_match_with_entries_in_pd(self, query_paper_titles):
        entries_in_pd = self.input_df.copy()
        entries_in_pd['ref_title'] = entries_in_pd['ref_title'].apply(str.lower)
        query_paper_titles = [i.lower() for i in query_paper_titles]

        # matched_entries = entries_in_pd[entries_in_pd['ref_title'].isin(query_paper_titles)]
        matched_entries = self.input_df[entries_in_pd['ref_title'].isin(query_paper_titles)]
        return matched_entries,matched_entries.shape[0]

    # select the sentences that can match with the topic words
    def match_ref_paper(self, query_paper_titles,match_mode='full', match_ratio=70):
        # query_paper_title = query_paper_title.lower()
        # two modes for str matching
        if match_mode == 'full':
            matched_entries, matched_num = self.full_match_with_entries_in_pd(query_paper_titles)
        return matched_entries, matched_num


def generate_uid():
    uid_str=""
    hash = hashlib.sha1()
    hash.update(str(time.time()).encode('utf-8'))
    uid_str= hash.hexdigest()[:10]

    return uid_str

def index(request):
    return render(request, 'demo/index.html')

def delete_files(request):
    if request.method == 'POST':
        try:
            folders = ['./src/static/data/pdf/', './src/static/data/tsv/', './src/static/data/txt/', './src/static/data/md/']
            for folder in folders:
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        return JsonResponse({'success': False, 'message': str(e)})
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)})

    return JsonResponse({'success': False, 'message': 'Invalid request method'})

def clean_str(input_str):
    input_str = str(input_str).strip().lower()
    if input_str == "none" or input_str == "nan" or len(input_str) == 0:
        return ""
    input_str = input_str.replace('\\n',' ').replace('\n',' ').replace('\r',' ').replace('——',' ').replace('——',' ').replace('__',' ').replace('__',' ').replace('........','.').replace('....','.').replace('....','.').replace('..','.').replace('..','.').replace('..','.').replace('. . . . . . . . ','. ').replace('. . . . ','. ').replace('. . . . ','. ').replace('. . ','. ').replace('. . ','. ')
    input_str = re.sub(r'\\u[0-9a-z]{4}', ' ', input_str).replace('  ',' ').replace('  ',' ')
    return input_str

def PosRank_get_top5_ngrams(input_pd):
    pos = {'NOUN', 'PROPN', 'ADJ'}
    extractor = PosRank()

    abs_top5_unigram_list_list = []
    abs_top5_bigram_list_list = []
    abs_top5_trigram_list_list = []
    intro_top5_unigram_list_list = []
    intro_top5_bigram_list_list = []
    intro_top5_trigram_list_list = []

    for line_index,pd_row in input_pd.iterrows():

        input_str=pd_row["abstract"].replace('-','')
        extractor.load_document(input=input_str,language='en',normalization=None)

        #unigram
        unigram_extractor=extractor
        unigram_extractor.candidate_selection(maximum_word_number=1,minimum_word_number=1)
        unigram_extractor.candidate_weighting(window=6,pos=pos,normalized=False)
        abs_top5_unigram_list = []
        for (keyphrase, score) in unigram_extractor.get_n_best(n=5, stemming=True):
            keyphrase = keyphrase.replace('-','')
            if len(keyphrase)>2:
                abs_top5_unigram_list.append(keyphrase)
        #pdb.set_trace()
        #bigram
        bigram_extractor=extractor
        bigram_extractor.candidate_selection(maximum_word_number=2,minimum_word_number=2)
        bigram_extractor.candidate_weighting(window=6,pos=pos,normalized=False)
        abs_top5_bigram_list = []
        for (keyphrase, score) in bigram_extractor.get_n_best(n=5, stemming=True):
            keyphrase = keyphrase.replace('-','')
            if len(keyphrase)>2:
                abs_top5_bigram_list.append(keyphrase)

        #trigram
        trigram_extractor=extractor
        trigram_extractor.candidate_selection(maximum_word_number=3,minimum_word_number=3)
        trigram_extractor.candidate_weighting(window=6,pos=pos,normalized=False)
        abs_top5_trigram_list = []
        for (keyphrase, score) in trigram_extractor.get_n_best(n=5, stemming=True):
            keyphrase = keyphrase.replace('-','')
            if len(keyphrase)>2:
                abs_top5_trigram_list.append(keyphrase)

        abs_top5_unigram_list_list.append(abs_top5_unigram_list)
        abs_top5_bigram_list_list.append(abs_top5_bigram_list)
        abs_top5_trigram_list_list.append(abs_top5_trigram_list)

    return abs_top5_unigram_list_list,abs_top5_bigram_list_list,abs_top5_trigram_list_list

def process_file(file_name, survey_id, mode):
    embedder_instance = get_embedder()
    if embedder_instance is None:
        print("警告: embedder 未初始化，跳过PDF处理")
        # 返回一个默认值或抛出更友好的错误
        collection_name = f"collection_{survey_id}_{int(time.time())}"
        name = file_name.split('/')[-1].replace('.pdf', '')
        return collection_name, name
    
    result = process_pdf(file_name, survey_id, embedder_instance, mode)
    collection_name = result[0]
    name = result[-1]
    return collection_name, name

def sanitize_filename_py(filename):
    last_dot = filename.rfind('.')
    
    def sanitize_part(part):
        part = part.lower()
        part = re.sub(r'[^a-z0-9]', ' ', part)
        part = re.sub(r'\s+', ' ', part)
        part = part.strip()
        words = part.split(' ')        
        if len(words) == 0:
            return ''  
        words[0] = words[0].capitalize()
        
        return ' '.join(words)
    
    if last_dot == -1:
        # No extension
        return sanitize_part(filename)
    elif last_dot == 0:
        # Hidden file
        extension = filename[1:]
        return '.' + sanitize_part(extension)
    else:
        # With extension
        name = filename[:last_dot]
        extension = filename[last_dot + 1:]
        return sanitize_part(name) + '.' + sanitize_part(extension)

def get_existing_survey_ids():

    tsv_directory = os.path.join("src", "static", "data", "tsv")
    survey_ids = []
    try:
        for file_name in os.listdir(tsv_directory):
            if file_name.endswith(".tsv"):
                # 去掉 .tsv 后缀
                survey_ids.append(file_name[:-4])
    except Exception as e:
        print("Error reading tsv directory:", e)
    return survey_ids

def get_surveys(request):

    surveys = get_existing_survey_ids()
    return JsonResponse({'surveys': surveys})


def get_library_nebula(request):
    force_rebuild = str(request.GET.get('refresh', '')).lower() in {'1', 'true', 'yes'}
    try:
        payload = get_library_nebula_payload(force_rebuild=force_rebuild)
        return JsonResponse(payload, json_dumps_params={'ensure_ascii': False})
    except Exception as e:
        return JsonResponse(
            {
                'status': 'error',
                'error': str(e),
                'message': 'Failed to build library nebula payload.'
            },
            status=500,
            json_dumps_params={'ensure_ascii': False}
        )

def _strip_thinking_blocks(text):
    if not text:
        return ""
    text = re.sub(r'<\s*think\s*>[\s\S]*?<\s*/\s*think\s*>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<\s*thinking\s*>[\s\S]*?<\s*/\s*thinking\s*>', '', text, flags=re.IGNORECASE)
    return text.strip()

def _get_optiresearch_payload_for_survey(survey_id):
    if not survey_id:
        return None

    results_dir = os.path.join('src', 'static', 'data', 'results')
    txt_dir = os.path.join('src', 'static', 'data', 'txt', survey_id)
    info_dir = os.path.join('src', 'static', 'data', 'info', survey_id)
    latex_filename = f'survey_{survey_id}_latex.pdf'
    regular_filename = f'survey_{survey_id}.pdf'
    latex_path = os.path.join(results_dir, latex_filename)
    regular_path = os.path.join(results_dir, regular_filename)
    generated_result_path = os.path.join(txt_dir, 'generated_result.json')
    processed_md_path = os.path.join(info_dir, f'survey_{survey_id}_processed.md')
    vanilla_md_path = os.path.join(info_dir, f'survey_{survey_id}_vanilla.md')
    chat_enabled = (
        os.path.exists(generated_result_path)
        or os.path.exists(processed_md_path)
        or os.path.exists(vanilla_md_path)
    )

    if not (os.path.exists(latex_path) or os.path.exists(regular_path) or os.path.exists(generated_result_path)):
        return None

    return {
        'enabled': os.path.exists(latex_path),
        'survey_id': survey_id,
        'latex_pdf_url': f'/static/data/results/{latex_filename}' if os.path.exists(latex_path) else '',
        'pdf_url': f'/static/data/results/{regular_filename}' if os.path.exists(regular_path) else '',
        'chat_enabled': chat_enabled,
        'has_generated_result': os.path.exists(generated_result_path),
        'has_processed_markdown': os.path.exists(processed_md_path),
    }

def _extract_survey_id_from_latex_pdf(file_path):
    file_name = os.path.basename(file_path or '')
    match = re.match(r'^survey_(.+?)_latex\.pdf$', file_name)
    return match.group(1) if match else ''

def _build_optiresearch_payload_from_pdf(file_path, developer_override=False):
    if not file_path or not os.path.exists(file_path):
        return None

    survey_id = _extract_survey_id_from_latex_pdf(file_path)
    payload = _get_optiresearch_payload_for_survey(survey_id) if survey_id else None

    if payload:
        payload['enabled'] = True
        payload['latex_pdf_url'] = f"/static/data/results/{os.path.basename(file_path)}"
        payload['developer_override'] = developer_override
        return payload

    return {
        'enabled': True,
        'survey_id': survey_id,
        'latex_pdf_url': f"/static/data/results/{os.path.basename(file_path)}",
        'pdf_url': '',
        'chat_enabled': False,
        'has_generated_result': False,
        'has_processed_markdown': False,
        'developer_override': developer_override
    }

def _get_optiresearch_developer_payload():
    if not OPTIRESEARCH_DEV_MODE:
        return None

    preferred_path = OPTIRESEARCH_DEV_PDF_RELATIVE_PATH
    if preferred_path and os.path.exists(preferred_path):
        return _build_optiresearch_payload_from_pdf(preferred_path, developer_override=True)

    if not OPTIRESEARCH_DEV_FALLBACK_TO_LATEST:
        return None

    results_dir = os.path.join('src', 'static', 'data', 'results')
    if not os.path.exists(results_dir):
        return None

    latex_candidates = []
    for file_name in os.listdir(results_dir):
        if not file_name.startswith('survey_') or not file_name.endswith('_latex.pdf'):
            continue
        full_path = os.path.join(results_dir, file_name)
        latex_candidates.append((os.path.getmtime(full_path), full_path))

    if not latex_candidates:
        return None

    latest_pdf_path = sorted(latex_candidates, key=lambda item: item[0], reverse=True)[0][1]
    return _build_optiresearch_payload_from_pdf(latest_pdf_path, developer_override=True)

def _find_latest_optiresearch_payload():
    candidates = []
    results_dir = os.path.join('src', 'static', 'data', 'results')
    txt_root = os.path.join('src', 'static', 'data', 'txt')

    if os.path.exists(results_dir):
        for file_name in os.listdir(results_dir):
            if not file_name.startswith('survey_') or not file_name.endswith('_latex.pdf'):
                continue
            survey_id = file_name[len('survey_'):-len('_latex.pdf')]
            file_path = os.path.join(results_dir, file_name)
            candidates.append((os.path.getmtime(file_path), survey_id))

    if os.path.exists(txt_root):
        for survey_id in os.listdir(txt_root):
            generated_result_path = os.path.join(txt_root, survey_id, 'generated_result.json')
            if os.path.exists(generated_result_path):
                candidates.append((os.path.getmtime(generated_result_path), survey_id))

    if not candidates:
        return None

    for _, survey_id in sorted(candidates, key=lambda item: item[0], reverse=True):
        payload = _get_optiresearch_payload_for_survey(survey_id)
        if payload:
            return payload
    return None

def _load_optiresearch_context(survey_id):
    if not survey_id:
        return "", ""

    info_dir = os.path.join('src', 'static', 'data', 'info', survey_id)
    txt_dir = os.path.join('src', 'static', 'data', 'txt', survey_id)

    processed_md_path = os.path.join(info_dir, f'survey_{survey_id}_processed.md')
    vanilla_md_path = os.path.join(info_dir, f'survey_{survey_id}_vanilla.md')
    generated_result_path = os.path.join(txt_dir, 'generated_result.json')

    if os.path.exists(processed_md_path):
        with open(processed_md_path, 'r', encoding='utf-8') as file:
            return file.read(), processed_md_path

    if os.path.exists(vanilla_md_path):
        with open(vanilla_md_path, 'r', encoding='utf-8') as file:
            return file.read(), vanilla_md_path

    if os.path.exists(generated_result_path):
        with open(generated_result_path, 'r', encoding='utf-8') as file:
            generated_result = json.load(file)
        return generated_result.get('content', ''), generated_result_path

    return "", ""

def _clean_optiresearch_section_title(title):
    title = re.sub(r'[`*_#>\[\]\(\)]', ' ', title or '')
    title = re.sub(r'\s+', ' ', title)
    return title.strip()

def _build_optiresearch_sections(context):
    if not context:
        return []

    sections = []
    heading_stack = []
    current_section = None

    def flush_section():
        nonlocal current_section
        if not current_section:
            return
        content = "\n".join(current_section['lines']).strip()
        if content:
            sections.append({
                'section_path': current_section['section_path'],
                'title_level': current_section['title_level'],
                'content': content
            })
        current_section = None

    for raw_line in context.splitlines():
        stripped_line = raw_line.strip()
        heading_match = re.match(r'^(#{1,6})\s+(.+?)\s*$', stripped_line)
        if heading_match:
            flush_section()
            level = len(heading_match.group(1))
            title = _clean_optiresearch_section_title(heading_match.group(2))
            if not title:
                continue

            while heading_stack and heading_stack[-1]['level'] >= level:
                heading_stack.pop()
            heading_stack.append({'level': level, 'title': title})

            current_section = {
                'section_path': " > ".join(item['title'] for item in heading_stack),
                'title_level': level,
                'lines': []
            }
            continue

        if current_section is None:
            current_section = {
                'section_path': 'Document Overview',
                'title_level': 0,
                'lines': []
            }
        current_section['lines'].append(raw_line)

    flush_section()

    if not sections and context.strip():
        sections.append({
            'section_path': 'Document Overview',
            'title_level': 0,
            'content': context.strip()
        })

    return sections

def _extract_optiresearch_keywords(question):
    stopwords = {
        'the', 'and', 'for', 'with', 'that', 'this', 'from', 'into', 'about', 'what',
        'which', 'when', 'where', 'why', 'how', 'does', 'have', 'has', 'had', 'were',
        'was', 'are', 'is', 'can', 'could', 'would', 'should', 'their', 'there',
        'them', 'then', 'than', 'also', 'been', 'being', 'your', 'you', 'our',
        'his', 'her', 'its', 'they', 'who', 'whom', 'whose'
    }
    terms = re.findall(r'[a-zA-Z0-9][a-zA-Z0-9\-]{1,}', (question or '').lower())
    keywords = []
    for term in terms:
        if len(term) < 3 or term in stopwords:
            continue
        if term not in keywords:
            keywords.append(term)
    return keywords

def _select_optiresearch_sections(question, sections, max_sections=4):
    if not sections:
        return []

    keywords = _extract_optiresearch_keywords(question)
    normalized_question = (question or '').lower()
    scored_sections = []

    for index, section in enumerate(sections):
        title = section['section_path'].lower()
        content = section['content'].lower()
        score = 0

        for keyword in keywords:
            score += title.count(keyword) * 6
            score += content.count(keyword) * 2

        if normalized_question and normalized_question in content:
            score += 10
        if any(part.strip().lower() in title for part in re.split(r'[?.,;:]\s*', normalized_question) if part.strip()):
            score += 3

        # Slight preference for titled sections over the document root.
        if section['title_level'] > 0:
            score += 1

        scored_sections.append({
            'section_path': section['section_path'],
            'title_level': section['title_level'],
            'content': section['content'],
            'score': score,
            'index': index
        })

    scored_sections.sort(key=lambda item: (-item['score'], item['index']))
    selected = scored_sections[:max_sections]

    if selected and selected[0]['score'] == 0:
        selected = scored_sections[:min(2, len(scored_sections))]

    return selected

def _parse_optiresearch_response(response_text, fallback_sources):
    cleaned = _strip_thinking_blocks(response_text)
    fallback_sources = list(dict.fromkeys([source for source in fallback_sources if source]))
    allowed_sources = set(fallback_sources)

    def normalize_sources(sources):
        normalized = [str(source).strip() for source in sources if str(source).strip()]
        normalized = [source for source in normalized if source in allowed_sources]
        return list(dict.fromkeys(normalized)) or fallback_sources

    json_match = re.search(r'\{[\s\S]*\}', cleaned)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            answer = str(payload.get('answer', '')).strip() or cleaned.strip()
            sources = payload.get('sources', [])
            if isinstance(sources, str):
                sources = [sources]
            if not isinstance(sources, list):
                sources = []
            return answer, normalize_sources(sources)
        except Exception:
            pass

    sources_line_match = re.search(r'(?:^|\n)Sources?\s*:\s*(.+)$', cleaned, flags=re.IGNORECASE)
    if sources_line_match:
        sources_line = sources_line_match.group(1).strip()
        answer = re.sub(r'(?:^|\n)Sources?\s*:\s*.+$', '', cleaned, flags=re.IGNORECASE).strip()
        parsed_sources = [item.strip() for item in re.split(r'\s*\|\s*|,\s*', sources_line) if item.strip()]
        return answer or cleaned.strip(), normalize_sources(parsed_sources)

    return cleaned.strip(), fallback_sources

def _load_autoresearch_survey_title(survey_id, context=""):
    generated_result_path = os.path.join('src', 'static', 'data', 'txt', survey_id, 'generated_result.json')
    if os.path.exists(generated_result_path):
        try:
            with open(generated_result_path, 'r', encoding='utf-8') as file:
                payload = json.load(file)
            title = str(payload.get('survey_title') or payload.get('topic') or '').strip()
            if title:
                return title
        except Exception:
            pass

    if context:
        for line in context.splitlines():
            stripped = line.strip()
            heading_match = re.match(r'^#\s+(.*)$', stripped)
            if heading_match:
                candidate = heading_match.group(1).strip()
                candidate = re.sub(r'(?i)^a survey of\s+', '', candidate).strip()
                if candidate:
                    return candidate
                break

    if survey_id and survey_id == Global_survey_id and Global_survey_title:
        return Global_survey_title

    return f"Survey {survey_id}" if survey_id else "Current Survey"

def _load_autoresearch_citation_data(survey_id):
    citation_path = os.path.join('src', 'static', 'data', 'info', survey_id, 'citation_data.json')
    if not os.path.exists(citation_path):
        return []

    try:
        with open(citation_path, 'r', encoding='utf-8') as file:
            payload = json.load(file)
        if isinstance(payload, list):
            return payload
    except Exception:
        pass
    return []

def _extract_json_payload(response_text, root_key=None):
    cleaned = _strip_thinking_blocks(response_text).strip()
    if not cleaned:
        raise ValueError('Empty model response')

    candidates = [cleaned]
    for pattern in (r'\{[\s\S]*\}', r'\[[\s\S]*\]'):
        match = re.search(pattern, cleaned)
        if match:
            candidates.append(match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if root_key and isinstance(parsed, list):
                return {root_key: parsed}
            return parsed
        except Exception:
            continue

    raise ValueError('Unable to parse JSON response')

def _normalize_autoresearch_text(value, max_length=320):
    normalized = re.sub(r'\s+', ' ', str(value or '')).strip()
    if len(normalized) <= max_length:
        return normalized
    return normalized[:max_length - 3].rstrip() + '...'

def _build_autoresearch_context(context):
    sections = _build_optiresearch_sections(context)
    focus_query = 'research gaps limitations future work open challenges bottlenecks opportunities emerging directions'
    selected_sections = _select_optiresearch_sections(focus_query, sections, max_sections=6)

    context_blocks = []
    overview = context.strip()[:3200]
    if overview:
        context_blocks.append(f"[Survey Overview]\n{overview}")

    seen_paths = set()
    for section in selected_sections:
        section_path = section.get('section_path', '').strip() or 'Document Overview'
        if section_path in seen_paths:
            continue
        seen_paths.add(section_path)
        context_blocks.append(
            f"[Section: {section_path}]\n{section.get('content', '')[:2200]}"
        )

    if not context_blocks and context.strip():
        context_blocks.append(context[:12000])

    return "\n\n".join(context_blocks)[:16000]

def _safe_distance(value):
    try:
        return float(value)
    except Exception:
        return 9999.0

def _select_citations_for_seed(seed_text, citation_data_list, max_items=3):
    if not citation_data_list:
        return []

    keywords = _extract_optiresearch_keywords(seed_text)
    normalized_seed = str(seed_text or '').lower()
    scored_items = []

    for index, item in enumerate(citation_data_list):
        source = str(item.get('source', '')).strip()
        content = str(item.get('content', '')).strip()
        if not source and not content:
            continue

        source_lower = source.lower()
        content_lower = content.lower()
        score = 0

        for keyword in keywords:
            score += source_lower.count(keyword) * 7
            score += content_lower.count(keyword) * 2

        if normalized_seed and normalized_seed[:120] in content_lower:
            score += 8

        scored_items.append({
            'source': source or 'Uploaded paper',
            'content': _normalize_autoresearch_text(content, max_length=280),
            'distance': _safe_distance(item.get('distance')),
            'score': score,
            'index': index,
        })

    if not scored_items:
        return []

    scored_items.sort(key=lambda item: (-item['score'], item['distance'], item['index']))

    selected = []
    seen_sources = set()
    for item in scored_items:
        source_key = item['source'].lower()
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        selected.append({
            'source': item['source'],
            'content': item['content'],
        })
        if len(selected) >= max_items:
            break

    return selected

def _build_idea_evidence_packs(ideas, citation_data_list):
    evidence_packs = []
    for index, idea in enumerate(ideas, start=1):
        idea_id = str(idea.get('id') or f'I{index}')
        title = str(idea.get('title') or '').strip() or f'Idea {index}'
        core_insight = str(idea.get('core_insight') or idea.get('description') or '').strip()
        seed_text = " ".join(part for part in [title, core_insight] if part).strip()
        citations = _select_citations_for_seed(seed_text, citation_data_list, max_items=3)
        evidence_packs.append({
            'idea_id': idea_id,
            'title': title,
            'core_insight': core_insight,
            'citations': citations,
        })
    return evidence_packs

def _citation_sources_from_pack(citation_pack):
    return [item.get('source', '').strip() for item in citation_pack if item.get('source')]

def _attach_hypothesis_evidence(hypotheses, evidence_packs):
    evidence_by_idea = {pack['idea_id']: pack for pack in evidence_packs}
    enriched = []

    for index, hypothesis in enumerate(hypotheses, start=1):
        item = dict(hypothesis)
        item['hypothesis_id'] = str(item.get('hypothesis_id') or f'H{index}')
        item['idea_id'] = str(item.get('idea_id') or f'I{index}')
        cited_papers = item.get('cited_papers', [])
        if isinstance(cited_papers, str):
            cited_papers = [cited_papers]
        if not isinstance(cited_papers, list):
            cited_papers = []

        evidence_pack = evidence_by_idea.get(item['idea_id'], {'citations': []})
        pack_citations = evidence_pack.get('citations', [])
        pack_sources = set(_citation_sources_from_pack(pack_citations))
        normalized_sources = []
        for source in cited_papers:
            source_name = str(source).strip()
            if source_name and source_name in pack_sources and source_name not in normalized_sources:
                normalized_sources.append(source_name)

        if not normalized_sources:
            normalized_sources = _citation_sources_from_pack(pack_citations)[:2]

        evidence_snippets = []
        for citation in pack_citations:
            if citation.get('source') in normalized_sources:
                evidence_snippets.append(citation)

        item['title'] = str(item.get('title') or f'Hypothesis {index}').strip()
        item['research_gap'] = str(item.get('research_gap') or '').strip()
        item['hypothesis_statement'] = str(item.get('hypothesis_statement') or '').strip()
        item['mechanism'] = str(item.get('mechanism') or '').strip()
        item['test_plan'] = str(item.get('test_plan') or '').strip()
        item['expected_signal'] = str(item.get('expected_signal') or '').strip()
        item['evidence_reasoning'] = str(item.get('evidence_reasoning') or '').strip()
        item['cited_papers'] = normalized_sources
        item['evidence_snippets'] = evidence_snippets[:2]
        enriched.append(item)

    return enriched

def _merge_autoresearch_rankings(hypotheses, ranked_hypotheses, selected_candidate_ids):
    hypothesis_map = {
        str(hypothesis.get('hypothesis_id')): dict(hypothesis)
        for hypothesis in hypotheses
        if hypothesis.get('hypothesis_id')
    }

    merged_rankings = []
    for index, ranking in enumerate(ranked_hypotheses, start=1):
        hypothesis_id = str(ranking.get('hypothesis_id') or '')
        if not hypothesis_id:
            continue
        hypothesis = hypothesis_map.get(hypothesis_id, {})
        merged = dict(hypothesis)
        merged.update(ranking)
        merged['rank'] = int(merged.get('rank') or index)
        merged_rankings.append(merged)

    selected_candidates = []
    seen_ids = set()
    for hypothesis_id in selected_candidate_ids:
        normalized_id = str(hypothesis_id).strip()
        if not normalized_id or normalized_id in seen_ids:
            continue
        seen_ids.add(normalized_id)
        for item in merged_rankings:
            if str(item.get('hypothesis_id')) == normalized_id:
                selected_candidates.append(item)
                break

    return merged_rankings, selected_candidates

def _coerce_positive_int(value, default_value, min_value=1, max_value=20):
    try:
        parsed = int(value)
    except Exception:
        parsed = default_value
    return max(min_value, min(parsed, max_value))

def _coerce_autoresearch_history(history):
    if not isinstance(history, list):
        return []

    normalized_history = []
    for index, cycle in enumerate(history, start=1):
        if not isinstance(cycle, dict):
            continue
        selected_candidates = cycle.get('selected_candidates', [])
        if not isinstance(selected_candidates, list):
            selected_candidates = []
        normalized_candidates = []
        for candidate in selected_candidates:
            if not isinstance(candidate, dict):
                continue
            normalized_candidates.append({
                'hypothesis_id': str(candidate.get('hypothesis_id') or '').strip(),
                'title': str(candidate.get('title') or '').strip(),
                'hypothesis_statement': str(candidate.get('hypothesis_statement') or '').strip(),
                'total_score': candidate.get('total_score'),
            })

        normalized_history.append({
            'iteration': _coerce_positive_int(cycle.get('iteration', index), default_value=index, min_value=1, max_value=20),
            'selected_candidates': normalized_candidates,
            'stop_metrics': cycle.get('stop_metrics', {}),
        })
    return normalized_history

def _flatten_history_candidates(history):
    flattened = []
    for cycle in history:
        for candidate in cycle.get('selected_candidates', []):
            if not isinstance(candidate, dict):
                continue
            flattened.append(candidate)
    return flattened

def _tokenize_autoresearch_text(value):
    return set(re.findall(r'[a-zA-Z0-9][a-zA-Z0-9\-]{2,}', str(value or '').lower()))

def _candidate_signature(candidate):
    return " ".join([
        str(candidate.get('title') or ''),
        str(candidate.get('hypothesis_statement') or ''),
    ]).strip()

def _jaccard_similarity(left_text, right_text):
    left_tokens = _tokenize_autoresearch_text(left_text)
    right_tokens = _tokenize_autoresearch_text(right_text)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))

def _to_float(value, default_value=0.0):
    try:
        return float(value)
    except Exception:
        return default_value

def _build_prior_cycle_digest(history):
    digest = []
    for cycle in history:
        digest.append({
            'iteration': cycle.get('iteration'),
            'selected_candidates': [
                {
                    'hypothesis_id': candidate.get('hypothesis_id', ''),
                    'title': candidate.get('title', ''),
                    'hypothesis_statement': candidate.get('hypothesis_statement', ''),
                    'total_score': candidate.get('total_score'),
                }
                for candidate in cycle.get('selected_candidates', [])
            ]
        })
    return digest

def _compute_autoresearch_stop_decision(selected_candidates, history, cycle_index, max_iterations, execution_mode, agent_recommendation, agent_reason):
    previous_candidates = _flatten_history_candidates(history)
    previous_top_score = 0.0
    if history:
        last_cycle_candidates = history[-1].get('selected_candidates', [])
        if last_cycle_candidates:
            previous_top_score = max(_to_float(candidate.get('total_score')) for candidate in last_cycle_candidates)

    top_score = max((_to_float(candidate.get('total_score')) for candidate in selected_candidates), default=0.0)
    avg_score = (
        sum(_to_float(candidate.get('total_score')) for candidate in selected_candidates) / max(1, len(selected_candidates))
        if selected_candidates else 0.0
    )
    score_gain = top_score - previous_top_score if history else top_score

    overlap_scores = []
    for candidate in selected_candidates:
        current_signature = _candidate_signature(candidate)
        best_overlap = 0.0
        for previous_candidate in previous_candidates:
            previous_signature = _candidate_signature(previous_candidate)
            best_overlap = max(best_overlap, _jaccard_similarity(current_signature, previous_signature))
        overlap_scores.append(best_overlap)
    average_overlap = sum(overlap_scores) / max(1, len(overlap_scores)) if overlap_scores else 0.0

    minimum_cycles_before_stop = 2
    stop_votes = 0
    reasons = []

    normalized_agent_recommendation = str(agent_recommendation or '').strip().lower()
    if cycle_index >= max_iterations:
        return {
            'should_continue': False,
            'stop_reason': f'Max iteration {max_iterations} reached.',
            'stop_metrics': {
                'top_score': round(top_score, 2),
                'avg_score': round(avg_score, 2),
                'score_gain': round(score_gain, 2),
                'average_overlap': round(average_overlap, 3),
                'agent_recommendation': normalized_agent_recommendation or 'n/a',
            }
        }

    if normalized_agent_recommendation == 'stop':
        stop_votes += 1
        reasons.append(agent_reason or 'Validation agent advised stopping.')

    if history and average_overlap >= 0.58:
        stop_votes += 1
        reasons.append('New candidates overlap too much with previous cycles.')

    if history and cycle_index >= minimum_cycles_before_stop and score_gain < 4.0:
        stop_votes += 1
        reasons.append('Top candidate quality is no longer improving enough.')

    if cycle_index >= 3 and avg_score < 74.0:
        stop_votes += 1
        reasons.append('Average shortlisted quality dropped below the target band.')

    if cycle_index < minimum_cycles_before_stop:
        should_continue = True
        stop_reason = f'Force at least {minimum_cycles_before_stop} cycles before stopping.'
    else:
        should_continue = stop_votes < 2
        if should_continue:
            stop_reason = 'Continue: the cycle still adds enough novelty or score gain.'
        else:
            stop_reason = " ".join(reasons) if reasons else 'Auto stop triggered by the convergence rule.'

    return {
        'should_continue': should_continue,
        'stop_reason': stop_reason,
        'stop_metrics': {
            'top_score': round(top_score, 2),
            'avg_score': round(avg_score, 2),
            'score_gain': round(score_gain, 2),
            'average_overlap': round(average_overlap, 3),
            'agent_recommendation': normalized_agent_recommendation or 'n/a',
        }
    }

def _autoresearch_stage_snapshot(brainstorming_status='pending', hypothesis_status='pending', validation_status='pending',
                                 brainstorming_task='Waiting to start',
                                 hypothesis_task='Waiting to start',
                                 validation_task='Waiting to start'):
    stage_progress = {
        'pending': 0,
        'active': 55,
        'completed': 100,
        'failed': 100,
    }
    return [
        {
            'key': 'brainstorming',
            'label': 'Brainstorming Agent',
            'status': brainstorming_status,
            'progress': stage_progress.get(brainstorming_status, 0),
            'task': brainstorming_task,
        },
        {
            'key': 'hypothesis',
            'label': 'Hypothesis Agent',
            'status': hypothesis_status,
            'progress': stage_progress.get(hypothesis_status, 0),
            'task': hypothesis_task,
        },
        {
            'key': 'validation',
            'label': 'Validation Agent',
            'status': validation_status,
            'progress': stage_progress.get(validation_status, 0),
            'task': validation_task,
        },
    ]

@csrf_exempt
def get_optiresearch_state(request):
    requested_survey_id = request.GET.get('survey_id', '').strip()
    developer_payload = _get_optiresearch_developer_payload()
    if developer_payload:
        payload = developer_payload
    else:
        payload = _get_optiresearch_payload_for_survey(requested_survey_id) if requested_survey_id else _find_latest_optiresearch_payload()

    if payload is None:
        return JsonResponse({
            'enabled': False,
            'survey_id': requested_survey_id,
            'latex_pdf_url': '',
            'pdf_url': '',
            'chat_enabled': False,
            'message': 'OptiResearch is unavailable until the LaTeX PDF is generated.'
        })

    if payload.get('developer_override') and not payload.get('chat_enabled'):
        payload['message'] = 'OptiResearch developer mode is ready in reader-only mode.'
    else:
        payload['message'] = (
            'OptiResearch is ready.'
            if payload['enabled']
            else 'OptiResearch is unavailable until the LaTeX PDF is generated.'
        )
    return JsonResponse(payload)

@csrf_exempt
def ask_optiresearch(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)

    try:
        data = json.loads(request.body.decode('utf-8'))
    except Exception:
        return JsonResponse({'error': 'Invalid JSON payload'}, status=400)

    survey_id = str(data.get('survey_id', '')).strip()
    question = str(data.get('question', '')).strip()

    if not survey_id:
        return JsonResponse({'error': 'survey_id is required'}, status=400)
    if not question:
        return JsonResponse({'error': 'question is required'}, status=400)

    context, source_path = _load_optiresearch_context(survey_id)
    if not context.strip():
        return JsonResponse({'error': 'Survey content is not available for OptiResearch yet.'}, status=404)

    sections = _build_optiresearch_sections(context)
    selected_sections = _select_optiresearch_sections(question, sections, max_sections=4)
    if not selected_sections:
        selected_sections = [{
            'section_path': 'Document Overview',
            'title_level': 0,
            'content': context[:6000],
            'score': 0,
            'index': 0
        }]

    section_blocks = []
    fallback_sources = []
    for section in selected_sections:
        fallback_sources.append(section['section_path'])
        trimmed_content = section['content'][:3500]
        section_blocks.append(
            f"[Section: {section['section_path']}]\n{trimmed_content}"
        )
    selected_context = "\n\n".join(section_blocks)

    prompt = f"""
You are OptiResearch, an academic reading assistant.
Answer the user's question strictly based on the provided survey sections.

Rules:
- If the answer is directly supported, answer clearly and concisely.
- If the content is insufficient, say so explicitly instead of inventing details.
- Choose sources only from the provided section paths.
- Keep the answer under 220 words.
- Return JSON only in this exact shape:
  {{"answer":"...","sources":["Section A","Section B"]}}
- Include 1 to 3 section paths in `sources` when evidence exists.

Survey sections:
{selected_context}

User question:
{question}
"""

    try:
        client = getQwenClient()
        raw_answer = generateResponse(client, prompt)
        answer, citations = _parse_optiresearch_response(raw_answer, fallback_sources)
    except Exception as e:
        return JsonResponse({'error': f'OptiResearch failed to answer the question: {str(e)}'}, status=500)

    return JsonResponse({
        'success': True,
        'survey_id': survey_id,
        'answer': answer,
        'citations': citations,
        'matched_sections': [
            {
                'section_path': section['section_path'],
                'score': section['score']
            }
            for section in selected_sections
        ],
        'source_path': source_path
    })

@csrf_exempt
@timeout_handler(900)
def run_autoresearch_sync(request, operation_id=None):
    operation_id = operation_id or getattr(request, 'operation_id', f"autoresearch_{int(time.time())}")
    update_progress(
        operation_id,
        5,
        "Preparing AutoResearch cycle...",
        {
            'agent_stages': _autoresearch_stage_snapshot(
                brainstorming_status='active',
                brainstorming_task='Preparing survey context',
            )
        }
    )
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)

    try:
        data = json.loads(request.body.decode('utf-8'))
    except Exception:
        return JsonResponse({'error': 'Invalid JSON payload'}, status=400)

    survey_id = str(data.get('survey_id', '')).strip()
    execution_mode = 'auto'
    cycle_index = _coerce_positive_int(data.get('cycle_index', 1), default_value=1, min_value=1, max_value=20)
    max_iterations = _coerce_positive_int(data.get('max_iterations', 5), default_value=5, min_value=2, max_value=5)
    idea_count = _coerce_positive_int(data.get('idea_count', 10), default_value=10, min_value=3, max_value=15)
    candidate_count = _coerce_positive_int(data.get('candidate_count', 3), default_value=3, min_value=1, max_value=10)
    candidate_count = min(candidate_count, idea_count)
    history = _coerce_autoresearch_history(data.get('history', []))

    if not survey_id:
        return JsonResponse({'error': 'survey_id is required'}, status=400)

    context, source_path = _load_optiresearch_context(survey_id)
    if not context.strip():
        return JsonResponse({'error': 'Survey content is not available for AutoResearch yet.'}, status=404)

    survey_title = _load_autoresearch_survey_title(survey_id, context)
    citation_data_list = _load_autoresearch_citation_data(survey_id)
    autoresearch_context = _build_autoresearch_context(context)
    prior_cycle_digest = _build_prior_cycle_digest(history)
    update_progress(
        operation_id,
        20,
        "Brainstorming agent is generating ideas...",
        {
            'agent_stages': _autoresearch_stage_snapshot(
                brainstorming_status='active',
                brainstorming_task='Generating idea pool from review',
                hypothesis_task='Queued behind idea generation',
                validation_task='Waiting for structured hypotheses',
            )
        }
    )

    brainstorming_prompt = f"""
You are the Brainstorming Agent in AutoResearch.
Your job is to propose exactly {idea_count} research ideas for cycle {cycle_index} based only on the survey review below.

Rules:
- Focus on novel or high-value directions implied by the survey.
- Do not perform feasibility filtering yet.
- Keep each idea concrete and distinct.
- Avoid repeating or lightly rephrasing ideas that already appeared in earlier shortlisted candidates.
- Return JSON only in this exact shape:
{{
  "ideas": [
    {{
      "id": "I1",
      "title": "Short title",
      "core_insight": "One concise paragraph",
      "novelty_basis": "Why this seems new from the review",
      "why_now": "Why this matters now"
    }}
  ]
}}

Survey title: {survey_title}

Prior shortlisted candidates from earlier cycles:
{json.dumps(prior_cycle_digest, ensure_ascii=False, indent=2)}

Survey review context:
{autoresearch_context}
"""

    try:
        client = getQwenClient()
        brainstorming_raw = generateResponse(client, brainstorming_prompt)
        brainstorming_payload = _extract_json_payload(brainstorming_raw, root_key='ideas')
        ideas = brainstorming_payload.get('ideas', [])
        if not isinstance(ideas, list) or not ideas:
            raise ValueError('Brainstorming agent returned no ideas')
        ideas = ideas[:idea_count]
        for index, idea in enumerate(ideas, start=1):
            idea['id'] = str(idea.get('id') or f'I{index}')
            idea['title'] = str(idea.get('title') or f'Idea {index}').strip()
            idea['core_insight'] = str(idea.get('core_insight') or idea.get('description') or '').strip()
            idea['novelty_basis'] = str(idea.get('novelty_basis') or '').strip()
            idea['why_now'] = str(idea.get('why_now') or '').strip()
    except Exception as e:
        update_progress(
            operation_id,
            -1,
            f"Brainstorming agent failed: {str(e)}",
            {
                'agent_stages': _autoresearch_stage_snapshot(
                    brainstorming_status='failed',
                    brainstorming_task=f'Failed: {str(e)}',
                )
            }
        )
        return JsonResponse({'error': f'Brainstorming agent failed: {str(e)}'}, status=500)

    evidence_packs = _build_idea_evidence_packs(ideas, citation_data_list)
    update_progress(
        operation_id,
        52,
        "Hypothesis agent is structuring hypothesis cards...",
        {
            'agent_stages': _autoresearch_stage_snapshot(
                brainstorming_status='completed',
                brainstorming_task=f'Generated {len(ideas)} ideas',
                hypothesis_status='active',
                hypothesis_task='Converting ideas into fixed-schema cards',
                validation_task='Waiting for scored candidates',
            )
        }
    )

    hypothesis_prompt = f"""
You are the Hypothesis Agent in AutoResearch.
Transform each brainstormed idea into one structured, falsifiable hypothesis.
You must ground each hypothesis in the provided uploaded-paper evidence pack when evidence exists.

Rules:
- Produce exactly one hypothesis per idea.
- Use only citation source names that appear inside each idea's evidence pack.
- Keep the fields concise and specific.
- Avoid semantic duplication with prior shortlisted candidates.
- Return JSON only in this exact shape:
{{
  "hypotheses": [
    {{
      "idea_id": "I1",
      "hypothesis_id": "H1",
      "title": "Card title",
      "research_gap": "Gap statement",
      "hypothesis_statement": "If ..., then ... because ...",
      "mechanism": "Underlying logic",
      "test_plan": "How to test it",
      "expected_signal": "What result would support it",
      "evidence_reasoning": "How the uploaded papers motivate it",
      "cited_papers": ["Paper A", "Paper B"]
    }}
  ]
}}

Survey title: {survey_title}

Prior shortlisted candidates from earlier cycles:
{json.dumps(prior_cycle_digest, ensure_ascii=False, indent=2)}

Idea and evidence packs:
{json.dumps(evidence_packs, ensure_ascii=False, indent=2)}
"""

    try:
        hypothesis_raw = generateResponse(client, hypothesis_prompt)
        hypothesis_payload = _extract_json_payload(hypothesis_raw, root_key='hypotheses')
        hypotheses = hypothesis_payload.get('hypotheses', [])
        if not isinstance(hypotheses, list) or not hypotheses:
            raise ValueError('Hypothesis agent returned no hypotheses')
        hypotheses = _attach_hypothesis_evidence(hypotheses[:len(ideas)], evidence_packs)
    except Exception as e:
        update_progress(
            operation_id,
            -1,
            f"Hypothesis agent failed: {str(e)}",
            {
                'agent_stages': _autoresearch_stage_snapshot(
                    brainstorming_status='completed',
                    brainstorming_task=f'Generated {len(ideas)} ideas',
                    hypothesis_status='failed',
                    hypothesis_task=f'Failed: {str(e)}',
                )
            }
        )
        return JsonResponse({'error': f'Hypothesis agent failed: {str(e)}'}, status=500)

    update_progress(
        operation_id,
        78,
        "Validation agent is scoring and ranking candidates...",
        {
            'agent_stages': _autoresearch_stage_snapshot(
                brainstorming_status='completed',
                brainstorming_task=f'Generated {len(ideas)} ideas',
                hypothesis_status='completed',
                hypothesis_task=f'Built {len(hypotheses)} hypothesis cards',
                validation_status='active',
                validation_task='Ranking and deciding whether to continue',
            )
        }
    )

    validation_prompt = f"""
You are the Validation Agent in AutoResearch, acting as a rigorous reviewer.
Review the hypotheses and rank them for a single research cycle.

Scoring dimensions:
- novelty: 1-10
- literature_grounding: 1-10
- clarity: 1-10
- potential_impact: 1-10
- total_score: 0-100

Rules:
- Rank all hypotheses from strongest to weakest.
- Select the best {candidate_count} candidates.
- Keep reviewer summaries short and specific.
- If this is auto mode, recommend `continue` only when the new shortlist is still materially novel and meaningfully stronger or different from earlier cycles.
- Return JSON only in this exact shape:
{{
  "review_summary": "Overall summary",
  "continue_recommendation": "continue",
  "continue_reason": "Why continue or stop",
  "ranked_hypotheses": [
    {{
      "hypothesis_id": "H1",
      "rank": 1,
      "novelty": 8,
      "literature_grounding": 7,
      "clarity": 9,
      "potential_impact": 8,
      "total_score": 82,
      "reviewer_summary": "Short reviewer comment"
    }}
  ],
  "selected_candidate_ids": ["H1", "H3", "H2"]
}}

Structured hypotheses:
{json.dumps(hypotheses, ensure_ascii=False, indent=2)}

Prior shortlisted candidates from earlier cycles:
{json.dumps(prior_cycle_digest, ensure_ascii=False, indent=2)}

Execution mode: {execution_mode}
Cycle index: {cycle_index}
Max iterations: {max_iterations}
"""

    try:
        validation_raw = generateResponse(client, validation_prompt)
        validation_payload = _extract_json_payload(validation_raw)
        ranked_hypotheses = validation_payload.get('ranked_hypotheses', [])
        selected_candidate_ids = validation_payload.get('selected_candidate_ids', [])
        review_summary = str(validation_payload.get('review_summary') or '').strip()
        continue_recommendation = str(validation_payload.get('continue_recommendation') or '').strip().lower()
        continue_reason = str(validation_payload.get('continue_reason') or '').strip()

        if not isinstance(ranked_hypotheses, list) or not ranked_hypotheses:
            raise ValueError('Validation agent returned no rankings')
        if isinstance(selected_candidate_ids, str):
            selected_candidate_ids = [selected_candidate_ids]
        if not isinstance(selected_candidate_ids, list):
            selected_candidate_ids = []

        merged_rankings, selected_candidates = _merge_autoresearch_rankings(
            hypotheses,
            ranked_hypotheses,
            selected_candidate_ids[:candidate_count]
        )
        if not selected_candidates:
            selected_candidates = merged_rankings[:candidate_count]
    except Exception as e:
        update_progress(
            operation_id,
            -1,
            f"Validation agent failed: {str(e)}",
            {
                'agent_stages': _autoresearch_stage_snapshot(
                    brainstorming_status='completed',
                    brainstorming_task=f'Generated {len(ideas)} ideas',
                    hypothesis_status='completed',
                    hypothesis_task=f'Built {len(hypotheses)} hypothesis cards',
                    validation_status='failed',
                    validation_task=f'Failed: {str(e)}',
                )
            }
        )
        return JsonResponse({'error': f'Validation agent failed: {str(e)}'}, status=500)

    stop_decision = _compute_autoresearch_stop_decision(
        selected_candidates,
        history,
        cycle_index,
        max_iterations,
        execution_mode,
        continue_recommendation,
        continue_reason
    )

    result_payload = {
        'success': True,
        'survey_id': survey_id,
        'survey_title': survey_title,
        'source_path': source_path,
        'cycle': {
            'iteration': cycle_index,
            'execution_mode': execution_mode,
            'max_iterations': max_iterations,
            'idea_count_requested': idea_count,
            'candidate_count_requested': candidate_count,
            'ideas': ideas,
            'hypotheses': hypotheses,
            'ranked_hypotheses': merged_rankings,
            'selected_candidates': selected_candidates,
            'review_summary': review_summary,
            'citation_pool_size': len(citation_data_list),
            'continue_recommendation': continue_recommendation or 'continue',
            'continue_reason': continue_reason,
            'should_continue': stop_decision['should_continue'],
            'stop_reason': stop_decision['stop_reason'],
            'stop_metrics': stop_decision['stop_metrics'],
        }
    }
    update_progress(
        operation_id,
        100,
        f"Cycle {cycle_index} completed successfully.",
        {
            'agent_stages': _autoresearch_stage_snapshot(
                brainstorming_status='completed',
                brainstorming_task=f'Generated {len(ideas)} ideas',
                hypothesis_status='completed',
                hypothesis_task=f'Built {len(hypotheses)} hypothesis cards',
                validation_status='completed',
                validation_task=f"Selected {len(selected_candidates)} candidates",
            )
        }
    )
    return JsonResponse(result_payload)

@csrf_exempt
def run_autoresearch(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)

    operation_id = f"autoresearch_{int(time.time())}"
    request.operation_id = operation_id
    success = task_manager.start_task(
        operation_id,
        run_autoresearch_sync,
        request,
        operation_id
    )
    if not success:
        return JsonResponse({'error': 'AutoResearch task already running'}, status=409)

    return JsonResponse({
        'operation_id': operation_id,
        'status': 'started',
        'message': 'AutoResearch cycle started successfully. Use the operation_id to check progress.',
        'progress_url': f'/get_operation_progress/?operation_id={operation_id}'
    })

@csrf_exempt
@timeout_handler(1800)  # 15鍒嗛挓瓒呮椂
def upload_refs_sync(request):
    """同步版本的文件上传处理函数"""
    start_time = time.time()
    operation_id = getattr(request, 'operation_id', f"upload_{int(start_time)}")
    print(f"[DEBUG] upload_refs_sync started with operation_id: {operation_id}")
    update_progress(operation_id, 0, "Starting file upload...")
    
    RECOMMENDED_PDF_DIR = os.path.join("src", "static", "data", "pdf", "recommend_pdfs")
    if request.method == 'POST':
        try:
            # 检查是否有上传的文件（新方式使用file_paths，旧方式使用FILES）
            has_uploaded_files = False
            if hasattr(request, 'file_paths') and request.file_paths:
                has_uploaded_files = True
            elif request.FILES:
                has_uploaded_files = True
            
            if not has_uploaded_files:
                if not os.path.exists(RECOMMENDED_PDF_DIR):
                    return JsonResponse({'error': 'No file part'}, status=400)
        
            update_progress(operation_id, 10, "Initializing upload process...")
            
            is_valid_submission = True
            has_label_id = False
            has_ref_link = False

            filenames = []
            collection_names = []
            filesizes = []
            
            # 创建统一的file_dict，兼容新旧两种方式
            file_dict = {}
            
            # 如果使用新的file_paths方式
            if hasattr(request, 'file_paths') and request.file_paths:
                update_progress(operation_id, 15, "Loading files from disk...")
                for file_path in request.file_paths:
                    file_name = os.path.basename(file_path)
                    
                    # 从磁盘读取文件内容
                    with open(file_path, 'rb') as f:
                        file_content = BytesIO(f.read())
                    
                    # 创建类似Django上传文件的对象
                    uploaded_file = InMemoryUploadedFile(
                        file_content,
                        field_name="file",
                        name=file_name,
                        content_type="application/pdf",
                        size=os.path.getsize(file_path),
                        charset=None
                    )
                    file_dict[file_name] = uploaded_file
            else:
                # 使用传统的request.FILES方式
                file_dict = request.FILES.copy()

            global Global_survey_id
            global Global_test_flag
            global Global_collection_names
            global Global_survey_title
            global Global_file_names
            global Global_citation_data
            global Global_description_list
            global Global_collection_names_clustered
            global Global_ref_list
            global Global_cluster_names
            global Global_category_label

            Global_collection_names = []
            Global_file_names = []
            Global_citation_data = []
            Global_description_list = []
            Global_collection_names_clustered = []
            Global_ref_list = []
            Global_cluster_names = []
            Global_category_label = []

            Global_survey_title = request.POST.get('topic', False)
            process_pdf_mode = request.POST.get('mode', False)
            
            update_progress(operation_id, 20, "Processing recommended PDFs...")
            
            if os.path.exists(RECOMMENDED_PDF_DIR):
                for pdf_name in os.listdir(RECOMMENDED_PDF_DIR):
                    if pdf_name.endswith(".pdf"):
                        pdf_path = os.path.join(RECOMMENDED_PDF_DIR, pdf_name)

                        pdf_content = BytesIO()
                        with open(pdf_path, 'rb') as f:
                            shutil.copyfileobj(f, pdf_content)
                        pdf_content.seek(0)

                        uploaded_pdf = InMemoryUploadedFile(
                            pdf_content,
                            field_name="file",
                            name=pdf_name,
                            content_type="application/pdf",
                            size=os.path.getsize(pdf_path),
                            charset=None
                        )

                        file_dict[f"recommend_{pdf_name}"] = uploaded_pdf

                shutil.rmtree(RECOMMENDED_PDF_DIR)

            update_progress(operation_id, 30, "Setting up survey ID...")
            
            # 始终生成新的survey_id，无论前端传递什么参数
            Global_survey_id = 'test_4' if Global_test_flag else generate_uid()
            uid_str = Global_survey_id
            print(f"[DEBUG] Generated new survey_id: {Global_survey_id}")

            update_progress(operation_id, 40, "Processing uploaded files...")
            
            total_files = len(file_dict)
            processed_files = 0
            
            for file_name in file_dict:
                file = file_dict[file_name]
                if not file.name:
                    continue
                if file:
                    try:
                        sanitized_filename = sanitize_filename_py(os.path.splitext(file.name)[0])
                        file_extension = os.path.splitext(file.name)[1].lower()
                        if sanitized_filename in filenames:
                            continue
                        sanitized_filename = f"{sanitized_filename}{file_extension}"

                        file_path = os.path.join('src', 'static', 'data', 'pdf', Global_survey_id, sanitized_filename)
                        if default_storage.exists(file_path):
                            default_storage.delete(file_path)
                        
                        saved_file_name = default_storage.save(file_path, file)
                        file_size = round(float(file.size) / 1024000, 2)

                        collection_name, processed_file = process_file(saved_file_name, Global_survey_id, process_pdf_mode)
                        Global_collection_names.append(collection_name)
                        Global_file_names.append(processed_file)
                        filenames.append(processed_file)
                        filesizes.append(file_size)
                        
                        processed_files += 1
                        progress = 40 + (processed_files / total_files) * 30
                        update_progress(operation_id, progress, f"Processed {processed_files}/{total_files} files")
                        
                    except Exception as e:
                        print(f"Error processing file {file_name}: {e}")
                        continue

            update_progress(operation_id, 70, "Generating JSON data...")

            # 清理预处理 markdown 临时目录
            recommend_md_dir = os.path.join("src", "static", "data", "pdf", "recommend_pdfs_md")
            if os.path.exists(recommend_md_dir):
                shutil.rmtree(recommend_md_dir)
                print("Cleaned up recommend_pdfs_md directory.")

            new_file_name = Global_survey_id
            csvfile_name = new_file_name + '.'+ file_name.split('.')[-1]

            json_data_pd = pd.DataFrame()
            json_files_path = f'./src/static/data/txt/{Global_survey_id}/*.json'
            json_files = glob.glob(json_files_path)

            # Dictionary to hold title and abstract pairs
            title_abstract_dict = {}
            filtered_json_files = [
                json_file for json_file in json_files
                if os.path.splitext(os.path.basename(json_file))[0] in filenames
            ]
            ref_paper_num = len(filtered_json_files)
            print(f'The length of the json files is {ref_paper_num}')

            update_progress(operation_id, 80, "Processing JSON files...")
            
            # Iterate over each JSON file
            for i, file_path in enumerate(filtered_json_files):
                try:
                    with open(file_path, 'r', encoding= "utf-8") as file:
                        data = json.load(file)

                        # Extract necessary information
                        title = data.get("title", "")
                        abstract = data.get("abstract", "")
                        authors = data.get("authors", "")
                        introduction = data.get("introduction", "")

                        new_data = {
                            "reference paper title": title,
                            "reference paper citation information (can be collected from Google scholar/DBLP)": authors,
                            "reference paper abstract (Please copy the text AND paste here)": abstract,
                            "reference paper introduction (Please copy the text AND paste here)": introduction,
                            "reference paper doi link (optional)": "",
                            "reference paper category label (optional)": ""
                        }

                        new_data_df = pd.DataFrame([new_data])
                        json_data_pd = pd.concat([json_data_pd, new_data_df], ignore_index=True)
                        title_abstract_dict[title] = abstract
                        
                        progress = 80 + (i / len(filtered_json_files)) * 10
                        update_progress(operation_id, progress, f"Processing JSON {i+1}/{len(filtered_json_files)}")
                        
                except Exception as e:
                    print(f"Error processing JSON file {file_path}: {e}")
                    continue

            update_progress(operation_id, 90, "Finalizing data...")
            
            input_pd = json_data_pd
            output_path = f'./src/static/data/info/{Global_survey_id}/title_abstract_pairs.json'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, 'w', encoding="utf-8") as outfile:
                json.dump(title_abstract_dict, outfile, indent=4, ensure_ascii=False)

            print(f'Title-abstract pairs have been saved to {output_path}')

            # 初始化 output_tsv_filename，确保它总是被定义
            output_tsv_filename = "./src/static/data/tsv/" + new_file_name + '.tsv'

            if ref_paper_num>0:

                print('The filenames are:', filenames)
                print('The json files are:', filtered_json_files)
                input_pd['ref_title'] = [filename for filename in filenames]
                input_pd["ref_context"] = [""]*ref_paper_num
                input_pd["ref_entry"] = input_pd["reference paper citation information (can be collected from Google scholar/DBLP)"]
                input_pd["abstract"] = input_pd["reference paper abstract (Please copy the text AND paste here)"].apply(lambda x: clean_str(x) if len(str(x))>0 else 'Invalid abstract')
                input_pd["intro"] = input_pd["reference paper introduction (Please copy the text AND paste here)"].apply(lambda x: clean_str(x) if len(str(x))>0 else 'Invalid introduction')

                input_pd["label"] = input_pd["reference paper category label (optional)"].apply(lambda x: str(x) if len(str(x))>0 else '')

                try:
                    output_df = input_pd[["ref_title","ref_context","ref_entry","abstract","intro"]]

                    if has_label_id == True:
                        output_df["label"]=input_pd["label"]
                    else:
                        output_df["label"]=[""]*input_pd.shape[0]

                    output_df.to_csv(output_tsv_filename, sep='\t')
                except Exception as e:
                    print(f"Cannot output tsv: {e}")
                    is_valid_submission = False

            else:
                is_valid_submission = False

            update_progress(operation_id, 100, "Upload completed successfully!")
            
            if is_valid_submission == True:
                ref_ids = [i for i in range(output_df['ref_title'].shape[0])]
                ref_list = {
                            'ref_ids':ref_ids,
                            'is_valid_submission':is_valid_submission,
                            "uid":uid_str,
                            "tsv_filename":output_tsv_filename,
                            # 'topic_words': clusters_topic_words,
                            'filenames': filenames,
                            'filesizes': filesizes,
                            'survey_id': Global_survey_id,
                            'operation_id': operation_id
                            }

            else:
                ref_list = {'ref_ids':[],'is_valid_submission':is_valid_submission,"uid":uid_str,"tsv_filename":output_tsv_filename, 'filenames': filenames, 'filesizes': filesizes, 'survey_id': Global_survey_id, 'operation_id': operation_id}
            ref_list = json.dumps(ref_list)
            print("--- %s seconds used in processing files ---" % (time.time() - start_time))
            
            # 清理临时文件（如果使用了file_paths方式）
            if hasattr(request, 'file_paths') and request.file_paths:
                try:
                    temp_dir = os.path.dirname(request.file_paths[0])
                    if 'tmp_upload' in temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                        print(f"[DEBUG] Cleaned up temporary upload directory: {temp_dir}")
                except Exception as e:
                    print(f"[DEBUG] Failed to clean up temporary directory: {e}")
            
            return HttpResponse(ref_list)
            
        except TimeoutError as e:
            update_progress(operation_id, -1, f"Upload timed out: {str(e)}")
            # 清理临时文件
            if hasattr(request, 'file_paths') and request.file_paths:
                try:
                    temp_dir = os.path.dirname(request.file_paths[0])
                    if 'tmp_upload' in temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                except:
                    pass
            return JsonResponse({'error': f'Upload operation timed out: {str(e)}'}, status=408)
        except Exception as e:
            update_progress(operation_id, -1, f"Upload failed: {str(e)}")
            # 清理临时文件
            if hasattr(request, 'file_paths') and request.file_paths:
                try:
                    temp_dir = os.path.dirname(request.file_paths[0])
                    if 'tmp_upload' in temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                except:
                    pass
            return JsonResponse({'error': f'Upload failed: {str(e)}'}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def generate_arxiv_query(request):
    """
    搜索论文的API端点

    支持两种数据源：
    1. arxiv - 从arXiv API搜索论文（默认）
    2. local - 从本地PDF数据库搜索

    请求参数：
    - topic: 搜索主题（必需）
    - source: 数据源，可选 "arxiv" 或 "local"（默认 "arxiv"）
    - local_db_path: 本地PDF数据库根文件夹路径（source="local"时必需）
    """
    def search_arxiv_with_query(query, max_results=50):
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://export.arxiv.org/api/query?search_query={encoded_query}&start=0&max_results={max_results}&sortBy=submittedDate"

        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error fetching data with query: {query} | status code: {response.status_code}")
            return []

        try:
            root = ET.fromstring(response.text)
        except Exception as e:
            print("Error parsing XML:", e)
            return []

        ns = "{http://www.w3.org/2005/Atom}"
        entries = root.findall(f"{ns}entry")
        papers = []
        for entry in entries:
            title_elem = entry.find(f"{ns}title")
            title = title_elem.text.strip() if title_elem is not None else ""
            summary_elem = entry.find(f"{ns}summary")
            summary_text = summary_elem.text.strip() if summary_elem is not None else ""
            link_elem = entry.find(f"{ns}id")
            link_text = link_elem.text.strip() if link_elem is not None else ""
            arxiv_id = link_text.split('/')[-1]
            pdf_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            papers.append({
                "title": title,
                "summary": summary_text,
                "pdf_link": pdf_link,
                "arxiv_id": arxiv_id,
                "source": "arxiv"
            })

        return papers

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            topic = data.get('topic', '').strip()
            if not topic:
                return JsonResponse({'error': 'Topic is required.'}, status=400)

            # 获取数据源参数
            source = data.get('source', 'arxiv').lower().strip()
            local_db_path = data.get('local_db_path', '').strip()

            max_results = 50
            min_results = 10

            # Hybrid 模式：同时搜索 arXiv 和 JSON Vector DB，综合排序
            if source == 'hybrid':
                try:
                    print(f"[Hybrid] Searching arXiv and JSON Vector DB for: {topic}")

                    # 1. 搜索 JSON Vector DB
                    from .json_vector_db import get_db_instance
                    json_db = get_db_instance('src/static/local_pdfs')

                    json_papers = []
                    if json_db.index is None:
                        print("[Hybrid] Building JSON Vector DB index...")
                        json_db.build_index()

                    if json_db.index is not None:
                        json_results = json_db.search(topic, k=max_results)
                        for r in json_results:
                            json_papers.append({
                                "title": r["title"],
                                "summary": f"Similarity: {r['score']:.4f} | From: {r['json_source']}",
                                "pdf_link": r["pdf_path"],
                                "arxiv_id": r["paper_id"],
                                "source": "json_vec",
                                "score": r["score"],
                                "json_source": r["json_source"],
                                "hybrid_score": r["score"] * 0.5 + 0.5  # 将 0-1 分数映射到 0.5-1.0
                            })
                        print(f"[Hybrid] Found {len(json_papers)} papers from JSON Vector DB")

                    # 2. 搜索 arXiv
                    strict_query = generate_query_qwen(topic)
                    arxiv_papers = search_arxiv_with_query(strict_query, max_results=max_results)

                    # 为 arXiv 论文添加 hybrid_score（按排名递减）
                    for i, paper in enumerate(arxiv_papers):
                        # 前10名得 1.0，之后线性递减到 0.5
                        rank_score = max(0.5, 1.0 - (i / max_results) * 0.5)
                        paper["hybrid_score"] = rank_score
                        paper["score"] = rank_score

                    print(f"[Hybrid] Found {len(arxiv_papers)} papers from arXiv")

                    # 3. 合并结果（去重：相同标题优先保留 JSON Vector DB 的，因为更精准）
                    all_papers = {}

                    # 先加入 JSON Vector DB 的结果（优先）
                    for paper in json_papers:
                        title_key = paper["title"].lower().strip()
                        all_papers[title_key] = paper

                    # 再加入 arXiv 的结果（如果标题不重复）
                    for paper in arxiv_papers:
                        title_key = paper["title"].lower().strip()
                        if title_key not in all_papers:
                            all_papers[title_key] = paper

                    # 4. 按 hybrid_score 排序
                    merged_papers = sorted(all_papers.values(), key=lambda x: x.get("hybrid_score", 0), reverse=True)

                    # 5. 取前 max_results 个
                    final_papers = merged_papers[:max_results]

                    if len(final_papers) < min_results:
                        return JsonResponse({
                            'error': f'Not enough papers found. Found {len(final_papers)}, need at least {min_results}.',
                            'count': len(final_papers),
                        }, status=400)

                    print(f"[Hybrid] Total unique papers: {len(final_papers)} (JSON: {len(json_papers)}, arXiv: {len(arxiv_papers)})")

                    return JsonResponse({
                        "papers": final_papers,
                        "count": len(final_papers),
                        "source": "hybrid",
                        "breakdown": {
                            "json_vec": len(json_papers),
                            "arxiv": len(arxiv_papers),
                            "unique": len(final_papers)
                        }
                    }, status=200)

                except Exception as e:
                    import traceback
                    print(f"Error in hybrid search: {e}")
                    traceback.print_exc()
                    return JsonResponse({
                        'error': f'Hybrid search failed: {str(e)}',
                    }, status=500)

            # JSON Vector Database 模式（仅本地 JSON 数据库）
            if source == 'json_vec':
                json_folder = data.get('json_folder', 'src/static/local_pdfs').strip()

                if not os.path.exists(json_folder):
                    return JsonResponse({
                        'error': f'JSON folder does not exist: {json_folder}'
                    }, status=400)

                try:
                    print(f"[JSON Vector DB] Searching for: {topic}")

                    # 获取或创建数据库实例
                    from .json_vector_db import get_db_instance
                    db = get_db_instance(json_folder)

                    # 如果索引未构建，则构建
                    if db.index is None:
                        print("[JSON Vector DB] Building index...")
                        success = db.build_index()
                        if not success:
                            return JsonResponse({
                                'error': 'Failed to build JSON vector database index.',
                                'hint': 'Check if JSON files exist in the specified folder.'
                            }, status=500)

                    # 执行向量搜索
                    results = db.search(topic, k=max_results)

                    if len(results) < min_results:
                        return JsonResponse({
                            'error': f'Not enough papers found in JSON database. Found {len(results)}, need at least {min_results}.',
                            'count': len(results),
                        }, status=400)

                    # 转换为与 arxiv/local 兼容的格式
                    # pdf_link 指向本地 PDF 路径，下载时会复制文件
                    papers = []
                    for r in results:
                        papers.append({
                            "title": r["title"],
                            "summary": f"Similarity: {r['score']:.4f} | From: {r['json_source']}",
                            "pdf_link": r["pdf_path"],  # 本地 PDF 路径
                            "arxiv_id": r["paper_id"],
                            "source": "json_vec",
                            "score": r["score"],
                            "json_source": r["json_source"]
                        })

                    return JsonResponse({
                        "papers": papers,
                        "count": len(papers),
                        "source": "json_vec",
                        "json_folder": json_folder
                    }, status=200)

                except Exception as e:
                    import traceback
                    print(f"Error in JSON vector search: {e}")
                    traceback.print_exc()
                    return JsonResponse({
                        'error': f'JSON vector search failed: {str(e)}',
                        'hint': 'Make sure sentence-transformers and faiss are installed.'
                    }, status=500)

            # arXiv模式（默认）
            strict_query = generate_query_qwen(topic)
            papers_strict = search_arxiv_with_query(strict_query, max_results=max_results)

            total_papers = {paper["title"]: paper for paper in papers_strict}

            if len(total_papers) >= min_results:
                papers_list = list(total_papers.values())  # dict -> list

                return JsonResponse({
                    "papers": papers_list,
                    "count": len(papers_list),
                    "source": "arxiv"
                }, status=200)

            attempts = 0
            MAX_ATTEMPTS = 5
            current_query = strict_query

            while len(total_papers) < min_results and attempts < MAX_ATTEMPTS:
                # 生成更宽松的查询
                generic_query = generate_generic_query_qwen(current_query, topic)
                papers_generic = search_arxiv_with_query(generic_query, max_results=max_results)

                # 合并新结果
                new_count = 0
                for paper in papers_generic:
                    if paper["title"] not in total_papers:
                        total_papers[paper["title"]] = paper
                        new_count += 1

                attempts += 1
                current_query = generic_query

                if len(total_papers) >= min_results:
                    papers_list = list(total_papers.values())

                    return JsonResponse({
                        "papers": papers_list,
                        "count": len(papers_list),
                        "source": "arxiv"
                    }, status=200)

            return JsonResponse({
                'error': f'Not enough references found even after {attempts} attempts.',
                'count': len(total_papers),
            }, status=400)

        except Exception as e:
            import traceback
            print(f"Error in generate_arxiv_query: {e}")
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method.'}, status=405)

@csrf_exempt
@timeout_handler(1800)  # 30分钟超时
def download_pdfs_sync(request, operation_id=None):
    """同步版本的PDF下载函数"""
    def clean_filename(filename):
        filename = filename.strip()  # 去掉首尾空格和换行符
        filename = re.sub(r'[\\/*?:"<>|\n\r]', '', filename)  # 移除非法字符
        return filename
    
    operation_id = operation_id or getattr(request, 'operation_id', f"download_{int(time.time())}")
    print(f"[DEBUG] download_pdfs_sync started with operation_id: {operation_id}")
    update_progress(operation_id, 0, "Starting PDF downloads...")
    
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            pdf_links = data.get("pdf_links", [])
            pdf_titles = data.get("pdf_titles", [])  # PDF 标题列表
            print(f"Starting download of {len(pdf_links)} PDFs")

            if not pdf_links:
                update_progress(operation_id, -1, "No PDFs to download")
                return JsonResponse({"message": "No PDFs to download."}, status=400)

            base_dir = os.path.join(os.getcwd(), "src", "static", "data", "pdf", "recommend_pdfs")
            os.makedirs(base_dir, exist_ok=True)  # 确保文件夹存在

            downloaded_files = []
            failed_downloads = []
            
            update_progress(operation_id, 10, f"Preparing to download {len(pdf_links)} PDFs...")
            
            for i, pdf_url in enumerate(pdf_links):
                try:
                    print(f"Processing {i+1}/{len(pdf_links)}: {pdf_url}")
                    progress = 10 + (i / len(pdf_links)) * 80
                    update_progress(operation_id, progress, f"Processing PDF {i+1}/{len(pdf_links)}")

                    # 处理文件名，确保合法
                    sanitized_title = clean_filename(pdf_titles[i]) if i < len(pdf_titles) else f"file_{i}"
                    pdf_filename = os.path.join(base_dir, f"{sanitized_title}.pdf")

                    # 检查是否是本地文件路径
                    is_local_file = pdf_url.startswith("local_") or os.path.isfile(pdf_url) or (
                        len(pdf_url) > 3 and pdf_url[1:3] == ":\\"  # Windows 路径如 "C:\..."
                    )

                    if is_local_file or os.path.exists(pdf_url):
                        # 本地文件模式：复制文件
                        source_path = pdf_url

                        # 如果 pdf_url 是本地 ID 格式 (local_xxxxx)，需要查找实际路径
                        if pdf_url.startswith("local_"):
                            # 从请求中获取本地数据库路径（通过 sources 数组）
                            sources = data.get("sources", [])
                            local_db_path = ""
                            for j, src in enumerate(sources):
                                if j == i and src == "local":
                                    # 尝试从 local_db_path 参数获取
                                    local_db_path = data.get("local_db_path", "")
                                    break

                            # 尝试查找文件
                            if local_db_path:
                                source_path = get_local_pdf_path(pdf_url, local_db_path) or pdf_url

                        # 如果 pdf_url 是 json_vec 的 ID 格式 (json_xxxxx)
                        elif pdf_url.startswith("json_"):
                            sources = data.get("sources", [])
                            json_folder = ""
                            for j, src in enumerate(sources):
                                if j == i and src == "json_vec":
                                    json_folder = data.get("json_folder", "src/static/local_pdfs")
                                    break

                            # 从 JSON Vector DB 查找实际路径
                            if json_folder:
                                from .json_vector_db import get_db_instance
                                db = get_db_instance(json_folder)
                                paper_info = db.get_paper_by_id(pdf_url)
                                if paper_info:
                                    source_path = paper_info["pdf_path"]
                                else:
                                    source_path = pdf_url

                        # 处理相对路径（对于 json_vec 返回的相对路径）
                        if not os.path.isabs(source_path) and not os.path.exists(source_path):
                            # 可能是相对于 json_folder 的路径
                            json_folder = data.get("json_folder", "src/static/local_pdfs")
                            potential_path = os.path.join(json_folder, source_path)
                            if os.path.exists(potential_path):
                                source_path = potential_path

                        if os.path.exists(source_path):
                            # 复制本地文件
                            import shutil
                            shutil.copy2(source_path, pdf_filename)
                            file_size = os.path.getsize(pdf_filename)
                            downloaded_files.append(pdf_filename)
                            print(f"Success (local copy): {pdf_filename} ({file_size/1024/1024:.2f}MB)")

                            # 检查是否有预处理好的 MineRU markdown 文件夹
                            # MineRU 输出结构: {parent}/{pdf_name}/{pdf_name}/auto/{pdf_name}.md
                            pdf_base_name = os.path.splitext(os.path.basename(source_path))[0]
                            pdf_parent_dir = os.path.dirname(source_path)
                            pre_processed_md_dir = os.path.join(pdf_parent_dir, pdf_base_name, pdf_base_name)
                            pre_processed_md_file = os.path.join(pre_processed_md_dir, "auto", f"{pdf_base_name}.md")
                            print(f"[DEBUG] Checking pre-processed md: {pre_processed_md_file}, exists={os.path.exists(pre_processed_md_file)}")

                            if os.path.exists(pre_processed_md_file):
                                recommend_md_dir = os.path.join(os.getcwd(), "src", "static", "data", "pdf", "recommend_pdfs_md")
                                os.makedirs(recommend_md_dir, exist_ok=True)
                                # 用 sanitize_filename_py 处理目标名，和 upload_refs_sync 中的命名保持一致
                                dest_folder_name = sanitize_filename_py(sanitized_title)
                                dest_md_dir = os.path.join(recommend_md_dir, dest_folder_name)
                                if os.path.exists(dest_md_dir):
                                    shutil.rmtree(dest_md_dir)
                                shutil.copytree(pre_processed_md_dir, dest_md_dir)
                                print(f"Copied pre-processed markdown for {pdf_base_name} -> {dest_folder_name}")
                        else:
                            print(f"Local file not found: {source_path}")
                            failed_downloads.append({"url": pdf_url, "reason": "Local file not found"})
                    else:
                        # 远程下载模式
                        response = requests.get(
                            pdf_url,
                            stream=True,
                            timeout=(10, 60),
                            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                        )

                        if response.status_code == 200:
                            # 下载 PDF，添加文件大小检查
                            total_size = 0
                            max_size = 50 * 1024 * 1024  # 50MB 限制

                            with open(pdf_filename, "wb") as pdf_file:
                                for chunk in response.iter_content(chunk_size=8192):
                                    if chunk:
                                        total_size += len(chunk)
                                        if total_size > max_size:
                                            print(f"File too large, skipping: {pdf_url}")
                                            failed_downloads.append({"url": pdf_url, "reason": "File too large (>50MB)"})
                                            break
                                        pdf_file.write(chunk)

                            if total_size <= max_size:
                                downloaded_files.append(pdf_filename)
                                print(f"Success (download): {pdf_filename} ({total_size/1024/1024:.2f}MB)")
                            else:
                                # 删除部分下载的文件
                                if os.path.exists(pdf_filename):
                                    os.remove(pdf_filename)
                        else:
                            print(f"Failed to download {pdf_url}, status code: {response.status_code}")
                            failed_downloads.append({"url": pdf_url, "reason": f"HTTP {response.status_code}"})

                except requests.exceptions.Timeout:
                    print(f"Timeout downloading {pdf_url}")
                    failed_downloads.append({"url": pdf_url, "reason": "Timeout"})
                except requests.exceptions.ConnectionError:
                    print(f"Connection error downloading {pdf_url}")
                    failed_downloads.append({"url": pdf_url, "reason": "Connection error"})
                except Exception as e:
                    print(f"Error processing {pdf_url}: {e}")
                    failed_downloads.append({"url": pdf_url, "reason": str(e)})

            print(f"Download finished: {len(downloaded_files)} successful, {len(failed_downloads)} failed")
            update_progress(operation_id, 100, f"Download completed: {len(downloaded_files)} successful, {len(failed_downloads)} failed")
            
            # 构建响应消息
            message = f"Downloaded {len(downloaded_files)} PDFs successfully!"
            if failed_downloads:
                message += f" {len(failed_downloads)} downloads failed."
            
            result = {
                "message": message,
                "files": downloaded_files,
                "failed": failed_downloads,
                "success_count": len(downloaded_files),
                "total_count": len(pdf_links)
            }
            
            return JsonResponse(result)

        except json.JSONDecodeError:
            update_progress(operation_id, -1, "Invalid JSON data")
            return JsonResponse({"message": "Invalid JSON data."}, status=400)
        except Exception as e:
            print(f"Unexpected error in download_pdfs: {e}")
            update_progress(operation_id, -1, f"Error: {str(e)}")
            return JsonResponse({"message": "An error occurred.", "error": str(e)}, status=500)

    return JsonResponse({"message": "Invalid request method."}, status=405)

@csrf_exempt
def download_pdfs(request):
    """异步版本的PDF下载接口，立即返回operation_id避免Cloudflare 524超时"""
    if request.method == "POST":
        # 生成操作ID
        operation_id = f"download_{int(time.time())}"
        
        print(f"[DEBUG] Starting async download task: {operation_id}")
        
        # 启动异步任务
        success = task_manager.start_task(
            operation_id, 
            download_pdfs_sync, 
            request
        )
        
        if not success:
            print(f"[DEBUG] Task {operation_id} already running")
            return JsonResponse({'error': 'Download task already running'}, status=409)
        
        print(f"[DEBUG] Async task {operation_id} started successfully")
        
        # 立即返回operation_id，不等待处理完成
        return JsonResponse({
            'operation_id': operation_id,
            'status': 'started',
            'message': 'PDF download started successfully. Use the operation_id to check progress.',
            'progress_url': f'/get_operation_progress/?operation_id={operation_id}'
        })
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def annotate_categories(request):
    html = generateOutlineHTML_qwen(Global_survey_id)
    print("The outline has been parsed successfully.")
    return JsonResponse({'html': html})

@csrf_exempt
def get_topic(request):
    topic = request.POST.get('topics', False)
    references, ref_links, ref_ids = get_refs(topic)
    global Global_survey_id
    Global_survey_id = topic
    ref_list = {
        'references' : references,
        'ref_links'  : ref_links,
        'ref_ids'    : ref_ids
    }
    ref_list = json.dumps(ref_list)
    return HttpResponse(ref_list)

@csrf_exempt
@timeout_handler(1800)  # 30分钟超时
def automatic_taxonomy_sync(request, operation_id=None):
    """同步版本的自动分类函数"""
    operation_id = operation_id or getattr(request, 'operation_id', f"taxonomy_{int(time.time())}")
    update_progress(operation_id, 0, "Starting automatic taxonomy...")
    
    global Global_description_list, Global_df_selected, Global_cluster_names, Global_ref_list, Global_category_label, Global_collection_names_clustered, Global_cluster_num
    global Global_survey_id, Global_collection_names, Global_citation_data, Global_file_names, Global_survey_title
    
    try:
        update_progress(operation_id, 10, "Loading reference data...")
        
        if request.method == 'POST':
            # 处理不同的请求格式
            try:
                # 尝试解析 JSON 数据
                if request.content_type == 'application/json':
                    data = json.loads(request.body)
                    Global_cluster_num = data.get('Global_cluster_num', 5)
                    refs_data = data.get('refs', [])
                    query = data.get('taxonomy_standard', '')
                else:
                    # 处理 form-data 格式
                    Global_cluster_num = int(request.POST.get('Global_cluster_num', 5))
                    refs_data = request.POST.get('refs', '[]')
                    query = request.POST.get('taxonomy_standard', '')
                    
                    # 解析 refs 数据
                    if isinstance(refs_data, str):
                        try:
                            refs_data = json.loads(refs_data)
                        except json.JSONDecodeError:
                            refs_data = []
                    
            except (json.JSONDecodeError, ValueError) as e:
                # 如果解析失败，尝试 form-data
                try:
                    Global_cluster_num = int(request.POST.get('Global_cluster_num', 5))
                    refs_data = request.POST.get('refs', '[]')
                    query = request.POST.get('taxonomy_standard', '')
                    
                    # 解析 refs 数据
                    if isinstance(refs_data, str):
                        try:
                            refs_data = json.loads(refs_data)
                        except json.JSONDecodeError:
                            refs_data = []
                except Exception as parse_error:
                    print(f"Error parsing request data: {parse_error}")
                    return JsonResponse({'error': f'Invalid request format: {parse_error}'}, status=400)
            
            # 处理 refs 数据
            if isinstance(refs_data, list):
                ref_list = [int(item) for item in refs_data if str(item).isdigit()]
            else:
                ref_list = []
            
            print(f"Parsed ref_list: {ref_list}")
            print(f"Global_cluster_num: {Global_cluster_num}")
            print(f"Query: {query}")

            Global_citation_data = []
            Global_description_list = []
            
            update_progress(operation_id, 20, "Generating query patterns...")
            
            # 生成查询模式
            query_list = normalize_query_list(generate_sentence_patterns(query))
            
            update_progress(operation_id, 30, "Processing collections...")
            
            # 处理每个集合
            for name in Global_collection_names:
                context, citation_data = query_embeddings_new_new(name, query_list)
                Global_citation_data.extend(citation_data)
                
                description = generate(context, query, name)
                Global_description_list.append(description)
            
            update_progress(operation_id, 50, "Saving citation data...")
            
            # 保存引用数据
            citation_path = f'./src/static/data/info/{Global_survey_id}/citation_data.json'
            os.makedirs(os.path.dirname(citation_path), exist_ok=True)
            with open(citation_path, 'w', encoding="utf-8") as outfile:
                json.dump(Global_citation_data, outfile, indent=4, ensure_ascii=False)
            
            update_progress(operation_id, 60, "Updating TSV file...")
            
            # 更新 TSV 文件
            file_path = f'./src/static/data/tsv/{Global_survey_id}.tsv'
            with open(file_path, 'r', newline='', encoding='utf-8') as infile:
                reader = csv.reader(infile, delimiter='\t')
                rows = list(reader)
            
            if rows:
                headers = rows[0]
                headers.append('retrieval_result')
                
                updated_rows = [headers]
                for row, description in zip(rows[1:], Global_description_list):
                    row.append(description)
                    updated_rows.append(row)
                
                with open(file_path, 'w', newline='', encoding='utf-8') as outfile:
                    writer = csv.writer(outfile, delimiter='\t')
                    writer.writerows(updated_rows)
                
                print('Updated file has been saved to', file_path)
            else:
                print('Input file is empty.')
            
            Global_ref_list = ref_list
            
            print('Categorization survey id', Global_survey_id)
            
            update_progress(operation_id, 70, "Performing clustering...")
            
            # 执行聚类
            colors, category_label = Clustering_refs(n_clusters=Global_cluster_num)
            Global_category_label = category_label
            
            update_progress(operation_id, 80, "Processing cluster results...")
            
            # 处理聚类结果
            df_tmp = Global_df_selected.reset_index()
            df_tmp['index'] = df_tmp.index
            ref_titles = list(df_tmp.groupby(df_tmp['label'])['ref_title'].apply(list))
            ref_indexs = list(df_tmp.groupby(df_tmp['label'])['index'].apply(list))
            
            # 读取主题信息
            info = pd.read_json(f'./src/static/data/info/{Global_survey_id}/topic.json')
            category_label = info['KeyBERT'].to_list()
            category_label_summarized = []
            
            tsv_path = f'./src/static/data/tsv/{Global_survey_id}.tsv'
            
            cluster_num = Global_cluster_num
            category_label_summarized = generate_cluster_name_new(tsv_path, Global_survey_title, cluster_num)
            Global_cluster_names = category_label_summarized
            
            update_progress(operation_id, 90, "Generating final results...")
            
            # 准备返回数据
            cate_list = {
                'colors': colors,
                'category_label': category_label_summarized,
                'survey_id': Global_survey_id,
                'ref_titles': [[i.title() for i in j] for j in ref_titles],
                'ref_indexs': ref_indexs
            }
            print(cate_list)
            
            # 保存聚类信息
            cluster_info = {category_label_summarized[i]: ref_titles[i] for i in range(len(category_label_summarized))}
            for key, value in cluster_info.items():
                temp = [legal_pdf(i) for i in value]
                cluster_info[key] = temp
                Global_collection_names_clustered.append(temp)
            
            cluster_info_path = f'./src/static/data/info/{Global_survey_id}/cluster_info.json'
            with open(cluster_info_path, 'w', encoding="utf-8") as outfile:
                json.dump(cluster_info, outfile, indent=4, ensure_ascii=False)
            
            # 生成大纲
            outline_generator = OutlineGenerator(Global_df_selected, Global_cluster_names)
            outline_generator.get_cluster_info()
            messages, outline = outline_generator.generate_outline_qwen(Global_survey_title, Global_cluster_num)
            
            outline_json = {'messages': messages, 'outline': outline}
            output_path = TXT_PATH + Global_survey_id + '/outline.json'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding="utf-8") as outfile:
                json.dump(outline_json, outfile, indent=4, ensure_ascii=False)
            
            update_progress(operation_id, 100, "Automatic taxonomy completed successfully!")
            
            # 返回 JSON 字符串格式，与原函数保持一致
            cate_list_json = json.dumps(cate_list)
            return HttpResponse(cate_list_json)
        
        else:
            return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    except Exception as e:
        print(f"Error in automatic_taxonomy: {str(e)}")
        import traceback
        traceback.print_exc()
        update_progress(operation_id, -1, f"Error: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def save_updated_cluster_info(request):
    global Global_collection_names
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            survey_id = Global_survey_id
            updated_cate_list = data.get('updated_cate_list')
            ref_indexs = updated_cate_list.get("ref_indexs", [])
            rearranged_collection_names = [
                [Global_collection_names[index] for index in group] for group in ref_indexs
            ]
            updated_cate_list["collection_name"] = rearranged_collection_names

            if not survey_id or not updated_cate_list:
                return JsonResponse({"error": "Missing survey_id or updated_cate_list"}, status=400)

            save_dir = os.path.join('./src/static/data/info/', str(survey_id))
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'cluster_info_updated.json')

            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(updated_cate_list, f, ensure_ascii=False, indent=4)

            return JsonResponse({"message": "Cluster info updated and saved successfully!"}, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method. Only POST is allowed."}, status=405)

import os
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

@csrf_exempt
def save_outline(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            updated_outline = data.get('outline', [])

            outline_data = {
                "messages": [
                    {
                        "role": "system",
                        "content": "Finish the outline of the survey paper..."
                    },
                    {
                        "role": "user",
                        "content": "Finish the outline..."
                    }
                ],
                "outline": str(updated_outline)
            }

            file_path = os.path.join(settings.BASE_DIR, 'static', 'data', 'txt', Global_survey_id,'outline.json')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(outline_data, file, indent=4, ensure_ascii=False)
            
            html = generateOutlineHTML_qwen(Global_survey_id)

            return JsonResponse({"status": "success", "html": html})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)
    else:
        return JsonResponse({"status": "error", "message": "Invalid request method"}, status=405)

@csrf_exempt
def select_sections(request):
    sections = request.POST
    survey = {}

    for k,v in sections.items():
        survey['title'] = "A Survey of " + Survey_dict[Global_survey_id]

        if k == "abstract":
            survey['abstract'] = ["The issue of class imbalance is pervasive in various practical applications of machine learning and data mining, including information retrieval and filtering, and the detection of credit card fraud. The problem of imbalanced learning concerns the effectiveness of learning algorithms when faced with underrepresented data and severe class distribution skews. The classification of data with imbalanced class distribution significantly hinders the performance of most standard classifier learning algorithms that assume a relatively balanced class distribution and equal misclassification costs.",
                                  "In this survey, we present a comprehensive overview of predictive modeling on imbalanced data. We categorize existing literature into three clusters: Sampling approaches, Algorithmic approaches, and Meta-learning approaches, which we introduce in detail. Our aim is to provide readers with a thorough understanding of the different strategies proposed to tackle the class imbalance problem and evaluate their effectiveness in enhancing the performance of learning algorithms."]
        if k == "introduction":
            survey['introduction'] = [
              {
                'subtitle': 'Background',
                'content' : '''Class imbalance is a common problem in machine learning and data mining, where the distribution of classes in the training dataset is highly skewed, with one class being significantly underrepresented compared to the other(s). This issue is prevalent in many real-world applications, including fraud detection, medical diagnosis, anomaly detection, and spam filtering, to name a few.
                               \nThe problem of imbalanced data affects the performance of many learning algorithms, which typically assume a balanced class distribution and equal misclassification costs. When the data is imbalanced, standard learning algorithms tend to favor the majority class, resulting in low accuracy in predicting the minority class. This drawback can lead to serious consequences, such as false negative errors in fraud detection or misdiagnosis in medical applications.
                               \nTo address the class imbalance problem, various techniques have been proposed, including resampling methods, cost-sensitive learning, and ensemble methods, among others. Resampling methods involve creating synthetic samples or under/oversampling the minority/majority classes to balance the data. Cost-sensitive learning assigns different misclassification costs to different classes to prioritize the minority class's correct prediction. Ensemble methods combine multiple models to improve predictive performance.
                               \nThe effectiveness of these techniques varies depending on the dataset and problem at hand. Hence, it is crucial to conduct a comprehensive evaluation of the different approaches to identify the most suitable one for a specific application. As such, your survey paper aims to provide an overview of the current state-of-the-art predictive modeling techniques for imbalanced data and highlight their strengths and limitations.
                            '''
              },
             {
                'subtitle': 'Methodologies', # Sampling approaches, Algorithmic approaches, and Meta-learning approaches
                'content' : '''Exisiting works are mainly categorized into Sampling approaches, Algorithmic approaches, and Meta-learning approaches.
                              \nSampling approaches:
                              \nResampling techniques are among the most popular methods for handling imbalanced data. These techniques involve either oversampling the minority class or undersampling the majority class to create a more balanced dataset. Examples of oversampling methods include SMOTE (Synthetic Minority Over-sampling Technique), ADASYN (Adaptive Synthetic Sampling), and Borderline-SMOTE. Undersampling techniques include random undersampling and Tomek Links. Moreover, hybrid methods, which combine both oversampling and undersampling, have also been proposed.
                              \nAlgorithmic approaches:
                              \nAnother approach to address the class imbalance problem is to modify the learning algorithm itself. Examples of such algorithmic approaches include cost-sensitive learning, where different costs are assigned to different types of misclassifications. Another approach is to adjust the decision threshold of the classifier, where the threshold is shifted to increase sensitivity towards the minority class. Additionally, ensemble methods, such as bagging, boosting, and stacking, have been proposed to combine multiple classifiers to improve predictive performance.
                              \nMeta-learning approaches:
                              \nMeta-learning approaches aim to automatically select the most suitable sampling or algorithmic approach for a specific dataset and problem. These approaches involve training a meta-classifier on multiple base classifiers, each using a different sampling or algorithmic approach. The meta-classifier then selects the most appropriate approach based on the characteristics of the input dataset. Examples of meta-learning approaches include MetaCost, MetaCostNN, and RAkEL.
                              \nThese approaches have shown promising results in addressing the class imbalance problem. However, their effectiveness depends on the specific characteristics of the dataset and problem at hand. Therefore, a comprehensive evaluation of different approaches is necessary to identify the most suitable one for a particular application.
                            '''
             },
             {
                'subtitle': 'Reminder',
                'content' : 'The rest of the paper is organized as follows. In section 2, we introduce the class imbalance problem and its causes and characteristics. Evaluation metrics are addressed in section 3. Section 4 presents an overview of the existing techniques for handling imbalanced data. Applications is illustrated in Section 5. Section 6 shows challenges and open issues. Conclusion and future directions are in Section 7.'
             }
            ]

        if k == "c_and_c":
            survey['c_and_c'] = '''Imbalanced data is a common problem in many real-world applications of machine learning and data mining, where the distribution of classes is highly skewed, with one or more classes being significantly underrepresented compared to the others. This can occur due to various reasons, such as sampling bias, data collection limitations, class overlap, or natural class distribution. The causes of imbalanced data can differ across different domains and applications, and understanding them is essential for developing effective predictive modeling techniques.
                            \nIn addition to the causes, imbalanced data is characterized by several properties that make it challenging for traditional machine learning algorithms. Firstly, the data imbalance results in a class distribution bias, where the majority class dominates the data, and the minority class(es) are often overshadowed, leading to poor classification performance. Secondly, the imbalance can lead to an asymmetric misclassification cost, where misclassifying the minority class is often more costly than misclassifying the majority class, resulting in high false negative rates. Thirdly, imbalanced data can exhibit class overlap, where instances from different classes are difficult to distinguish, leading to low discriminative power of the features and classifiers. Finally, imbalanced data can pose challenges for model evaluation and comparison, as traditional performance metrics such as accuracy, precision, and recall, can be misleading or inadequate in imbalanced settings.
                            \nUnderstanding the causes and characteristics of imbalanced data is crucial for developing effective and efficient predictive modeling techniques that can handle such data. The next section of this survey will discuss the various approaches proposed in the literature to address the imbalanced learning problem, with a focus on sampling, algorithmic, and meta-learning approaches.
                            '''
        if k == "evaluation":
            survey['evaluation'] = '''Evaluation metrics are an essential aspect of machine learning and data mining, as they quantify the performance of predictive models on a given dataset. In the case of imbalanced data, traditional evaluation metrics such as accuracy, precision, and recall may not be sufficient or even appropriate due to the class imbalance and asymmetry in misclassification costs. Therefore, alternative metrics have been proposed to measure the performance of predictive models on imbalanced datasets.
                            \nOne commonly used evaluation metric for imbalanced data is the area under the receiver operating characteristic curve (AUC-ROC). The AUC-ROC is a measure of the model's ability to distinguish between positive and negative instances and is computed as the area under the curve of the ROC plot. The ROC plot is a graphical representation of the trade-off between true positive rate (TPR) and false positive rate (FPR) for different decision thresholds. A perfect classifier would have an AUC-ROC score of 1, while a random classifier would have a score of 0.5.
                            \nAnother popular evaluation metric for imbalanced data is the area under the precision-recall curve (AUC-PR). The AUC-PR measures the precision-recall trade-off of the model and is computed as the area under the curve of the precision-recall plot. The precision-recall plot shows the relationship between precision and recall for different decision thresholds. A perfect classifier would have an AUC-PR score of 1, while a random classifier would have a score proportional to the ratio of positive to negative instances.
                            \nOther evaluation metrics for imbalanced data include F-measure, geometric mean, balanced accuracy, and cost-sensitive measures such as weighted and cost-sensitive versions of traditional metrics. F-measure is a harmonic mean of precision and recall, which balances the trade-off between them. The geometric mean is another metric that balances TPR and FPR and is useful in highly imbalanced datasets. Balanced accuracy is the average of TPR and TNR (true negative rate) and is useful in datasets where the class imbalance is extreme. Cost-sensitive measures incorporate the cost of misclassification and can be tailored to the specific application domain.
                            \nChoosing an appropriate evaluation metric for imbalanced data is essential to avoid biased or misleading performance estimates. The selection of metrics should be based on the application requirements, the class distribution, and the misclassification costs. In the next section, we will discuss various sampling, algorithmic, and meta-learning approaches proposed in the literature to address the imbalanced learning problem and their associated evaluation metrics.
                            '''

        if k == "methodology":
            survey['methodology'] = [
                'Our survey categorized existing works into three types: Sampling approaches, Algorithmic approaches, and Meta-learning approaches. Sampling approaches involve oversampling or undersampling, while algorithmic approaches modify the learning algorithm itself. Meta-learning approaches aim to automatically select the most suitable approach based on the characteristics of the input dataset.',
                [{'subtitle': 'Sampling approaches',
                  'content': 'For sampling approaches, Batista, et al. [1] proposed a simple experimental design to assess the performance of class imbalance treatment methods.  E.A.P.A. et al. [2] performs a broad experimental evaluation involving ten methods, three of them proposed by the authors, to deal with the class imbalance problem in thirteen uci data sets.  Batuwita, et al. [3] presents a method to improve fsvms for cil (called fsvm-cil), which can be used to handle the class imbalance problem in the presence of outliers and noise.  V. et al. [4] implements a wrapper approach that computes the amount of under-sampling and synthetic generation of the minority class examples (smote) to improve minority class accuracy.  Chen, et al. [5] presents ranked minority oversampling in boosting (ramoboost), which is a ramo technique based on the idea of adaptive synthetic data generation in an ensemble learning system.  Chen, et al. [6] proposes a new feature selection method, feature assessment by sliding thresholds (fast), which is based on the area under a roc curve generated by moving the decision boundary of a single feature classifier with thresholds placed using an even-bin distribution.  Davis, et al. [7] shows that a deep connection exists between roc space and pr space, such that a curve dominates in roc space if and only if it dominates in pr space.  In classifying documents, the system combines the predictions of the learners by applying evolutionary techniques as well [8]. Ertekin, et al. [9] is concerns with the class imbalance problem which has been known to hinder the learning performance of classification algorithms.  Ertekin, et al. [10] demonstrates that active learning is capable of solving the problem.  Garcı́aÿ, et al. [11] analyzes a generalization of a new metric to evaluate the classification performance in imbalanced domains, combining some estimate of the overall accuracy with a plain index about how dominant the class with the highest individual accuracy is.  Ghasemi, et al. [12] proposes an active learning algorithm that can work when only samples of one class as well as a set of unlabeled data are available.  He, et al. [13] provides a comprehensive review of the development of research in learning from imbalanced data.  Li, et al. [14] proposes an oversampling method based on support degree in order to guide people to select minority class samples and generate new minority class samples.  Li, et al. [15] analyzes the intrinsic factors behind this failure and proposes a suitable re-sampling method.  Liu, et al. [16] proposes two algorithms to overcome this deficiency.  J. et al. [17] considers the application of these ensembles to imbalanced data : classification problems where the class proportions are significantly different.  Seiffert, et al. [18] presents a new hybrid sampling/boosting algorithm, called rusboost, for learning from skewed training data.  Song, et al. [19] proposes an improved adaboost algorithm called baboost (balanced adaboost), which gives higher weights to the misclassified examples from the minority class.  Sun, et al. [20] develops a cost-sensitive boosting algorithm to improve the classification performance of imbalanced data involving multiple classes.  Van et al. [21] presents a comprehensive suite of experimentation on the subject of learning from imbalanced data.  Wasikowski, et al. [22] presents a first systematic comparison of the three types of methods developed for imbalanced data classification problems and of seven feature selection metrics evaluated on small sample data sets from different applications.  an active under-sampling approach is proposed for handling the imbalanced problem in Yang, et al. [23]. Zhou, et al. [24] studies empirically the effect of sampling and threshold-moving in training cost-sensitive neural networks. \n'},
                 {'subtitle': 'Algorithmic approaches',
                  'content': 'For algorithmic approaches, Baccianella, et al. [25] proposed a simple way to turn standard measures for or into ones robust to imbalance.  Lin, et al. [26] applies a fuzzy membership to each input point and reformulate the svms such that different input points can make different constributions to the learning of decision surface. \n'},
                 {'subtitle': 'Meta-learning approaches',
                  'content': 'For meta-learning approaches, Drummond, et al. [27] proposed an alternative to roc representation, in which the expected cost of a classi er is represented explicitly.  Tao, et al. [28] develops a mechanism to overcome these problems.  Torgo et al. [29] presents a generalization of regression error characteristic (rec) curves.  C. et al. [30] demonstrates that class probability estimates attained via supervised learning in imbalanced scenarios systematically underestimate the probabilities for minority class instances, despite ostensibly good overall calibration.  Yoon, et al. [31] proposes preprocessing majority instances by partitioning them into clusters.  Zheng, et al. [32] investigates the usefulness of explicit control of that combination within a proposed feature selection framework.'}]]



        if k == "app":
            survey['app'] = '''The problem of imbalanced data is pervasive in many real-world applications of predictive modeling, where the data is often skewed towards one or more minority class or classes. Such applications include, but are not limited to, fraud detection in finance, rare disease diagnosis in healthcare, fault detection in manufacturing, spam filtering in email systems, and anomaly detection in cybersecurity. In these scenarios, accurately identifying the minority class instances is of utmost importance, as they often represent critical and rare events that have significant impact or consequences.
                            \nHowever, traditional classification algorithms tend to perform poorly on imbalanced datasets, since they are often biased towards the majority class due to its abundance in the data. This results in low accuracy, high false negative rates, and poor generalization performance, especially for the minority class(es) of interest. In addition, the cost of misclassifying the minority class is often much higher than that of the majority class, making it even more critical to achieve high accuracy and low false negative rates for these instances.
                            \nTo overcome the class imbalance problem, a variety of predictive modeling techniques have been proposed and developed in the literature, specifically designed to handle imbalanced datasets. These techniques range from simple preprocessing methods that adjust the class distribution, to more complex algorithmic modifications that incorporate class imbalance considerations into the learning process. The effectiveness of these techniques depends on the specific characteristics of the dataset and problem, and thus, their selection and evaluation require careful experimentation and analysis.
                            \nOverall, the development and application of predictive modeling techniques for imbalanced data is an active and important research area, with many practical and societal implications. Advancements in this field have the potential to improve the accuracy, efficiency, and fairness of many critical applications, and thus, benefit society as a whole.
                            '''

        if k == "app":
            survey['clg'] = '''Selecting the most appropriate sampling, algorithmic, or meta-learning approach for a specific dataset: There is no one-size-fits-all solution, and choosing the right approach can be challenging.
                            \nLack of standard evaluation metrics that can capture the performance of models on imbalanced data, especially for rare events: Existing evaluation metrics like accuracy can be misleading in imbalanced datasets, and there is a need for metrics that can capture the performance of models on rare events.
                            \nInterpretability and explainability of models trained on imbalanced data: It can be difficult to understand how a model arrives at its predictions, especially when the data is heavily skewed, and there is a need for more interpretable models.
                            \nScalability of methods to handle very large datasets with imbalanced class distributions: As datasets grow in size, it can be challenging to scale methods to handle the imbalanced class distribution efficiently.
                            \nNeed for better feature engineering techniques to handle imbalanced data: Feature engineering is an important step in predictive modeling, and there is a need for better techniques that can handle imbalanced data.
                            \nDevelopment of new learning algorithms that are specifically designed to work well on imbalanced datasets: Most standard learning algorithms assume a relatively balanced class distribution, and there is a need for new algorithms that can handle imbalanced data more effectively.
                            \nResearch into the use of semi-supervised and unsupervised learning techniques for imbalanced data: Semi-supervised and unsupervised learning techniques have shown promise in imbalanced data, and there is a need for more research to explore their potential.
                            \nPotential benefits of using ensemble methods to combine multiple models trained on imbalanced data: Ensemble methods can improve the performance of models on imbalanced data by combining multiple models, and there is a need for more research to explore their potential.
                            \nDeveloping more effective methods for dealing with concept drift and evolving class distributions over time in imbalanced datasets: As class distributions evolve over time, it can be challenging to adapt models to the new distribution, and there is a need for more effective methods to handle concept drift.
                            '''


        if k == "conclusion":
            conclusion = '''In conclusion, the class imbalance problem is a significant challenge in predictive modeling, which can lead to biased models and poor performance. In this survey, we have provided a comprehensive overview of existing works on predictive modeling on imbalanced data. We have discussed different approaches to address this problem, including sampling approaches, algorithmic approaches, and meta-learning approaches, as well as evaluation metrics and challenges in this field. We also presented some potential future research directions in this area. The insights and knowledge provided in this survey paper can help researchers and practitioners better understand the challenges and opportunities in predictive modeling on imbalanced data and design more effective approaches to address this problem in real-world applications.
            \nThere are also some potencial directions for future research:
            \n1. Incorporating domain knowledge: Incorporating domain-specific knowledge can help improve the performance of models on imbalanced data. Research can be done on developing techniques to effectively integrate domain knowledge into the modeling process.
            \n2. Explainability of models: With the increasing adoption of machine learning models in critical applications, it is important to understand how the models make predictions. Research can be done on developing explainable models for imbalanced data, which can provide insights into the reasons for model predictions.
            \n3. Online learning: Imbalanced data can evolve over time, and models need to be adapted to new data as it becomes available. Research can be done on developing online learning algorithms that can adapt to imbalanced data in real-time.
            \n4. Multi-label imbalanced classification: In many real-world scenarios, multiple classes can be imbalanced simultaneously. Research can be done on developing techniques for multi-label imbalanced classification that can effectively handle such scenarios.
            \n5. Transfer learning: In some cases, imbalanced data in one domain can be used to improve the performance of models in another domain. Research can be done on developing transfer learning techniques for imbalanced data, which can leverage knowledge from related domains to improve performance.
            \n6. Incorporating fairness considerations: Models trained on imbalanced data can have biases that can disproportionately affect certain groups. Research can be done on developing techniques to ensure that models trained on imbalanced data are fair and do not discriminate against any particular group.
            \n7. Imbalanced data in deep learning: Deep learning has shown great promise in various applications, but its effectiveness on imbalanced data is not well understood. Research can be done on developing techniques to effectively apply deep learning to imbalanced data.
            \n8. Large-scale imbalanced data: With the increasing availability of large-scale datasets, research can be done on developing scalable techniques for predictive modeling on imbalanced data.
            '''
            survey['conclusion'] = conclusion

    survey['references'] = []
    try:
        for ref in Global_df_selected['ref_entry']:
            entry = str(ref)
            survey['references'].append(entry)
    except:
        import traceback
        print(traceback.print_exc())

    survey_dict = json.dumps(survey)

    return HttpResponse(survey_dict)

@csrf_exempt
def get_survey(request):
    survey_dict = get_survey_text()
    survey_dict = json.dumps(survey_dict)
    return HttpResponse(survey_dict)
    
@csrf_exempt
@timeout_handler(1800)  # 30分钟超时
def get_survey_id_sync(request, operation_id=None):
    """同步版本的获取调研ID函数"""
    operation_id = operation_id or getattr(request, 'operation_id', f"survey_{int(time.time())}")
    update_progress(operation_id, 0, "Starting survey generation...")
    
    global Global_survey_id, Global_survey_title, Global_collection_names_clustered, Global_citation_data
    
    try:
        update_progress(operation_id, 10, "Initializing survey generation...")
        
        if not Global_survey_id:
            update_progress(operation_id, -1, "Survey ID not found")
            return JsonResponse({"error": "Survey ID not found"}, status=400)
        
        if not Global_collection_names_clustered:
            update_progress(operation_id, -1, "No clustered collections found")
            return JsonResponse({"error": "No clustered collections found"}, status=400)
        
        update_progress(operation_id, 20, "Preparing survey data...")
        
        print("Global_collection_names_clustered: ")
        for i, element in enumerate(Global_collection_names_clustered):
            print(f"第 {i} 个元素：{element}")
        
        update_progress(operation_id, 30, "Generating survey content...")
        
        # 在子线程中执行survey生成，以便能够跟踪进度
        def generate_survey_with_progress():
            try:
                update_progress(operation_id, 40, "Generating survey outline...")
                
                # 这里调用实际的survey生成函数，不再需要pipeline参数
                generateSurvey_qwen_new(
                    Global_survey_id, 
                    Global_survey_title, 
                    Global_collection_names_clustered, 
                    None,  # pipeline参数设置为None，函数内部已经改为API调用
                    Global_citation_data,
                    embedder = get_embedder()
                )
                
                update_progress(operation_id, 90, "Survey generation completed!")
                return True
                
            except Exception as e:
                update_progress(operation_id, -1, f"Survey generation failed: {str(e)}")
                print(f"Error in generateSurvey_qwen_new: {e}")
                return False
        
        # 执行survey生成
        success = generate_survey_with_progress()
        
        if success:
            update_progress(operation_id, 100, "Survey ready!")
            
            response_data = {
                "survey_id": Global_survey_id,
                "message": "Survey generated successfully",
                "operation_id": operation_id,
                "processing_time": round(time.time() - start_time, 2)
            }
            
            return JsonResponse(response_data)
        else:
            return JsonResponse({"error": "Survey generation failed"}, status=500)
            
    except TimeoutError as e:
        update_progress(operation_id, -1, f"Survey generation timed out: {str(e)}")
        return JsonResponse({'error': f'Survey generation timed out after 30 minutes: {str(e)}'}, status=408)
    except Exception as e:
        update_progress(operation_id, -1, f"Survey generation failed: {str(e)}")
        return JsonResponse({'error': f'Survey generation failed: {str(e)}'}, status=500)

@csrf_exempt
@timeout_handler(1800)  # 30分钟超时
def generate_pdf_sync(request, operation_id=None):
    if request.method == 'POST':
        # 获取operation_id用于进度跟踪
        operation_id = operation_id or getattr(request, 'operation_id', f"pdf_{int(time.time())}")
        update_progress(operation_id, 10, "Starting PDF generation...")
        
        survey_id = request.POST.get('survey_id', '') or Global_survey_id
        if not survey_id:
            update_progress(operation_id, -1, "Missing survey_id; cannot generate PDF filename")
            return JsonResponse({'error': 'survey_id is required (and no global survey ID is set).'}, status=400)
        markdown_content = request.POST.get('content', '')
        
        update_progress(operation_id, 20, "Processing markdown content...")
        
        markdown_dir = f'./src/static/data/info/{survey_id}/'
        markdown_filename = f'survey_{survey_id}_vanilla.md'
        markdown_filepath = os.path.join(markdown_dir, markdown_filename)

        if not os.path.exists(markdown_dir):
            os.makedirs(markdown_dir)
            print(f"Directory '{markdown_dir}' created.")
        else:
            print(f"Directory '{markdown_dir}' already exists.")

        with open(markdown_filepath, 'w', encoding='utf-8') as markdown_file:
            markdown_file.write(markdown_content)
        print(f"Markdown content saved to: {markdown_filepath}")

        update_progress(operation_id, 40, "Finalizing survey paper...")
        
        markdown_content = finalize_survey_paper(markdown_content, Global_collection_names, Global_file_names)
        markdown_dir = f'./src/static/data/info/{survey_id}/'
        markdown_filename = f'survey_{survey_id}_processed.md'
        markdown_filepath = os.path.join(markdown_dir, markdown_filename)

        if not os.path.exists(markdown_dir):
            os.makedirs(markdown_dir)
            print(f"Directory '{markdown_dir}' created.")
        else:
            print(f"Directory '{markdown_dir}' already exists.")

        with open(markdown_filepath, 'w', encoding='utf-8') as markdown_file:
            markdown_file.write(markdown_content)
        print(f"Markdown content saved to: {markdown_filepath}")

        update_progress(operation_id, 60, "Generating PDF file...")

        pdf_filename = f'survey_{survey_id}.pdf'
        pdf_dir = './src/static/data/results'
        pdf_filepath = os.path.join(pdf_dir, pdf_filename)

        if not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir)
            print(f"Directory '{pdf_dir}' created.")
        else:
            print(f"Directory '{pdf_dir}' already exists.")

        print(f"PDF will be saved to: {pdf_filepath}")

        update_progress(operation_id, 80, "Converting markdown to PDF...")

        pdf = MarkdownPdf()
        pdf.meta["title"] = "Survey Results"
        pdf.add_section(Section(markdown_content, toc=False))
        pdf.save(pdf_filepath)

        update_progress(operation_id, 100, "PDF generation completed!")

        # 返回JSON格式的结果而不是二进制PDF数据
        return JsonResponse({
            'success': True,
            'message': 'PDF generated successfully',
            'survey_id': survey_id,
            'pdf_filename': pdf_filename,
            'pdf_path': pdf_filepath
        })

    return JsonResponse({'error': 'Invalid request method'}, status=400)

@csrf_exempt
def generate_pdf(request):
    """异步版本的PDF生成接口，避免Cloudflare 524超时"""
    if request.method == 'POST':
        operation_id = f"pdf_{int(time.time())}"
        survey_id = request.POST.get('survey_id', '') or Global_survey_id
        request.operation_id = operation_id
        success = task_manager.start_task(
            operation_id,
            generate_pdf_sync,
            request,
            operation_id
        )
        if not success:
            return JsonResponse({'error': 'PDF generation task already running'}, status=409)
        return JsonResponse({
            'operation_id': operation_id,
            'survey_id': survey_id,
            'status': 'started',
            'message': 'PDF generation started successfully. Use the operation_id to check progress.',
            'progress_url': f'/get_operation_progress/?operation_id={operation_id}'
        })
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
@timeout_handler(1800)  # 30分钟超时
def generate_pdf_from_tex_sync(request, operation_id=None):
    global Global_survey_id, Global_survey_title
    if request.method == 'POST':
        # 获取operation_id用于进度跟踪
        operation_id = operation_id or getattr(request, 'operation_id', f"latex_{int(time.time())}")
        update_progress(operation_id, 10, "Starting LaTeX PDF generation...")
        
        print(f"Request content type: {request.content_type}")
        print(f"Request POST data: {request.POST}")
        print(f"Request FILES: {request.FILES}")
        survey_id = request.POST.get('survey_id', '') or Global_survey_id
        markdown_content = request.POST.get('content', '')
        print(f"Survey ID: {survey_id}")
        print(f"Has content: {bool(markdown_content)}")
        
        if not survey_id:
            return JsonResponse({'error': 'survey_id is missing'}, status=400)
            
        update_progress(operation_id, 20, "Setting up directories...")
        
        base_dir = f'./src/static/data/info/{survey_id}'
        md_path = os.path.join(base_dir, f'survey_{survey_id}_processed.md')
        new_md_path = os.path.join(base_dir, f'survey_{survey_id}_preprocessed.md')
        tex_path = os.path.join(base_dir, 'template.tex')
        new_tex_path = os.path.join(base_dir, 'template_with_figure.tex')
        sty_path = os.path.join(base_dir, 'acl.sty')
        pdf_dir = './src/static/data/results'
        os.makedirs(base_dir, exist_ok=True)
        print(f"Directory '{base_dir}' checked or created.")
        
        update_progress(operation_id, 30, "Copying template files...")
        
        origin_template = 'src/demo/latex_template/template.tex'
        origin_acl_sty = 'src/demo/latex_template/acl.sty'
        shutil.copy(origin_template, tex_path)
        shutil.copy(origin_acl_sty, sty_path)
        os.makedirs(pdf_dir, exist_ok=True)
        
        update_progress(operation_id, 40, "Processing survey content...")
        
        # 如果传入了content且processed.md文件不存在，则创建它
        if markdown_content and not os.path.exists(md_path):
            # 先保存原始内容
            vanilla_md_path = os.path.join(base_dir, f'survey_{survey_id}_vanilla.md')
            with open(vanilla_md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"Vanilla markdown saved to: {vanilla_md_path}")
            
            # 处理并保存最终的markdown
            processed_content = finalize_survey_paper(markdown_content, Global_collection_names, Global_file_names)
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(processed_content)
            print(f"Processed markdown saved to: {md_path}")
        elif not os.path.exists(md_path):
            return JsonResponse({'error': f'Processed markdown file not found: {md_path}. Please generate regular PDF first or provide survey content.'}, status=400)
        
        update_progress(operation_id, 50, "Processing markdown content...")
        
        preprocess_md(md_path, new_md_path)
        md_to_tex(new_md_path, tex_path, Global_survey_title)
        
        update_progress(operation_id, 70, "Inserting figures and formatting...")
        
        insert_figures(
            png_path=f'src/static/data/info/{survey_id}/outline.png',
            tex_path= tex_path, 
            json_path=f'src/static/data/info/{survey_id}/flowchart_results.json',
            ref_names= Global_ref_list,
            survey_title=Global_survey_title,
            new_tex_path=new_tex_path
        )
        
        update_progress(operation_id, 85, "Compiling LaTeX to PDF...")
        
        tex_to_pdf(
            new_tex_path,
            output_dir=os.path.dirname(new_tex_path),
            compiler="xelatex"
        )
        pdf_path = os.path.join(os.path.dirname(new_tex_path), 'template_with_figure.pdf' )
        final_pdf_path = os.path.join(pdf_dir, f'survey_{survey_id}_latex.pdf')
        shutil.copy2(pdf_path, final_pdf_path)
        
        update_progress(operation_id, 100, "LaTeX PDF generation completed!")
        
        # 返回JSON格式的结果而不是二进制PDF数据
        return JsonResponse({
            'success': True,
            'message': 'LaTeX PDF generated successfully',
            'survey_id': survey_id,
            'pdf_filename': f'survey_{survey_id}_latex.pdf',
            'pdf_path': final_pdf_path
        })
        
    return JsonResponse({'error': 'Invalid request method'}, status=400)

def get_refs(topic):
    '''
    Get the references from given topic
    Return with a list
    '''
    default_references = ['ref1','ref2','ref3','ref4','ref5','ref6','ref7','ref8','ref9','ref10']
    default_ref_links = ['', '', '', '', '', '', '', '', '', '']
    default_ref_ids = ['', '', '', '', '', '', '', '', '', '']
    references = []
    ref_links = []
    ref_ids = []

    try:
        ## here is the algorithm part
        ref_path   = os.path.join(DATA_PATH, topic + '.tsv')
        df         = pd.read_csv(ref_path, sep='\t')
        for i,r in df.iterrows():
            # print(r['intro'], r['ref_title'], i)
            if not pd.isnull(r['intro']):
                references.append(r['ref_title'])
                ref_links.append(r['ref_link'])
                ref_ids.append(i)
    except:
        print(traceback.print_exc())
        references = default_references
        ref_links = default_ref_links
        ref_ids = default_ref_ids
    print(len(ref_ids))
    return references, ref_links, ref_ids

def get_survey_text(refs=Global_ref_list):
    '''
    Get the survey text from a given ref list
    Return with a dict as below default value:
    '''
    survey = {
        'Title': "A Survey of " + Survey_dict[Global_survey_id],
        'Abstract': "test "*150,
        'Introduction': "test "*500,
        'Methodology': [
            "This is the proceeding",
            [{"subtitle": "This is the first subtitle", "content": "test "*500},
             {"subtitle": "This is the second subtitle", "content": "test "*500},
             {"subtitle": "This is the third subtitle", "content": "test "*500}]
        ],
        'Conclusion': "test "*150,
        'References': []
    }

    try:
        ## abs generation
        abs, last_sent = absGen(Global_survey_id, Global_df_selected, Global_category_label)
        survey['Abstract'] = [abs, last_sent]

        ## Intro generation
        #intro = introGen_supervised(Global_survey_id, Global_df_selected, Global_category_label, Global_category_description)
        intro = introGen(Global_survey_id, Global_df_selected, Global_category_label, Global_category_description)
        survey['Introduction'] = intro

        ## Methodology generation
        proceeding, detailed_des = methodologyGen(Global_survey_id, Global_df_selected, Global_category_label, Global_category_description)
        survey['Methodology'] = [proceeding, detailed_des]

        ## Conclusion generation
        conclusion = conclusionGen(Global_survey_id, Global_category_label)
        survey['Conclusion'] = conclusion

        try:
            for ref in Global_df_selected['ref_entry']:
                entry = str(ref)
                survey['References'].append(entry)
        except:
            colors, category_label, category_description = Clustering_refs(n_clusters=Survey_n_clusters[Global_survey_id])
            for ref in Global_df_selected['ref_entry']:
                entry = str(ref)
                survey['References'].append(entry)

    except:
        print(traceback.print_exc())
    return survey

def Clustering_refs(n_clusters):
    global Global_cluster_num
    df = pd.read_csv(TSV_PATH + Global_survey_id + '.tsv', sep='\t', index_col=0, encoding='utf-8')

    print(Global_ref_list)
    df_selected = df.iloc[Global_ref_list]
    df_selected, colors, best_n_topics = clustering(df_selected, [3,4,5], Global_survey_id)
    Global_cluster_num = best_n_topics

    global Global_df_selected
    Global_df_selected = df_selected
    category_description = [0]*len(colors)
    category_label = [0]*len(colors)

    return colors, category_label
    # return 1,0,1

def remove_invalid_citations(text, valid_collection_names):
    pattern = r"\[(.*?)\\\]"
    all_matches = re.findall(pattern, text)

    new_text = text
    for match in all_matches:
        cleaned_match = match.rstrip('\\')
        if cleaned_match not in valid_collection_names:
            new_text = new_text.replace(f"[{match}\\]", "")
    return new_text

# wza
def normalize_citations_with_mapping(paper_text):
    citations = re.findall(r'\[.*?\]', paper_text)
    unique_citations = list(dict.fromkeys(citations))
    citation_mapping = {citation: f'[{i + 1}]' for i, citation in enumerate(unique_citations)}

    normalized_text = paper_text
    for old_citation, new_citation in citation_mapping.items():
        normalized_text = normalized_text.replace(old_citation, new_citation)

    reverse_mapping = {
        i + 1: unique_citations[i].strip('[]').rstrip('\\')
        for i in range(len(unique_citations))
    }

    return normalized_text, reverse_mapping

def generate_references_section(citation_mapping, collection_pdf_mapping):
    references = ["# References"]
    ref_list = []
    for num in sorted(citation_mapping.keys()):
        collection_name = citation_mapping[num]
        pdf_name = collection_pdf_mapping.get(collection_name, "Unknown PDF")
        if pdf_name.endswith(".pdf"):
            pdf_name = pdf_name[:-4]
        ref_list.append(pdf_name)
        # 在每一行末尾添加两个空格以确保换行
        references.append(f"[{num}] {pdf_name}  ")

    return "\n".join(references), ref_list

def fix_citation_punctuation_md(text):
    pattern = r'\.\s*(\\\[\d+\])'
    replacement = r' \1.'
    fixed_text = re.sub(pattern, replacement, text)
    return fixed_text

def finalize_survey_paper(paper_text, 
                          Global_collection_names, 
                          Global_file_names):
    global Global_survey_id, Global_survey_title, Global_ref_list

    paper_text = remove_invalid_citations(paper_text, Global_collection_names)
    normalized_text, citation_mapping = normalize_citations_with_mapping(paper_text)
    normalized_text = fix_citation_punctuation_md(normalized_text)
    collection_pdf_mapping = dict(zip(Global_collection_names, Global_file_names))
    
    references_section, ref_list = generate_references_section(citation_mapping, collection_pdf_mapping)
    Global_ref_list = ref_list
    print(ref_list)

    json_path = os.path.join("src", "static", "data", "txt", Global_survey_id, "outline.json")
    output_png_path = os.path.join("src", "static", "data", "info", Global_survey_id, "outline")
    md_path = os.path.join("src", "static", "data", "info", Global_survey_id, f"survey_{Global_survey_id}_processed.md")
    flowchart_results_path = os.path.join("src", "static", "data", "info", Global_survey_id, "flowchart_results.json")
    detect_flowcharts(Global_survey_id)
    png_path = generate_graphviz_png(
        json_path=json_path,
        output_png_path=output_png_path,
        md_content=normalized_text,
        title=Global_survey_title,
        max_root_chars=30
    )

    try:
        normalized_text = insert_ref_images(flowchart_results_path, ref_list, normalized_text)
    except Exception as e:
        print(f"Error inserting ref image: {e}. Continuing with next step.")
    try:
        normalized_text = insert_outline_image(
            png_path=png_path,
            md_content=normalized_text,
            survey_title =Global_survey_title
        )
    except Exception as e:
        print(f"Error inserting outline image: {e}. Continuing with next step.")

    final_paper = normalized_text.strip() + "\n\n" + references_section
    return final_paper

# Cleanup function for Django shutdown
def cleanup_resources():
    """Clean up resources when Django shuts down"""
    try:
        cleanup_openai_client()
        cleanup_retriever()
        print("Successfully cleaned up resources")
    except Exception as e:
        print(f"Error during cleanup: {e}")

# Register cleanup function for Django shutdown
import atexit
atexit.register(cleanup_resources)

@csrf_exempt  
def upload_refs(request):
    """异步版本的文件上传接口，立即返回operation_id避免Cloudflare 524超时。先保存文件到磁盘，再异步处理。"""
    if request.method == 'POST':
        operation_id = f"upload_{int(time.time())}"
        print(f"[DEBUG] Starting async upload task: {operation_id}")

        # 1. 先将所有上传文件保存到临时目录
        temp_dir = os.path.join('src', 'static', 'data', 'tmp_upload', operation_id)
        os.makedirs(temp_dir, exist_ok=True)
        file_paths = []
        for file_key, file in request.FILES.items():
            temp_path = os.path.join(temp_dir, file.name)
            with open(temp_path, 'wb') as f:
                for chunk in file.chunks():
                    f.write(chunk)
            file_paths.append(temp_path)

        # 2. 收集POST参数
        post_data = dict(request.POST.items())
        # 3. 启动异步任务，传递文件路径和参数
        def upload_refs_sync_wrapper(file_paths, post_data, operation_id):
            # 构造一个伪request对象，兼容原有upload_refs_sync逻辑
            class DummyRequest:
                method = 'POST'
                FILES = {}
                POST = post_data
            dummy_request = DummyRequest()
            dummy_request.file_paths = file_paths
            dummy_request.operation_id = operation_id
            return upload_refs_sync(dummy_request)

        success = task_manager.start_task(
            operation_id,
            upload_refs_sync_wrapper,
            file_paths,
            post_data,
            operation_id
        )
        if not success:
            print(f"[DEBUG] Task {operation_id} already running")
            return JsonResponse({'error': 'Upload task already running'}, status=409)
        print(f"[DEBUG] Async task {operation_id} started successfully")
        return JsonResponse({
            'operation_id': operation_id,
            'status': 'started',
            'message': 'File upload started successfully. Use the operation_id to check progress.',
            'progress_url': f'/get_operation_progress/?operation_id={operation_id}'
        })
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def automatic_taxonomy(request):
    """异步版本的自动分类接口，避免Cloudflare 524超时"""
    if request.method == 'POST':
        # 生成操作ID
        operation_id = f"taxonomy_{int(time.time())}"
        
        # 启动异步任务
        success = task_manager.start_task(
            operation_id, 
            automatic_taxonomy_sync, 
            request,
            operation_id
        )
        
        if not success:
            return JsonResponse({'error': 'Taxonomy task already running'}, status=409)
        
        # 立即返回operation_id
        return JsonResponse({
            'operation_id': operation_id,
            'status': 'started',
            'message': 'Automatic taxonomy started successfully. Use the operation_id to check progress.',
            'progress_url': f'/get_operation_progress/?operation_id={operation_id}'
        })
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def get_survey_id(request):
    """异步版本的获取调研ID接口，避免Cloudflare 524超时"""
    if request.method == 'POST':
        # 生成操作ID
        operation_id = f"survey_{int(time.time())}"
        
        # 启动异步任务
        success = task_manager.start_task(
            operation_id, 
            get_survey_id_sync, 
            request
        )
        
        if not success:
            return JsonResponse({'error': 'Survey generation task already running'}, status=409)
        
        # 立即返回operation_id
        return JsonResponse({
            'operation_id': operation_id,
            'status': 'started',
            'message': 'Survey generation started successfully. Use the operation_id to check progress.',
            'progress_url': f'/get_operation_progress/?operation_id={operation_id}'
        })
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def generate_pdf_from_tex(request):
    """异步版本的LaTeX PDF生成接口，避免Cloudflare 524超时"""
    if request.method == 'POST':
        operation_id = f"latex_{int(time.time())}"
        survey_id = request.POST.get('survey_id', '') or Global_survey_id
        request.operation_id = operation_id
        success = task_manager.start_task(
            operation_id,
            generate_pdf_from_tex_sync,
            request,
            operation_id
        )
        if not success:
            return JsonResponse({'error': 'LaTeX PDF generation task already running'}, status=409)
        return JsonResponse({
            'operation_id': operation_id,
            'survey_id': survey_id,
            'status': 'started',
            'message': 'LaTeX PDF generation started successfully. Use the operation_id to check progress.',
            'progress_url': f'/get_operation_progress/?operation_id={operation_id}'
        })
    return JsonResponse({'error': 'Invalid request method'}, status=405)

# @csrf_exempt
# def test_async_simple(request):
#     """简单的异步测试函数，用于验证异步机制"""
#     if request.method == 'POST':
#         operation_id = f"test_{int(time.time())}"
#         
#         def simple_task(request):
#             """简单的测试任务"""
#             update_progress(operation_id, 10, "Starting test task...")
#             time.sleep(2)
#             update_progress(operation_id, 50, "Task half way...")
#             time.sleep(2)
#             update_progress(operation_id, 100, "Test task completed!")
#             return JsonResponse({'message': 'Test completed successfully', 'test_id': operation_id})
#         
#         success = task_manager.start_task(operation_id, simple_task, request)
#         
#         if not success:
#             return JsonResponse({'error': 'Test task already running'}, status=409)
#         
#         return JsonResponse({
#             'operation_id': operation_id,
#             'status': 'started',
#             'message': 'Test task started successfully.'
#         })
#     
#     return JsonResponse({'error': 'Invalid request method'}, status=405)
