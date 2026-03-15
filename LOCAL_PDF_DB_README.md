# Local PDF Database Feature

This feature allows you to use a local folder of PDFs as a reference source instead of searching arXiv. This is useful for specialized research domains (e.g., Optical Research) where you have a curated collection of papers.

## Overview

- **Minimal modifications** to existing codebase
- **Front-end toggle** to switch between arXiv and Local Database
- **Support for subfolders** - recursively scans all PDFs in the specified folder
- **Cross-platform** - works on Windows, Linux, and macOS

## How to Use

### 1. Prepare Your Local PDF Database

Organize your PDF files in a folder structure like this:

```
E:\\Papers\\Optical_Research\\
├── Journal_Articles/
│   ├── paper1.pdf
│   ├── paper2.pdf
│   └── ...
├── Conference_Papers/
│   ├── conf_paper1.pdf
│   └── ...
└── Reviews/
    ├── review1.pdf
    └── ...
```

The system will recursively scan all subfolders for PDF files.

### 2. Using the Web Interface

1. Enter your research topic in the topic input box
2. Select "Local PDF Database" from the "Data Source" dropdown
3. Enter the full path to your PDF folder (e.g., `E:\\Papers\\Optical_Research`)
4. Click "Search References" button
5. The system will scan your local folder and display all available PDFs
6. Select the papers you want and click "Download All PDFs on Server"

### 3. API Usage

#### Search Local Database

```javascript
POST /generate_arxiv_query/
Content-Type: application/json

{
    "topic": "optical fiber communication",
    "source": "local",
    "local_db_path": "E:\\\\Papers\\\\Optical_Research"
}
```

Response:
```json
{
    "papers": [
        {
            "title": "Advanced Optical Fiber Systems",
            "summary": "First 500 characters of text...",
            "pdf_link": "E:\\\\Papers\\\\Optical_Research\\\\Journal_Articles\\\\paper1.pdf",
            "arxiv_id": "local_a1b2c3d4e5f6",
            "source": "local"
        }
    ],
    "count": 25,
    "source": "local",
    "local_db_path": "E:\\\\Papers\\\\Optical_Research"
}
```

#### Download/Copy PDFs

```javascript
POST /download_pdfs/
Content-Type: application/json

{
    "pdf_links": ["E:\\\\Papers\\\\paper1.pdf", "https://arxiv.org/pdf/xxxx.pdf"],
    "pdf_titles": ["paper1", "arxiv_paper"],
    "sources": ["local", "arxiv"],
    "local_db_path": "E:\\\\Papers\\\\Optical_Research"
}
```

## File Structure Changes

### New Files

1. **`src/demo/local_pdf_db.py`** - Local PDF database module
   - `scan_local_pdf_database()` - Recursively scan folder for PDFs
   - `extract_pdf_metadata()` - Extract title and summary from PDF
   - `search_local_pdfs()` - Search and filter local PDFs

### Modified Files

1. **`src/demo/views.py`**
   - Added import for `local_pdf_db` module
   - Modified `generate_arxiv_query()` to support `source` and `local_db_path` parameters
   - Modified `download_pdfs_sync()` to handle local file copying

2. **`src/demo/templates/demo/index.html`**
   - Added data source selector dropdown
   - Added local database path input field
   - Modified `fetchRecommendations()` to pass source parameters
   - Modified `sendPDFLinksToServer()` to handle local files

## Technical Details

### PDF Metadata Extraction

The system tries to extract metadata from PDFs in this order:

1. **PDF metadata** (if PyMuPDF/fitz is installed) - uses the `title` field from PDF metadata
2. **Filename** - converts filename to title (e.g., `optical_fiber_paper.pdf` → `optical fiber paper`)
3. **First page text** - extracts first 500 characters as summary

### Requirements

Optional but recommended:
```bash
pip install PyMuPDF
```

Without PyMuPDF, the system will still work but will use filenames instead of PDF metadata.

### Windows Path Support

The system automatically detects Windows-style paths (e.g., `C:\\folder\\file.pdf`) and handles them correctly.

## Error Handling

Common errors and solutions:

| Error | Cause | Solution |
|-------|-------|----------|
| "Local database path is required" | Selected "Local" but didn't provide path | Enter the folder path |
| "Local database path does not exist" | Invalid folder path | Check the path is correct |
| "Not enough papers found in local database" | Less than 10 PDFs in folder | Add more PDFs or reduce min_results |
| "Local file not found" | PDF was moved/deleted after scanning | Refresh the search |

## Configuration

You can set a default local database path via environment variable:

```bash
# Windows
set LOCAL_PDF_DB_PATH=E:\\Papers\\Optical_Research

# Linux/Mac
export LOCAL_PDF_DB_PATH=/home/user/papers/optical_research
```

## Future Enhancements

Possible improvements:

1. **Full-text indexing** - Index PDF content for faster searching
2. **Vector similarity search** - Use embeddings to find semantically similar papers
3. **Metadata caching** - Cache PDF metadata to avoid re-scanning
4. **Multiple database support** - Switch between multiple local databases
5. **Hybrid search** - Search both arXiv and local database simultaneously
