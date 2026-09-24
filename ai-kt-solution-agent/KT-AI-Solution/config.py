"""
Configuration for KT Documentation Generator
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === GCP & Gemini Configuration ===
GCP_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
EMBEDDING_MODEL = "text-embedding-004"
GEMINI_MODEL = "gemini-2.5-pro"
# === Qdrant Configuration ===
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME_PREFIX = "kt_docs_"  # Prefix for collection names
VECTOR_SIZE = 768  # text-embedding-004 dimension

# === Rate Limiting Configuration ===
RATE_LIMIT_RETRIES = 4
RETRY_DELAYS = [25, 60, 120, 240]  # seconds: 25s (20-30s range), 60s, 120s, 240s

# === File Scanning Configuration ===
PYTHON_EXTENSIONS = [".py"]
JAVASCRIPT_EXTENSIONS = [".js", ".jsx", ".ts", ".tsx"]
STYLE_EXTENSIONS = [".css", ".scss", ".sass", ".less"]
HTML_EXTENSIONS = [".html", ".htm"]
CONFIG_EXTENSIONS = [".yaml", ".yml", ".json", ".toml", ".ini"]
INFRA_FILES = ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"]
DOC_EXTENSIONS = [".md"]
SQL_EXTENSIONS = [".sql"]

# Dependency files (important for understanding the project)
DEPENDENCY_FILES = ["requirements.txt", "setup.py", "pyproject.toml", "Pipfile", "package.json"]
JS_CONFIG_FILES = ["tsconfig.json", "webpack.config.js", "vite.config.js", "vite.config.ts", 
                   "next.config.js", "next.config.ts", "rollup.config.js", "babel.config.js",
                   ".eslintrc.json", ".eslintrc.js", ".prettierrc.json", "postcss.config.js",
                   "tailwind.config.js", "tailwind.config.ts"]
ENV_FILES = [".env.example", ".env.sample"]

EXCLUDE_PATTERNS = [
    # Python excludes
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    "*.egg-info",
    ".tox",
    
    # JavaScript/Node.js excludes
    "node_modules",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    
    # Build outputs
    "build",
    "dist",
    ".next",
    "out",
    ".output",
    
    # Test files and directories
    "*.test.js",
    "*.test.jsx",
    "*.test.ts",
    "*.test.tsx",
    "*.spec.js",
    "*.spec.jsx",
    "*.spec.ts",
    "*.spec.tsx",
    "__tests__",
    "*.test.py",
    "tests",
    "test",
    
    # Cache and temp files
    ".cache",
    ".parcel-cache",
    ".vite",
    "coverage",
    ".nyc_output",
    
    # Static asset folders (exclude entire folders)
    "public",
    "static",
    "assets",
    
    # Storybook
    ".storybook",
    "storybook-static",
    
    # Minified and bundled files
    "*.min.js",
    "*.min.css",
    "*.bundle.js",
    "*.chunk.js",
    
    # Image and media files
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.svg",
    "*.ico",
    "*.webp",
    "*.bmp",
    "*.tiff",
    
    # Font files
    "*.ttf",
    "*.woff",
    "*.woff2",
    "*.eot",
    "*.otf",
    
    # Video and audio
    "*.mp4",
    "*.mp3",
    "*.avi",
    "*.mov",
    "*.wav",
    
    # IDE and editor configs
    ".vscode",
    ".idea",
    "*.swp",
    "*.swo",
    
    # Version control
    ".git",
    ".gitignore",
    ".github",
    
    # Environment files (security)
    ".env",
    "venv",
    "env",
    
    # License and changelog files
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "CHANGELOG.md",
    "changelog.md",
    
    # Ignore files
    ".prettierignore",
    ".eslintignore",
    ".dockerignore",
    
    # Credentials and keys
    "gen-lang-client-",
    "*credentials*.json",
    "*-key.json"
]

# === Documentation Configuration ===
DOC_TYPES = [
    "system_overview",
    "api_reference",
    "data_models",
    "business_flows",
    "integrations",
    "deployment"
]

DOC_TYPE_LABELS = {
    "system_overview": "System Overview & Architecture",
    "api_reference": "API Reference",
    "data_models": "Data Models & Database Schema",
    "business_flows": "Business Logic & Workflows",
    "integrations": "External Integrations",
    "deployment": "Deployment & Operations"
}

# === LLM Configuration ===
SUMMARY_GENERATION_CONFIG = {
    "temperature": 0.3,
    "max_output_tokens": 8000,  # Sufficient for detailed 300-500 word summaries
    "top_p": 0.8,
    "top_k": 40
}

DOC_GENERATION_CONFIG = {
    "temperature": 0.3,
    "max_output_tokens": 7500,  # Increased to ensure complete documentation
    "top_p": 0.8,
    "top_k": 40
}

