"""
Multi-Agent Test Case Comparison System with XLSX Input/Output
==============================================================

Reads new test cases from XLSX, compares against old test cases,
writes results to new XLSX with exponential backoff for 429 errors.

Note: All configurations and credentials should be loaded from environment variables.
See README.md for setup instructions.
"""

import numpy as np
from typing import List, Dict, TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
import vertexai
from vertexai.language_models import TextEmbeddingModel

from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
import operator
import time
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
import random


# ============================================================================
# EXPONENTIAL BACKOFF DECORATOR
# ============================================================================

def exponential_backoff(max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 60.0):
    """Decorator for exponential backoff on 429 errors"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    if '429' in error_str or 'rate limit' in error_str or 'quota' in error_str or 'resource_exhausted' in error_str:
                        retries += 1
                        if retries >= max_retries:
                            raise Exception(f"Max retries ({max_retries}) exceeded for rate limit error")
                        delay = min(base_delay * (2 ** retries) + random.uniform(0, 1), max_delay)
                        print(f"⚠️  Rate limit hit. Retry {retries}/{max_retries} in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        raise
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# STATE DEFINITIONS
# ============================================================================

class TestCaseComparisonState(TypedDict):
    """State shared across all agents"""
    new_test_case: str
    new_test_case_id: str
    top_100_candidates: List[Dict]
    top_50_candidates: List[Dict]
    batch_results: List[Dict]
    top_candidates_after_batching: List[Dict]
    final_matches: List[Dict]
    topics: Dict
    processing_log: Annotated[List[str], operator.add]
    errors: List[str]


# ============================================================================
# AGENT 1: RETRIEVAL AGENT (OPTIMIZED - Uses stored embeddings)
# ============================================================================

class RetrievalAgent:
    """Agent 1: Retrieves top-100 similar test cases and filters to top-50
    OPTIMIZED: Uses embeddings stored in Qdrant payload instead of regenerating
    """
    
    def __init__(self, qdrant_client: QdrantClient, embedding_model: TextEmbeddingModel,
                 collection_name: str = "old_booking_engine_test_cases"):
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.collection_name = collection_name
    
    @exponential_backoff(max_retries=5, base_delay=2.0, max_delay=120.0)
    def get_embedding(self, text: str):
        """Get embedding with retry logic"""
        return self.embedding_model.get_embeddings([text])[0].values
    
#     def __call__(self, state: TestCaseComparisonState) -> TestCaseComparisonState:
#         new_test_case = state['new_test_case']
#         log = []
#         log.append(f"[Agent 1] Starting retrieval...")
        
#         try:
#             # Generate embedding for new test case only
#             new_embedding = self.get_embedding(new_test_case)
            
#             # Search Qdrant for top-100 (embeddings are in payload)
#             search_results = self.qdrant_client.search(
#                 collection_name=self.collection_name,
#                 query_vector=new_embedding,
#                 limit=100,
#                 with_payload=True,
#                 with_vectors=False  # Don't need vectors, embeddings in payload
#             )
            
#             top_100 = []
#             for result in search_results:
#                 # Get stored embedding from payload
#                 stored_embedding = result.payload.get('embedding')
                
#                 top_100.append({
#                     'id': result.payload.get('test_case_id', 'unknown'),
#                     'text': result.payload.get('test_case_text', ''),
#                     'embedding_score': result.score,
#                     'stored_embedding': stored_embedding,  # Use stored embedding
#                     'primary_domain': result.payload.get('primary_domain', 'unknown'),
#                     'complexity': result.payload.get('complexity', 'unknown'),
#                     'test_type': result.payload.get('test_type', 'unknown'),
#                     'step_count': result.payload.get('step_count', 0),
#                     'payload': result.payload
#                 })
            
#             log.append(f"[Agent 1] Retrieved {len(top_100)} candidates")
            
#             # Calculate cosine similarity using STORED embeddings (no API calls!)
#             log.append("[Agent 1] Calculating cosine similarity using stored embeddings...")
            
#             new_emb_array = np.array([new_embedding])
#             old_embeddings = []
            
#             for candidate in top_100:
#                 if candidate['stored_embedding']:
#                     old_embeddings.append(candidate['stored_embedding'])
#                 else:
#                     # Fallback: use embedding score as cosine similarity
#                     old_embeddings.append(new_embedding)  # Placeholder
            
#             old_emb_array = np.array(old_embeddings)
#             cosine_scores = cosine_similarity(new_emb_array, old_emb_array)[0]
            
#             for i, candidate in enumerate(top_100):
#                 if candidate['stored_embedding']:
#                     candidate['cosine_similarity'] = float(cosine_scores[i])
#                 else:
#                     candidate['cosine_similarity'] = candidate['embedding_score']
#                 # Remove stored embedding to save memory
#                 del candidate['stored_embedding']
            
#             top_50 = sorted(top_100, key=lambda x: x['cosine_similarity'], reverse=True)[:50]
#             log.append(f"[Agent 1] Filtered to top-50 (NO extra API calls!)")
            
#             state['top_100_candidates'] = top_100
#             state['top_50_candidates'] = top_50
#             state['processing_log'] = log
            
#         except Exception as e:
#             error_msg = f"[Agent 1 ERROR] {str(e)}"
#             log.append(error_msg)
#             state['errors'] = state.get('errors', []) + [error_msg]
#             state['processing_log'] = log
        
#         return state


# # ============================================================================
# # AGENT 2: BATCH SEMANTIC FILTER AGENT
# # ============================================================================

# class BatchSemanticFilterAgent:
#     """Agent 2: Process top-50 in batches of 10, filter each batch to top-5"""
    
#     def __init__(self, genai_client: genai.Client, model_name: str = "gemini-2.0-flash-exp", batch_size: int = 10):
#         self.client = genai_client
#         self.model_name = model_name
#         self.batch_size = batch_size
    
#     @exponential_backoff(max_retries=5, base_delay=2.0, max_delay=120.0)
#     def llm_call(self, prompt: str) -> str:
#         """LLM call with retry logic"""
#         response = self.client.models.generate_content(
#             model=self.model_name,
#             contents=prompt,
#             config=GenerateContentConfig(temperature=0.1, response_mime_type="application/json")
#         )
#         return response.text
    
#     def filter_batch(self, new_test_case: str, batch: List[Dict]) -> List[Dict]:
#         batch_text = ""
#         for i, candidate in enumerate(batch, 1):
#             batch_text += f"\n[CANDIDATE {i}]\nID: {candidate['id']}\nText: {candidate['text'][:300]}...\n"
        
#         prompt = f"""You are a QA expert comparing test cases.

# **NEW TEST CASE:**
# {new_test_case}

# **CANDIDATES ({len(batch)}):**
# {batch_text}

# Select TOP 5 most similar. Respond in JSON:
# {{"selected_candidates": [{{"candidate_number": 1-{len(batch)}, "semantic_score": 0-100, "reasoning": "brief"}}]}}
# """
        
#         try:
#             result = json.loads(self.llm_call(prompt))
#             selected = []
#             for selection in result['selected_candidates'][:5]:
#                 candidate_idx = selection['candidate_number'] - 1
#                 if 0 <= candidate_idx < len(batch):
#                     candidate = batch[candidate_idx].copy()
#                     candidate['semantic_score'] = selection['semantic_score']
#                     candidate['llm_reasoning'] = selection['reasoning']
#                     selected.append(candidate)
#             return selected
#         except Exception as e:
#             print(f"⚠️  Batch filtering error: {str(e)}")
#             return sorted(batch, key=lambda x: x['cosine_similarity'], reverse=True)[:5]
    
#     def __call__(self, state: TestCaseComparisonState) -> TestCaseComparisonState:
#         new_test_case = state['new_test_case']
#         top_50 = state['top_50_candidates']
#         log = []
#         log.append(f"[Agent 2] Processing {len(top_50)} candidates in batches")
        
#         try:
#             all_filtered = []
#             for i in range(0, len(top_50), self.batch_size):
#                 batch = top_50[i:i + self.batch_size]
#                 batch_num = (i // self.batch_size) + 1
#                 log.append(f"[Agent 2] Processing batch {batch_num}...")
#                 top_5_in_batch = self.filter_batch(new_test_case, batch)
#                 all_filtered.extend(top_5_in_batch)
#                 time.sleep(1)
            
#             all_filtered_sorted = sorted(all_filtered, key=lambda x: x.get('semantic_score', 0), reverse=True)
#             log.append(f"[Agent 2] Total after batching: {len(all_filtered_sorted)}")
            
#             state['top_candidates_after_batching'] = all_filtered_sorted
#             state['processing_log'] = log
            
#         except Exception as e:
#             error_msg = f"[Agent 2 ERROR] {str(e)}"
#             log.append(error_msg)
#             state['errors'] = state.get('errors', []) + [error_msg]
#             state['processing_log'] = log
        
#         return state


# # ============================================================================
# # AGENT 3: FINAL SELECTOR AGENT
# # ============================================================================

# class FinalSelectorAgent:
#     """Agent 3: Deep semantic analysis to select top 1-2 final matches"""
    
#     def __init__(self, genai_client: genai.Client, model_name: str = "gemini-2.0-flash-exp"):
#         self.client = genai_client
#         self.model_name = model_name
    
#     @exponential_backoff(max_retries=5, base_delay=2.0, max_delay=120.0)
#     def llm_call(self, prompt: str) -> str:
#         response = self.client.models.generate_content(
#             model=self.model_name,
#             contents=prompt,
#             config=GenerateContentConfig(temperature=0.1, response_mime_type="application/json")
#         )
#         return response.text
    
#     def __call__(self, state: TestCaseComparisonState) -> TestCaseComparisonState:
#         new_test_case = state['new_test_case']
#         candidates = state['top_candidates_after_batching']
#         log = []
#         log.append(f"[Agent 3] Analyzing {len(candidates)} candidates for final selection")
        
#         try:
#             top_10 = candidates[:10]
#             candidates_text = ""
#             for i, candidate in enumerate(top_10, 1):
#                 candidates_text += f"\n[CANDIDATE {i}]\nID: {candidate['id']}\nCosine: {candidate['cosine_similarity']:.4f}\nText:\n{candidate['text']}\n"
            
#             prompt = f"""You are an expert QA analyst performing final test case mapping.

# **NEW TEST CASE:**
# {new_test_case}

# **TOP CANDIDATES:**
# {candidates_text}

# Select TOP 1 or 2 best matches. Respond in JSON:
# {{
#   "final_matches": [{{
#     "candidate_number": 1-10,
#     "match_confidence": "HIGH/MEDIUM/LOW",
#     "semantic_similarity": 0-100,
#     "coverage_assessment": "FULL/PARTIAL/NO_COVERAGE",
#     "reasoning": "explanation"
#   }}]
# }}
# """
            
#             result = json.loads(self.llm_call(prompt))
#             final_matches = []
#             for match in result['final_matches']:
#                 candidate_idx = match['candidate_number'] - 1
#                 if 0 <= candidate_idx < len(top_10):
#                     candidate = top_10[candidate_idx].copy()
#                     candidate.update({
#                         'match_confidence': match['match_confidence'],
#                         'final_semantic_similarity': match['semantic_similarity'],
#                         'coverage_assessment': match['coverage_assessment'],
#                         'final_reasoning': match['reasoning']
#                     })
#                     final_matches.append(candidate)
            
#             log.append(f"[Agent 3] Selected {len(final_matches)} final match(es)")
#             state['final_matches'] = final_matches
#             state['processing_log'] = log
            
#         except Exception as e:
#             error_msg = f"[Agent 3 ERROR] {str(e)}"
#             log.append(error_msg)
#             state['errors'] = state.get('errors', []) + [error_msg]
#             state['processing_log'] = log
        
#         return state


# # ============================================================================
# # AGENT 4: TOPIC EXTRACTION AGENT
# # ============================================================================

# class TopicExtractionAgent:
#     """Agent 4: Extract topics from matched OLD and NEW test cases"""
    
#     def __init__(self, genai_client: genai.Client, model_name: str = "gemini-2.0-flash-exp"):
#         self.client = genai_client
#         self.model_name = model_name
    
#     @exponential_backoff(max_retries=5, base_delay=2.0, max_delay=120.0)
#     def llm_call(self, prompt: str) -> str:
#         response = self.client.models.generate_content(
#             model=self.model_name,
#             contents=prompt,
#             config=GenerateContentConfig(temperature=0.1, response_mime_type="application/json")
#         )
#         return response.text
    
#     def __call__(self, state: TestCaseComparisonState) -> TestCaseComparisonState:
#         new_test_case = state['new_test_case']
#         final_matches = state['final_matches']
#         log = []
#         log.append(f"[Agent 4] Extracting topics from {len(final_matches)} match(es)")
        
#         try:
#             topics_per_match = []
#             for match in final_matches:
#                 old_test_case = match['text']
                
#                 prompt = f"""Extract topics from these test cases.

# **NEW TEST CASE:**
# {new_test_case}

# **OLD TEST CASE:**
# {old_test_case}

# Respond in JSON:
# {{
#   "new_test_primary_topic": "main topic",
#   "old_test_primary_topic": "main topic",
#   "common_topics": ["shared1", "shared2"],
#   "topic_similarity_score": 0-100
# }}
# """
                
#                 topic_result = json.loads(self.llm_call(prompt))
#                 topics_per_match.append({
#                     'match_id': match['id'],
#                     'topics': topic_result
#                 })
#                 log.append(f"[Agent 4] Topic: {topic_result['new_test_primary_topic']}")
#                 time.sleep(0.5)
            
#             state['topics'] = {'matches': topics_per_match, 'extracted_at': datetime.now().isoformat()}
#             state['processing_log'] = log
            
#         except Exception as e:
#             error_msg = f"[Agent 4 ERROR] {str(e)}"
#             log.append(error_msg)
#             state['errors'] = state.get('errors', []) + [error_msg]
#             state['processing_log'] = log
        
#         return state


# # ============================================================================
# # LANGGRAPH WORKFLOW BUILDER
# # ============================================================================

# class TestCaseComparisonWorkflow:
#     """Main workflow orchestrator using LangGraph"""
    
#     def __init__(self, qdrant_url: str, qdrant_api_key: str, gcp_project_id: str,
#                  gcp_location: str, service_account_path: str,
#                  collection_name: str = "old_booking_engine_test_cases",
#                  embedding_model_name: str = "text-embedding-004",
#                  llm_model_name: str = "gemini-2.0-flash-exp"):
        
#         print("🔧 Initializing Multi-Agent Workflow...")
        
#         self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
#         print("  ✅ Qdrant connected")
        
#         credentials = service_account.Credentials.from_service_account_file(
#             service_account_path,
#             scopes=['https://www.googleapis.com/auth/cloud-platform']
#         )
        
#         vertexai.init(project=gcp_project_id, location=gcp_location, credentials=credentials)
#         self.embedding_model = TextEmbeddingModel.from_pretrained(embedding_model_name)
#         print(f"  ✅ Embedding model: {embedding_model_name}")
        
#         self.genai_client = genai.Client(vertexai=True, project=gcp_project_id, location=gcp_location)
#         self.llm_model_name = llm_model_name
#         print(f"  ✅ GenAI client: {llm_model_name}")
        
#         self.retrieval_agent = RetrievalAgent(self.qdrant_client, self.embedding_model, collection_name)
#         self.batch_filter_agent = BatchSemanticFilterAgent(self.genai_client, llm_model_name, batch_size=10)
#         self.final_selector_agent = FinalSelectorAgent(self.genai_client, llm_model_name)
#         self.topic_extraction_agent = TopicExtractionAgent(self.genai_client, llm_model_name)
        
#         self.workflow = self.build_workflow()
#         print("✅ Multi-Agent Workflow ready!\n")
    
#     def build_workflow(self) -> StateGraph:
#         workflow = StateGraph(TestCaseComparisonState)
#         workflow.add_node("retrieval", self.retrieval_agent)
#         workflow.add_node("batch_filter", self.batch_filter_agent)
#         workflow.add_node("final_selector", self.final_selector_agent)
#         workflow.add_node("topic_extraction", self.topic_extraction_agent)
        
#         workflow.set_entry_point("retrieval")
#         workflow.add_edge("retrieval", "batch_filter")
#         workflow.add_edge("batch_filter", "final_selector")
#         workflow.add_edge("final_selector", "topic_extraction")
#         workflow.add_edge("topic_extraction", END)
        
#         return workflow.compile()
    
#     def compare_test_case(self, new_test_case: str, new_test_case_id: str = None) -> Dict:
#         initial_state = {
#             'new_test_case': new_test_case,
#             'new_test_case_id': new_test_case_id or f"NEW_TC_{datetime.now().strftime('%Y%m%d%H%M%S')}",
#             'top_100_candidates': [],
#             'top_50_candidates': [],
#             'batch_results': [],
#             'top_candidates_after_batching': [],
#             'final_matches': [],
#             'topics': {},
#             'processing_log': [],
#             'errors': []
#         }
        
#         return self.workflow.invoke(initial_state)


# # ============================================================================
# # XLSX PROCESSOR
# # ============================================================================

# class XLSXTestCaseProcessor:
#     """Process test cases from XLSX and write results to new XLSX"""
    
#     def __init__(self, workflow: TestCaseComparisonWorkflow):
#         self.workflow = workflow
    
#     def read_test_cases(self, input_file: str, test_case_column: str = "Test Case") -> pd.DataFrame:
#         """Read test cases from input XLSX"""
#         print(f"📂 Reading test cases from: {input_file}")
#         df = pd.read_excel(input_file)
#         print(f"  ✅ Found {len(df)} test cases")
#         print(f"  📋 Available columns: {list(df.columns)}")
        
#         if test_case_column not in df.columns:
#             print(f"\n❌ ERROR: Column '{test_case_column}' not found!")
#             print(f"   Available columns are: {list(df.columns)}")
#             print(f"   Please update TEST_CASE_COLUMN in main() to match one of these.")
#             raise KeyError(f"Column '{test_case_column}' not found in Excel file")
        
#         return df
    
#     def process_all(self, input_file: str, output_file: str, test_case_column: str = "Test Case"):
#         """Process all test cases and write results"""
        
#         df = self.read_test_cases(input_file, test_case_column)
#         results = []
        
#         total = len(df)
#         for idx, row in df.iterrows():
#             test_case_text = str(row[test_case_column])
            
#             print(f"\n{'='*60}")
#             print(f"🔄 Processing {idx+1}/{total}")
#             print(f"{'='*60}")
            
#             try:
#                 result = self.workflow.compare_test_case(test_case_text)
                
#                 if result['final_matches']:
#                     for i, match in enumerate(result['final_matches']):
#                         topic_info = {}
#                         if result['topics'].get('matches'):
#                             for tm in result['topics']['matches']:
#                                 if tm['match_id'] == match['id']:
#                                     topic_info = tm['topics']
#                                     break
                        
#                         results.append({
#                             'New Test Case': test_case_text,
#                             'Matching Old Test Case': match['text'],
#                             'Match Confidence': match.get('match_confidence', 'N/A'),
#                             'Topic (New Test Case)': topic_info.get('new_test_primary_topic', 'N/A'),
#                             'Topic (Old Test Case)': topic_info.get('old_test_primary_topic', 'N/A'),
#                             'Common Topics': ', '.join(topic_info.get('common_topics', [])),
#                             'Cosine Similarity': round(match['cosine_similarity'], 4)
#                         })
                        
#                         print(f"  ✅ Match found: {match['id']} (Cosine: {match['cosine_similarity']:.4f})")
#                 else:
#                     results.append({
#                         'New Test Case': test_case_text,
#                         'Matching Old Test Case': 'No matching test case found',
#                         'Match Confidence': 'N/A',
#                         'Topic (New Test Case)': 'N/A',
#                         'Topic (Old Test Case)': 'N/A',
#                         'Common Topics': 'N/A',
#                         'Cosine Similarity': 0.0
#                     })
#                     print(f"  ⚠️ No match found")
                    
#             except Exception as e:
#                 print(f"  ❌ Error processing test case {idx+1}: {str(e)}")
#                 results.append({
#                     'New Test Case': test_case_text,
#                     'Matching Old Test Case': f'Processing error: {str(e)}',
#                     'Match Confidence': 'N/A',
#                     'Topic (New Test Case)': 'N/A',
#                     'Topic (Old Test Case)': 'N/A',
#                     'Common Topics': 'N/A',
#                     'Cosine Similarity': 0.0
#                 })
            
#             time.sleep(2)
        
#         self.write_results(results, output_file)
#         return results
    
#     def write_results(self, results: List[Dict], output_file: str):
#         """Write results to formatted XLSX"""
#         print(f"\n📝 Writing results to: {output_file}")
        
#         wb = Workbook()
#         ws = wb.active
#         ws.title = "Test Case Comparison Results"
        
#         headers = [
#             'New Test Case', 'Matching Old Test Case', 'Match Confidence', 
#             'Topic (New Test Case)', 'Topic (Old Test Case)', 'Common Topics', 'Cosine Similarity'
#         ]
        
#         header_font = Font(bold=True, color='FFFFFF')
#         header_fill = PatternFill('solid', fgColor='4472C4')
#         header_alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
#         thin_border = Border(
#             left=Side(style='thin'),
#             right=Side(style='thin'),
#             top=Side(style='thin'),
#             bottom=Side(style='thin')
#         )
        
#         for col, header in enumerate(headers, 1):
#             cell = ws.cell(row=1, column=col, value=header)
#             cell.font = header_font
#             cell.fill = header_fill
#             cell.alignment = header_alignment
#             cell.border = thin_border
        
#         for row_idx, result in enumerate(results, 2):
#             for col_idx, header in enumerate(headers, 1):
#                 cell = ws.cell(row=row_idx, column=col_idx, value=result.get(header, ''))
#                 cell.alignment = Alignment(vertical='top', wrap_text=True)
#                 cell.border = thin_border
        
#         column_widths = {'A': 50, 'B': 50, 'C': 18, 'D': 25, 'E': 25, 'F': 30, 'G': 15}
#         for col, width in column_widths.items():
#             ws.column_dimensions[col].width = width
        
#         ws.freeze_panes = 'A2'
#         wb.save(output_file)
#         print(f"  ✅ Results saved: {len(results)} rows written")


# # ============================================================================
# # MAIN EXECUTION
# # ============================================================================

# def main():
#     import os
    
#     # Configuration - UPDATE THESE PATHS
#     SERVICE_ACCOUNT_PATH = "C:\\Users\\gudladhana.harshith\\Desktop\\TTC Comparsion\\vertex_credentials.json"
#     os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = SERVICE_ACCOUNT_PATH
    
#     QDRANT_URL = "https://b5030dd4-154d-457a-b473-bb09f2b0ecab.us-east4-0.gcp.cloud.qdrant.io"
#     QDRANT_API_KEY = "REDACTED_TOKEN"
#     COLLECTION_NAME = "old_booking_engine_test_cases"
    
#     GCP_PROJECT_ID = "dg-assistant-451309"
#     GCP_LOCATION = "us-central1"
#     EMBEDDING_MODEL = "gemini-embedding-001"
#     LLM_MODEL = "gemini-2.5-pro"
    
#     # Input/Output files - UPDATE THESE PATHS
#     INPUT_XLSX = r"C:\Users\gudladhana.harshith\Downloads\New Shopping Cart.xlsx"
#     OUTPUT_XLSX = r"C:\Users\gudladhana.harshith\Downloads\Comparsion.xlsx"
    
#     # Column name in your input XLSX
#     TEST_CASE_COLUMN = "New Shopping Cart Test Case Scenario"  # Column containing test case text
    
#     # Initialize workflow
#     workflow = TestCaseComparisonWorkflow(
#         qdrant_url=QDRANT_URL,
#         qdrant_api_key=QDRANT_API_KEY,
#         gcp_project_id=GCP_PROJECT_ID,
#         gcp_location=GCP_LOCATION,
#         service_account_path=SERVICE_ACCOUNT_PATH,
#         collection_name=COLLECTION_NAME,
#         embedding_model_name=EMBEDDING_MODEL,
#         llm_model_name=LLM_MODEL
#     )
    
#     # Process all test cases
#     processor = XLSXTestCaseProcessor(workflow)
#     results = processor.process_all(
#         input_file=INPUT_XLSX,
#         output_file=OUTPUT_XLSX,
#         test_case_column=TEST_CASE_COLUMN
#     )
    
#     print(f"\n{'='*60}")
#     print(f"✅ PROCESSING COMPLETE")
#     print(f"  📊 Total test cases processed: {len(results)}")
#     print(f"  📁 Results saved to: {OUTPUT_XLSX}")
#     print(f"{'='*60}")


# if __name__ == "__main__":
#     main()


"""
Multi-Agent Test Case Comparison System (Updated)
==================================================

Direction: OLD test cases → find top 3 matching NEW test cases
Analysis:  Difference analysis + LLM matching score (no cosine output)
Input:     Final_Sheet.xlsx (both columns)
Output:    Comparison.xlsx with detailed gap analysis

Author: Harshith
Date: 2026-02-20
"""

import numpy as np
from typing import List, Dict, TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
from google.oauth2 import service_account
import vertexai
from vertexai.language_models import TextEmbeddingModel

from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
import operator
import time
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
import random


# ============================================================================
# EXPONENTIAL BACKOFF DECORATOR
# ============================================================================

def exponential_backoff(max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 60.0):
    """Decorator for exponential backoff on 429 errors"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    if '429' in error_str or 'rate limit' in error_str or 'quota' in error_str or 'resource_exhausted' in error_str:
                        retries += 1
                        if retries >= max_retries:
                            raise Exception(f"Max retries ({max_retries}) exceeded for rate limit error")
                        delay = min(base_delay * (2 ** retries) + random.uniform(0, 1), max_delay)
                        print(f"⚠️  Rate limit hit. Retry {retries}/{max_retries} in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        raise
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# STATE DEFINITION
# ============================================================================

class TestCaseComparisonState(TypedDict):
    """State shared across all agents"""
    # Input: the OLD test case we're finding matches for
    old_test_case: str
    old_test_case_id: str
    old_row_index: int
    # Pipeline stages
    top_candidates: List[Dict]          # After vector retrieval
    filtered_candidates: List[Dict]     # After batch semantic filter
    final_matches: List[Dict]           # Top 3 with LLM scores + difference analysis
    # Logging
    processing_log: Annotated[List[str], operator.add]
    errors: List[str]


# ============================================================================
# AGENT 1: RETRIEVAL AGENT
# ============================================================================

class RetrievalAgent:
    """
    Agent 1: For each OLD test case, retrieve top-50 similar NEW test cases
    from Qdrant using vector similarity. Filters by source='new_shopping_cart'.
    """

    def __init__(self, qdrant_client: QdrantClient, embedding_model: TextEmbeddingModel,
                 collection_name: str = "test_case_comparisons"):
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.collection_name = collection_name

    @exponential_backoff(max_retries=5, base_delay=2.0, max_delay=120.0)
    def get_embedding(self, text: str):
        return self.embedding_model.get_embeddings([text])[0].values

    def __call__(self, state: TestCaseComparisonState) -> TestCaseComparisonState:
        old_test_case = state['old_test_case']
        log = []
        log.append(f"[Agent 1] Retrieving NEW test case matches for OLD TC...")

        try:
            old_embedding = self.get_embedding(old_test_case)

            # Search only NEW test cases using source filter
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=old_embedding,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value="new_shopping_cart")
                        )
                    ]
                ),
                limit=50,
                with_payload=True,
                with_vectors=False,
            )

            candidates = []
            for result in search_results:
                candidates.append({
                    'id': result.payload.get('test_case_id', 'unknown'),
                    'text': result.payload.get('test_case_text', ''),
                    'vector_score': round(result.score, 4),
                    'primary_domain': result.payload.get('primary_domain', 'unknown'),
                    'test_type': result.payload.get('test_type', 'unknown'),
                    'intent': result.payload.get('intent', 'unknown'),
                    'row_index': result.payload.get('row_index', -1),
                })

            log.append(f"[Agent 1] Retrieved {len(candidates)} NEW test case candidates")
            state['top_candidates'] = candidates
            state['processing_log'] = log

        except Exception as e:
            error_msg = f"[Agent 1 ERROR] {str(e)}"
            log.append(error_msg)
            state['errors'] = state.get('errors', []) + [error_msg]
            state['processing_log'] = log

        return state


# ============================================================================
# AGENT 2: BATCH SEMANTIC FILTER AGENT
# ============================================================================

class BatchSemanticFilterAgent:
    """
    Agent 2: Process top-50 in batches of 10, keep top-5 per batch.
    Yields ~25 candidates for final selection.
    """

    def __init__(self, genai_client: genai.Client, model_name: str = "gemini-2.5-pro",
                 batch_size: int = 10):
        self.client = genai_client
        self.model_name = model_name
        self.batch_size = batch_size

    @exponential_backoff(max_retries=5, base_delay=2.0, max_delay=120.0)
    def llm_call(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=GenerateContentConfig(temperature=0.1, response_mime_type="application/json")
        )
        return response.text

    def filter_batch(self, old_test_case: str, batch: List[Dict]) -> List[Dict]:
        batch_text = ""
        for i, c in enumerate(batch, 1):
            batch_text += f"\n[CANDIDATE {i}]\nID: {c['id']}\nText: {c['text']}\n"

        prompt = f"""You are a QA expert comparing test cases.

The OLD test case below is from a legacy booking engine. The candidates are NEW test cases from a shopping cart system. Select the TOP 5 candidates that are most functionally similar to the old test case.

**OLD TEST CASE (base):**
{old_test_case}

**NEW TEST CASE CANDIDATES ({len(batch)}):**
{batch_text}

Select the TOP 5 most similar. Respond in JSON:
{{"selected_candidates": [{{"candidate_number": 1, "relevance_score": 85, "reasoning": "brief explanation"}}]}}

Rules:
- candidate_number is 1-indexed matching the candidate numbers above
- relevance_score is 0-100 (how well the new TC covers the old TC's intent)
- Select exactly 5 (or fewer if batch is smaller)
"""

        try:
            result = json.loads(self.llm_call(prompt))
            selected = []
            for sel in result.get('selected_candidates', [])[:5]:
                idx = sel['candidate_number'] - 1
                if 0 <= idx < len(batch):
                    candidate = batch[idx].copy()
                    candidate['relevance_score'] = sel.get('relevance_score', 0)
                    candidate['filter_reasoning'] = sel.get('reasoning', '')
                    selected.append(candidate)
            return selected
        except Exception as e:
            print(f"⚠️  Batch filter error: {e}")
            return sorted(batch, key=lambda x: x['vector_score'], reverse=True)[:5]

    def __call__(self, state: TestCaseComparisonState) -> TestCaseComparisonState:
        old_test_case = state['old_test_case']
        candidates = state['top_candidates']
        log = []
        log.append(f"[Agent 2] Filtering {len(candidates)} candidates in batches of {self.batch_size}")

        try:
            all_filtered = []
            for i in range(0, len(candidates), self.batch_size):
                batch = candidates[i:i + self.batch_size]
                batch_num = (i // self.batch_size) + 1
                log.append(f"[Agent 2] Batch {batch_num} ({len(batch)} candidates)...")
                top_in_batch = self.filter_batch(old_test_case, batch)
                all_filtered.extend(top_in_batch)
                time.sleep(1)

            # Sort by relevance score and keep top 15 for final analysis
            all_filtered.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            state['filtered_candidates'] = all_filtered[:15]
            log.append(f"[Agent 2] Passed {len(state['filtered_candidates'])} candidates to final selector")
            state['processing_log'] = log

        except Exception as e:
            error_msg = f"[Agent 2 ERROR] {str(e)}"
            log.append(error_msg)
            state['errors'] = state.get('errors', []) + [error_msg]
            state['processing_log'] = log

        return state


# ============================================================================
# AGENT 3: FINAL SELECTOR + DIFFERENCE ANALYSIS + LLM MATCHING SCORE
# ============================================================================

class FinalSelectorAgent:
    """
    Agent 3: Deep analysis to select top 3 NEW test case matches.
    For each match, produces:
      - LLM matching score (0-100)
      - Difference analysis (what old covers that new doesn't)
      - What is implemented in old but missing in new
    """

    def __init__(self, genai_client: genai.Client, model_name: str = "gemini-2.5-pro"):
        self.client = genai_client
        self.model_name = model_name

    @exponential_backoff(max_retries=5, base_delay=2.0, max_delay=120.0)
    def llm_call(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=GenerateContentConfig(temperature=0.1, response_mime_type="application/json")
        )
        return response.text

    def __call__(self, state: TestCaseComparisonState) -> TestCaseComparisonState:
        old_test_case = state['old_test_case']
        candidates = state['filtered_candidates']
        log = []
        log.append(f"[Agent 3] Final analysis on {len(candidates)} candidates → select top 3")

        try:
            candidates_text = ""
            for i, c in enumerate(candidates[:10], 1):
                candidates_text += f"\n[CANDIDATE {i}]\nID: {c['id']}\nText: {c['text']}\n"

            prompt = f"""You are a senior QA analyst performing a detailed test case migration analysis.

An OLD test case from a legacy booking engine needs to be mapped to the best matching NEW test cases from a shopping cart system. You must select the TOP 3 best matches and provide a deep comparison.

**OLD TEST CASE (from legacy booking engine):**
{old_test_case}

**NEW TEST CASE CANDIDATES:**
{candidates_text}

For each of the TOP 3 matches, analyze:
1. How well does the new test case cover the old test case's functionality?
2. What specific functionality/behavior exists in the OLD test case but is NOT covered by the NEW test case?
3. What is the overall matching quality?

Respond in JSON:
{{
  "top_3_matches": [
    {{
      "candidate_number": 1,
      "match_confidence": "HIGH",
      "llm_matching_score": 85,
      "difference_analysis": "Detailed explanation of differences between old and new test case. What does the old test case verify that the new one does not?",
      "implemented_in_old_not_in_new": "Specific features, validations, or flows that exist in the old test case but are missing or not covered in the new test case",
      "coverage_summary": "FULL or PARTIAL or MINIMAL — how much of the old TC is covered by the new TC"
    }}
  ]
}}

Rules:
- candidate_number is 1-indexed matching the candidates above
- llm_matching_score: 0-100 based on functional similarity (NOT just keyword overlap)
  - 90-100: Near-identical test intent and coverage
  - 70-89: Same feature area, minor differences in scope
  - 50-69: Related functionality, significant gaps
  - 0-49: Loosely related or poor match
- match_confidence: HIGH (score >= 75), MEDIUM (50-74), LOW (< 50)
- difference_analysis: Be specific about WHAT differs, not vague
- implemented_in_old_not_in_new: List concrete items (e.g., "error handling for invalid dates", "cache expiration check", "multi-passenger pricing")
- Select exactly 3 matches (even if quality is low, pick the best 3 available)
"""

            result = json.loads(self.llm_call(prompt))
            final_matches = []

            for match in result.get('top_3_matches', [])[:3]:
                idx = match['candidate_number'] - 1
                if 0 <= idx < len(candidates):
                    candidate = candidates[idx].copy()
                    candidate.update({
                        'match_confidence': match.get('match_confidence', 'LOW'),
                        'llm_matching_score': match.get('llm_matching_score', 0),
                        'difference_analysis': match.get('difference_analysis', 'N/A'),
                        'implemented_in_old_not_in_new': match.get('implemented_in_old_not_in_new', 'N/A'),
                        'coverage_summary': match.get('coverage_summary', 'N/A'),
                    })
                    final_matches.append(candidate)

            log.append(f"[Agent 3] Selected {len(final_matches)} final matches with difference analysis")
            state['final_matches'] = final_matches
            state['processing_log'] = log

        except Exception as e:
            error_msg = f"[Agent 3 ERROR] {str(e)}"
            log.append(error_msg)
            state['errors'] = state.get('errors', []) + [error_msg]
            state['processing_log'] = log

        return state


# ============================================================================
# LANGGRAPH WORKFLOW BUILDER
# ============================================================================

class TestCaseComparisonWorkflow:
    """
    3-agent pipeline:
      Agent 1 (Retrieval)     → Vector search for top-50 NEW TCs
      Agent 2 (Batch Filter)  → LLM batch filter to top-15
      Agent 3 (Final Selector)→ Deep analysis, top 3 with scores + diffs
    """

    def __init__(self, qdrant_url: str, qdrant_api_key: str, gcp_project_id: str,
                 gcp_location: str, service_account_path: str,
                 collection_name: str = "test_case_comparisons",
                 embedding_model_name: str = "gemini-embedding-001",
                 llm_model_name: str = "gemini-2.5-pro"):

        print("🔧 Initializing Multi-Agent Comparison Workflow...")

        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=120,
            prefer_grpc=True,
        )
        print("  ✅ Qdrant connected")

        credentials = service_account.Credentials.from_service_account_file(
            service_account_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        vertexai.init(project=gcp_project_id, location=gcp_location, credentials=credentials)
        self.embedding_model = TextEmbeddingModel.from_pretrained(embedding_model_name)
        print(f"  ✅ Embedding model: {embedding_model_name}")

        self.genai_client = genai.Client(vertexai=True, project=gcp_project_id, location=gcp_location)
        self.llm_model_name = llm_model_name
        print(f"  ✅ LLM model: {llm_model_name}")

        # Build agents
        self.retrieval_agent = RetrievalAgent(self.qdrant_client, self.embedding_model, collection_name)
        self.batch_filter_agent = BatchSemanticFilterAgent(self.genai_client, llm_model_name, batch_size=10)
        self.final_selector_agent = FinalSelectorAgent(self.genai_client, llm_model_name)

        self.workflow = self._build_workflow()
        print("✅ Workflow ready! (3 agents: Retrieval → Batch Filter → Final Selector)\n")

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(TestCaseComparisonState)

        workflow.add_node("retrieval", self.retrieval_agent)
        workflow.add_node("batch_filter", self.batch_filter_agent)
        workflow.add_node("final_selector", self.final_selector_agent)

        workflow.set_entry_point("retrieval")
        workflow.add_edge("retrieval", "batch_filter")
        workflow.add_edge("batch_filter", "final_selector")
        workflow.add_edge("final_selector", END)

        return workflow.compile()

    def compare_test_case(self, old_test_case: str, old_test_case_id: str = "",
                          old_row_index: int = 0) -> Dict:
        initial_state: TestCaseComparisonState = {
            'old_test_case': old_test_case,
            'old_test_case_id': old_test_case_id,
            'old_row_index': old_row_index,
            'top_candidates': [],
            'filtered_candidates': [],
            'final_matches': [],
            'processing_log': [],
            'errors': [],
        }
        return self.workflow.invoke(initial_state)


# ============================================================================
# XLSX PROCESSOR — Reads Final_Sheet.xlsx, iterates OLD TCs, writes results
# ============================================================================

class XLSXTestCaseProcessor:
    """
    Reads Final_Sheet.xlsx → iterates over OLD test cases →
    finds top-3 matching NEW test cases → writes Comparison.xlsx
    """

    OLD_TC_COLUMN = "Old Test Case Scenario"
    NEW_TC_COLUMN = "New Test Case Scenerio"

    def __init__(self, workflow: TestCaseComparisonWorkflow):
        self.workflow = workflow

    def process_all(self, input_file: str, output_file: str,
                    start_from: int = 0, limit: Optional[int] = None):
        """
        Process all OLD test cases and find matching NEW test cases.

        Args:
            input_file:  Path to Final_Sheet.xlsx
            output_file: Path for output Comparison.xlsx
            start_from:  Row index to resume from (for crash recovery)
            limit:       Max rows to process (None = all)
        """
        print(f"📂 Reading: {input_file}")
        df = pd.read_excel(input_file)
        print(f"  Total rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")

        # Get only rows with non-null OLD test cases
        old_tc_mask = df[self.OLD_TC_COLUMN].notna()
        old_tc_indices = df[old_tc_mask].index.tolist()
        print(f"  Old TCs (non-null): {len(old_tc_indices)}")

        # Apply start_from and limit
        indices_to_process = old_tc_indices[start_from:]
        if limit:
            indices_to_process = indices_to_process[:limit]

        print(f"  Processing: {len(indices_to_process)} old test cases (start_from={start_from})")

        results = []
        total = len(indices_to_process)
        failed_count = 0

        for progress_idx, row_idx in enumerate(indices_to_process):
            old_text = str(df.at[row_idx, self.OLD_TC_COLUMN]).strip()

            print(f"\n{'='*70}")
            print(f"🔄 [{progress_idx + 1}/{total}] Row {row_idx}")
            print(f"   Old TC: {old_text[:100]}...")
            print(f"{'='*70}")

            try:
                result = self.workflow.compare_test_case(
                    old_test_case=old_text,
                    old_test_case_id=f"OLD_TC_{row_idx:04d}",
                    old_row_index=row_idx,
                )

                if result['final_matches']:
                    for rank, match in enumerate(result['final_matches'], 1):
                        results.append({
                            'Row #': row_idx,
                            'Old Test Case': old_text,
                            'Match Rank': rank,
                            'Matching New Test Case': match['text'],
                            'LLM Matching Score': match.get('llm_matching_score', 0),
                            'Match Confidence': match.get('match_confidence', 'N/A'),
                            'Coverage': match.get('coverage_summary', 'N/A'),
                            'Difference Analysis': match.get('difference_analysis', 'N/A'),
                            'Implemented in Old but Not in New': match.get('implemented_in_old_not_in_new', 'N/A'),
                        })

                    best = result['final_matches'][0]
                    print(f"  ✅ Best match: score={best.get('llm_matching_score', 0)} "
                          f"conf={best.get('match_confidence', '?')} "
                          f"coverage={best.get('coverage_summary', '?')}")
                else:
                    results.append({
                        'Row #': row_idx,
                        'Old Test Case': old_text,
                        'Match Rank': 1,
                        'Matching New Test Case': 'No matching new test case found',
                        'LLM Matching Score': 0,
                        'Match Confidence': 'NONE',
                        'Coverage': 'NO_COVERAGE',
                        'Difference Analysis': 'No suitable new test case found in the shopping cart system',
                        'Implemented in Old but Not in New': old_text,
                    })
                    print(f"  ⚠️ No match found")

                # Log errors if any
                if result.get('errors'):
                    for err in result['errors']:
                        print(f"  ⚠️ {err}")

            except Exception as e:
                failed_count += 1
                print(f"  ❌ Error: {str(e)}")
                results.append({
                    'Row #': row_idx,
                    'Old Test Case': old_text,
                    'Match Rank': 1,
                    'Matching New Test Case': f'ERROR: {str(e)}',
                    'LLM Matching Score': 0,
                    'Match Confidence': 'ERROR',
                    'Coverage': 'ERROR',
                    'Difference Analysis': f'Processing failed: {str(e)}',
                    'Implemented in Old but Not in New': 'N/A',
                })

            # Pace API calls
            time.sleep(2)

            # Periodic save every 50 rows
            if (progress_idx + 1) % 50 == 0:
                checkpoint_file = output_file.replace('.xlsx', f'_checkpoint_{progress_idx + 1}.xlsx')
                self._write_results(results, checkpoint_file)
                print(f"\n💾 Checkpoint saved: {checkpoint_file} ({len(results)} rows)")

        # Final save
        self._write_results(results, output_file)

        print(f"\n{'='*70}")
        print(f"✅ PROCESSING COMPLETE")
        print(f"   Total old TCs processed: {total}")
        print(f"   Result rows written:     {len(results)}")
        print(f"   Failed:                  {failed_count}")
        print(f"   Output:                  {output_file}")
        print(f"{'='*70}")

        return results

    def _write_results(self, results: List[Dict], output_file: str):
        """Write results to formatted XLSX with styling"""
        wb = Workbook()
        ws = wb.active
        ws.title = "Old vs New TC Comparison"

        headers = [
            'Row #',
            'Old Test Case',
            'Match Rank',
            'Matching New Test Case',
            'LLM Matching Score',
            'Match Confidence',
            'Coverage',
            'Difference Analysis',
            'Implemented in Old but Not in New',
        ]

        # --- Styles ---
        header_font = Font(bold=True, color='FFFFFF', size=11)
        header_fill = PatternFill('solid', fgColor='2F5496')
        header_alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        thin_border = Border(
            left=Side(style='thin'), right=Side(style='thin'),
            top=Side(style='thin'), bottom=Side(style='thin'),
        )

        # Confidence color fills
        confidence_fills = {
            'HIGH': PatternFill('solid', fgColor='C6EFCE'),    # Green
            'MEDIUM': PatternFill('solid', fgColor='FFEB9C'),  # Yellow
            'LOW': PatternFill('solid', fgColor='FFC7CE'),     # Red
            'NONE': PatternFill('solid', fgColor='D9D9D9'),    # Gray
            'ERROR': PatternFill('solid', fgColor='FFC7CE'),   # Red
        }

        # --- Headers ---
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
            cell.border = thin_border

        # --- Data rows ---
        for row_idx, result in enumerate(results, 2):
            for col_idx, header in enumerate(headers, 1):
                value = result.get(header, '')
                # Convert lists/dicts to strings (LLM sometimes returns structured data)
                if isinstance(value, list):
                    value = ', '.join(str(v) for v in value)
                elif isinstance(value, dict):
                    value = json.dumps(value, ensure_ascii=False)
                elif not isinstance(value, (str, int, float, type(None))):
                    value = str(value)
                cell = ws.cell(row=row_idx, column=col_idx, value=value)
                cell.alignment = Alignment(vertical='top', wrap_text=True)
                cell.border = thin_border

                # Color-code confidence column
                if header == 'Match Confidence':
                    fill = confidence_fills.get(str(value).upper(), None)
                    if fill:
                        cell.fill = fill

                # Color-code score column
                if header == 'LLM Matching Score' and isinstance(value, (int, float)):
                    if value >= 75:
                        cell.fill = PatternFill('solid', fgColor='C6EFCE')
                    elif value >= 50:
                        cell.fill = PatternFill('solid', fgColor='FFEB9C')
                    else:
                        cell.fill = PatternFill('solid', fgColor='FFC7CE')

        # --- Column widths ---
        widths = {
            'A': 8,   # Row #
            'B': 55,  # Old Test Case
            'C': 10,  # Match Rank
            'D': 55,  # Matching New Test Case
            'E': 16,  # LLM Score
            'F': 16,  # Confidence
            'G': 14,  # Coverage
            'H': 60,  # Difference Analysis
            'I': 60,  # Implemented in Old not in New
        }
        for col, width in widths.items():
            ws.column_dimensions[col].width = width

        # Freeze header + Row# column
        ws.freeze_panes = 'B2'

        # --- Summary sheet ---
        ws_summary = wb.create_sheet("Summary")
        ws_summary['A1'] = "Metric"
        ws_summary['B1'] = "Value"
        ws_summary['A1'].font = Font(bold=True)
        ws_summary['B1'].font = Font(bold=True)

        total_old = len(set(r['Row #'] for r in results))
        high_matches = sum(1 for r in results if r.get('Match Confidence') == 'HIGH' and r.get('Match Rank') == 1)
        medium_matches = sum(1 for r in results if r.get('Match Confidence') == 'MEDIUM' and r.get('Match Rank') == 1)
        low_matches = sum(1 for r in results if r.get('Match Confidence') == 'LOW' and r.get('Match Rank') == 1)
        no_match = sum(1 for r in results if r.get('Match Confidence') in ('NONE', 'ERROR') and r.get('Match Rank') == 1)

        avg_score = 0
        scores = [r['LLM Matching Score'] for r in results if r.get('Match Rank') == 1 and isinstance(r.get('LLM Matching Score'), (int, float)) and r['LLM Matching Score'] > 0]
        if scores:
            avg_score = sum(scores) / len(scores)

        summary_data = [
            ("Total Old Test Cases Analyzed", total_old),
            ("Total Result Rows (top-3 per old TC)", len(results)),
            ("HIGH confidence matches (rank 1)", high_matches),
            ("MEDIUM confidence matches (rank 1)", medium_matches),
            ("LOW confidence matches (rank 1)", low_matches),
            ("No match found (rank 1)", no_match),
            ("Average LLM Score (rank 1)", round(avg_score, 1)),
            ("Generated At", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ]
        for i, (metric, value) in enumerate(summary_data, 2):
            ws_summary[f'A{i}'] = metric
            ws_summary[f'B{i}'] = value

        ws_summary.column_dimensions['A'].width = 40
        ws_summary.column_dimensions['B'].width = 20

        wb.save(output_file)
        print(f"📝 Saved: {output_file} ({len(results)} rows)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    import os

    # ========================
    # CONFIGURATION — UPDATE THESE
    # ========================

    SERVICE_ACCOUNT_PATH = r"C:\Users\gudladhana.harshith\Desktop\TTC Comparsion\vertex_credentials.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = SERVICE_ACCOUNT_PATH

    QDRANT_URL = "https://b5030dd4-154d-457a-b473-bb09f2b0ecab.us-east4-0.gcp.cloud.qdrant.io"
    QDRANT_API_KEY = "REDACTED_TOKEN"

    # Collection name — must match the indexer's collection name
    COLLECTION_NAME = "test_case_comparisons"

    GCP_PROJECT_ID = "dg-assistant-451309"
    GCP_LOCATION = "us-central1"
    EMBEDDING_MODEL = "gemini-embedding-001"
    LLM_MODEL = "gemini-2.5-pro"

    # Input/Output — UPDATE THESE PATHS
    INPUT_XLSX = r"C:\Users\gudladhana.harshith\Desktop\TTC Comparsion\Final Sheet.xlsx"
    OUTPUT_XLSX = r"C:\Users\gudladhana.harshith\Desktop\TTC Comparsion\comparsion_updated.xlsx"

    # Processing options
    START_FROM = 0    # Resume from this index (0 = start fresh)
    LIMIT = 10        # Process only 10 old test cases for initial review

    # ========================
    # EXECUTION
    # ========================

    workflow = TestCaseComparisonWorkflow(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        gcp_project_id=GCP_PROJECT_ID,
        gcp_location=GCP_LOCATION,
        service_account_path=SERVICE_ACCOUNT_PATH,
        collection_name=COLLECTION_NAME,
        embedding_model_name=EMBEDDING_MODEL,
        llm_model_name=LLM_MODEL,
    )

    processor = XLSXTestCaseProcessor(workflow)
    results = processor.process_all(
        input_file=INPUT_XLSX,
        output_file=OUTPUT_XLSX,
        start_from=START_FROM,
        limit=LIMIT,
    )


if __name__ == "__main__":
    main()
