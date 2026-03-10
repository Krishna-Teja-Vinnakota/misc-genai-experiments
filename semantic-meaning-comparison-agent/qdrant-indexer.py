

import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
from vertexai.language_models import TextEmbeddingModel
import vertexai
from google.oauth2 import service_account
import re
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import time
import hashlib
import json
from datetime import datetime
from collections import Counter


class DualTestCaseIndexer:
    """
    Indexes both New Shopping Cart and Old Booking Engine test cases
    from a single Excel file with two columns into Qdrant.
    
    Supports:
      - Unified collection (both sources tagged) OR
      - Separate collections per source
    """

    # Column names in Final_Sheet.xlsx
    NEW_TC_COLUMN = "New Test Case Scenerio"
    OLD_TC_COLUMN = "Old Test Case Scenario"

    def __init__(
        self,
        collection_name: str = "test_case_comparisons",
        embedding_model: str = "gemini-embedding-001",
        mode: str = "unified",  # "unified" | "separate"
    ):
        """
        Args:
            collection_name: Name of the Qdrant collection
            embedding_model: Embedding model identifier
            mode: "unified"  → single collection with `source` field
                  "separate" → two collections: {name}_new and {name}_old
        """
        # Configuration should be loaded from environment variables
        # See README.md for setup instructions
        
        self.qdrant_client = None  # Initialize with external credentials
        self.embedding_model = None  # Initialize with external credentials
        self.embedding_model_name = embedding_model

        self.base_collection_name = collection_name
        self.mode = mode
        self.embedding_dimension = 3072

        # Stats per source
        self.stats = {
            "new": {"total": 0, "successful": 0, "failed": 0, "skipped": 0},
            "old": {"total": 0, "successful": 0, "failed": 0, "skipped": 0},
            "processing_time": 0,
        }

        # Rate limiting
        self.last_api_call = time.time()
        self.min_api_interval = 0.1

    # ================================================================
    # COLLECTION HELPERS
    # ================================================================

    def _collection_names(self) -> Dict[str, str]:
        if self.mode == "unified":
            return {
                "new": self.base_collection_name,
                "old": self.base_collection_name,
            }
        return {
            "new": f"{self.base_collection_name}_new_shopping_cart",
            "old": f"{self.base_collection_name}_old_booking_engine",
        }

    def create_collections(self, recreate: bool = False):
        """Create required Qdrant collection(s)."""
        names = set(self._collection_names().values())
        existing = {c.name for c in self.qdrant_client.get_collections().collections}

        for name in names:
            if name in existing:
                if recreate:
                    print(f"🗑️  Deleting existing collection: {name}")
                    self.qdrant_client.delete_collection(name)
                else:
                    print(f"✅ Collection '{name}' already exists")
                    continue

            print(f"📦 Creating collection: {name}")
            self.qdrant_client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            print(f"✅ Collection '{name}' created")

    # ================================================================
    # TEXT CLEANING
    # ================================================================

    @staticmethod
    def clean_text(text: str) -> str:
        if pd.isna(text) or not text:
            return ""
        text = str(text)
        text = re.sub(r"^•\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n", text)
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        return text.strip()

    # ================================================================
    # METADATA EXTRACTION — NEW TEST CASES
    # ================================================================

    def extract_new_tc_metadata(self, text: str) -> Dict:
        """
        Metadata extraction tuned for New Shopping Cart test cases.
        These are short verification statements (avg 85 chars).
        Pattern: "Verify that ...", "Display ...", "Validate ..."
        """
        text_lower = text.lower()
        metadata: Dict = {}

        # --- Intent / Action Type ---
        intent_map = {
            "verify": r"^verify\b",
            "validate": r"^validate\b",
            "display": r"^display\b",
            "ensure": r"^ensure\b",
            "check": r"^check\b",
            "confirm": r"^confirm\b",
            "save": r"^save\b",
            "select": r"^select\b",
            "ability": r"^ability\b",
            "successful": r"^successful\b",
        }
        metadata["intent"] = "other"
        for intent, pattern in intent_map.items():
            if re.search(pattern, text_lower):
                metadata["intent"] = intent
                break

        # --- Domain Detection (multi-label, tuned for new TCs) ---
        domain_patterns = {
            "special_request": [r"special\s*request", r"special\s*need"],
            "insurance": [r"insurance", r"gsp", r"goal\s*seal", r"trip\s*protection", r"protection\s*plan"],
            "quote": [r"\bquote\b", r"quotation", r"quote\s*summary"],
            "booking": [r"\bbooking\b", r"\breservation\b", r"\bbook\b"],
            "payment": [r"\bpayment\b", r"\bdeposit\b", r"\bepd\b", r"early\s*payment", r"\btransaction\b", r"\brefund\b"],
            "pricing": [r"\bpric(e|ing)\b", r"\bcost\b", r"\bdiscount\b", r"\bsurcharge\b", r"\bcommission\b", r"\bmarkup\b"],
            "currency": [r"\bcurrency\b", r"\baud\b", r"\busd\b", r"\bgbp\b", r"\beur\b", r"\bnzd\b", r"\bcad\b", r"\bfiji\b"],
            "flight": [r"\bflight\b", r"\bairline\b", r"\bitinerary\b", r"\bseat\b", r"\bfare\b"],
            "hotel": [r"\bhotel\b", r"\broom\b", r"\baccommodation\b", r"\bcheck[- ]?in\b", r"\bcheck[- ]?out\b"],
            "tour": [r"\btour\b", r"\bexcursion\b", r"\bactivity\b", r"\bdeparture\b"],
            "passenger": [r"\bpassenger\b", r"\bpax\b", r"\btravell?er\b", r"\badult\b", r"\bchild\b", r"\binfant\b"],
            "cart": [r"\bcart\b", r"\bshopping\s*cart\b", r"\border\s*summary\b", r"\bline\s*item\b"],
            "search": [r"\bsearch\b", r"\bfilter\b", r"\bsort\b", r"\bresult\b"],
            "validation": [r"\bvalidat(e|ion)\b", r"\berror\b", r"\binvalid\b", r"\bmandatory\b", r"\brequired\b"],
            "ui_display": [r"\bdisplay\b", r"\bshow\b", r"\bhide\b", r"\bvisible\b", r"\bgreyed?\s*out\b", r"\btooltip\b"],
        }

        detected_domains = []
        domain_scores = {}
        for domain, patterns in domain_patterns.items():
            score = sum(1 for p in patterns if re.search(p, text_lower))
            domain_scores[domain] = score
            if score > 0:
                detected_domains.append(domain)

        metadata["domains"] = detected_domains if detected_domains else ["general"]
        metadata["primary_domain"] = (
            max(domain_scores.items(), key=lambda x: x[1])[0]
            if any(v > 0 for v in domain_scores.values())
            else "general"
        )

        # --- Test Type ---
        has_error = bool(re.search(r"\b(error|invalid|fail|exception|reject)", text_lower))
        has_validation = bool(re.search(r"\b(validat|mandatory|required|limit|restrict)", text_lower))
        has_ui = bool(re.search(r"\b(display|show|hide|visible|greyed|tooltip|popup|modal)", text_lower))
        has_calculation = bool(re.search(r"\b(calculat|comput|total|sum|amount)\b", text_lower))

        if has_error:
            metadata["test_type"] = "negative_test"
            metadata["test_category"] = "error_handling"
        elif has_calculation:
            metadata["test_type"] = "functional"
            metadata["test_category"] = "calculation"
        elif has_validation:
            metadata["test_type"] = "functional"
            metadata["test_category"] = "validation"
        elif has_ui:
            metadata["test_type"] = "ui"
            metadata["test_category"] = "display_verification"
        else:
            metadata["test_type"] = "functional"
            metadata["test_category"] = "feature_verification"

        # --- Complexity (new TCs are generally simpler) ---
        word_count = len(text.split())
        domain_count = len(detected_domains)
        complexity_score = 0
        if word_count > 30:
            complexity_score += 2
        elif word_count > 15:
            complexity_score += 1
        if domain_count > 2:
            complexity_score += 1
        if has_calculation:
            complexity_score += 1

        metadata["complexity"] = (
            "high" if complexity_score >= 3
            else "medium" if complexity_score >= 2
            else "low"
        )

        # --- Entities ---
        entities = []
        entity_patterns = {
            "gsp": r"\bgsp\b|goal\s*seal",
            "epd": r"\bepd\b|early\s*payment",
            "trip_protection": r"trip\s*protection",
            "order_summary": r"order\s*summary",
            "shopping_cart": r"shopping\s*cart",
            "quote_summary": r"quote\s*summary",
        }
        for ent, pat in entity_patterns.items():
            if re.search(pat, text_lower):
                entities.append(ent)

        # Currency entities
        for cur in ["aud", "usd", "gbp", "eur", "nzd", "cad", "fjd"]:
            if re.search(rf"\b{cur}\b", text_lower):
                entities.append(cur.upper())

        metadata["entities"] = list(set(entities))

        # --- Technical flags ---
        metadata["has_error_scenario"] = has_error
        metadata["has_price_calculation"] = has_calculation
        metadata["has_ui_check"] = has_ui
        metadata["has_currency_check"] = "currency" in detected_domains
        metadata["has_passenger_logic"] = "passenger" in detected_domains
        metadata["involves_insurance"] = "insurance" in detected_domains

        # --- Word count for reference ---
        metadata["word_count"] = word_count

        return metadata

    # ================================================================
    # METADATA EXTRACTION — OLD TEST CASES
    # ================================================================

    def extract_old_tc_metadata(self, text: str) -> Dict:
        """
        Metadata extraction tuned for Old Booking Engine test cases.
        These are longer narrative descriptions (avg 266 chars).
        Pattern: "This test scenario verifies that ..."
        """
        text_lower = text.lower()
        metadata: Dict = {}

        # --- Actor Detection ---
        user_actions = len(re.findall(
            r"\buser\s+(initiates?|selects?|enters?|submits?|clicks?|requests?|"
            r"adds?|removes?|updates?|navigates?|chooses?|confirms?|provides?|"
            r"attempts?|receives?|verifies?|benefits?)",
            text_lower,
        ))
        system_actions = len(re.findall(
            r"\bsystem\s+(retrieves?|saves?|prepares?|verifies?|confirms?|stores?|"
            r"checks?|updates?|sets?|configures?|generates?|calculates?|displays?|"
            r"returns?|processes?|validates?|identifies?|throws?)",
            text_lower,
        ))
        metadata["user_action_count"] = user_actions
        metadata["system_action_count"] = system_actions
        metadata["primary_actor"] = "user" if user_actions > system_actions else "system"

        # --- Domain Detection (multi-label) ---
        domain_patterns = {
            "flight": [
                r"\bflight\b", r"\bairline\b", r"\bitinerary\b", r"\bdeparture\b",
                r"\barrival\b", r"\bfare\s*basis\b", r"\bseat\b", r"\baircraft\b",
            ],
            "hotel": [
                r"\bhotel\b", r"\broom\b", r"\baccommodation\b", r"\bcheck[- ]in\b",
                r"\bcheck[- ]out\b", r"\blodging\b", r"\broom\s*type\b",
            ],
            "tour": [
                r"\btour\b", r"\bexcursion\b", r"\btour\s*group\b", r"\bactivity\b",
                r"\bdeparture\s*location\b", r"\btour\s*package\b",
            ],
            "booking": [
                r"\bbooking\b", r"\breservation\b", r"\bbook\b", r"\breserve\b",
                r"\bconfirmation\b", r"\bbooking\s*engine\b",
            ],
            "payment": [
                r"\bpayment\b", r"\bcredit\s*card\b", r"\btransaction\b", r"\bbilling\b",
                r"\bcharge\b", r"\brefund\b", r"\bdiscount\b", r"\bepd\b",
            ],
            "pricing": [
                r"\bpric(e|ing)\b", r"\bcost\b", r"\brate\b", r"\bquote\b", r"\bfare\b",
                r"\bsupplement\b", r"\bcalculat(e|ion)\b.*\bpric", r"\bbase\s*price\b",
                r"\bcommission\b", r"\bmarkup\b",
            ],
            "passenger": [
                r"\bpassenger\b", r"\btravell?er\b", r"\bguest\b", r"\boccupan(t|cy)\b",
                r"\badult\b", r"\bchild\b", r"\binfant\b",
            ],
            "search": [r"\bsearch\b", r"\bquery\b", r"\bfilter\b", r"\bsort\b", r"\bresult\b"],
            "cache": [r"\bcach(e|ing|ed)\b", r"\bexpiration\b", r"\bttl\b"],
            "validation": [
                r"\bvalidat(e|ion)\b", r"\bverif(y|ies|ication)\b",
                r"\binvalid\b", r"\berror\s*message\b",
            ],
            "date_time": [
                r"\bdate\b", r"\btime\b", r"\bdeparture\s*date\b", r"\barrival\s*date\b",
                r"\bdd/mm/yyyy\b", r"\btimestamp\b", r"\bfinal\s*payment\s*due\s*date\b",
            ],
            "service_status": [
                r"\bstatus\b", r"\boperational\b", r"\bdown\b",
                r"\bservice\s*status\b", r"\bhealth\b",
            ],
            "commission": [r"\bcommission\b", r"\bpolicy\b", r"\bagent\b"],
            "client": [r"\bclient\b", r"\bbrand\b", r"\bregion\b", r"\btravel\s*agent\b"],
        }

        detected_domains = []
        domain_scores = {}
        for domain, patterns in domain_patterns.items():
            score = sum(1 for p in patterns if re.search(p, text_lower))
            domain_scores[domain] = score
            if score > 0:
                detected_domains.append(domain)

        metadata["domains"] = detected_domains if detected_domains else ["general"]
        metadata["primary_domain"] = (
            max(domain_scores.items(), key=lambda x: x[1])[0]
            if any(v > 0 for v in domain_scores.values())
            else "general"
        )

        # --- Test Type ---
        has_error = bool(re.search(r"\b(error|exception|fail(ure|s)?|invalid|throws)\b", text_lower))
        has_validation = bool(re.search(r"\b(validat|verif|check)\b", text_lower))
        has_calculation = bool(re.search(r"\b(calculat|comput|determin).*\b(price|cost|total|sum)", text_lower))
        has_flow = user_actions >= 3 and system_actions >= 3

        if has_error:
            metadata["test_type"] = "negative_test"
            metadata["test_category"] = "error_handling"
        elif has_calculation:
            metadata["test_type"] = "functional"
            metadata["test_category"] = "calculation"
        elif has_validation:
            metadata["test_type"] = "functional"
            metadata["test_category"] = "validation"
        elif has_flow:
            metadata["test_type"] = "integration"
            metadata["test_category"] = "end_to_end_flow"
        else:
            metadata["test_type"] = "functional"
            metadata["test_category"] = "smoke_test"

        # --- Action Extraction ---
        all_actions = re.findall(
            r"(?:user|system)\s+(initiates?|selects?|enters?|submits?|clicks?|requests?|"
            r"adds?|removes?|updates?|navigates?|chooses?|confirms?|provides?|attempts?|"
            r"retrieves?|saves?|prepares?|verifies?|stores?|checks?|sets?|configures?|"
            r"generates?|calculates?|displays?|returns?|processes?|validates?|identifies?|throws?)",
            text_lower,
        )
        metadata["action_sequence"] = list(dict.fromkeys(all_actions))[:10]

        # --- Entities ---
        entities = []
        services = [
            "tour group store", "tropics data service", "tds",
            "assembly service", "cache", "database", "api",
            "tropics service", "booking engine",
        ]
        entities.extend([svc for svc in services if svc in text_lower])

        if "hotel" in detected_domains:
            for rt in ["single", "twin", "double", "suite", "triple"]:
                if rt in text_lower:
                    entities.append(f"{rt}_room")

        regions = ["us region", "gb", "uk", "europe", "asia"]
        entities.extend([r for r in regions if r in text_lower])

        metadata["entities"] = list(set(entities))

        # --- Technical flags ---
        metadata["involves_cache"] = bool(re.search(r"\bcach(e|ing)\b", text_lower))
        metadata["involves_database"] = "database" in text_lower or "db" in text_lower
        metadata["involves_api"] = "api" in text_lower or "endpoint" in text_lower
        metadata["involves_service_check"] = bool(
            re.search(r"(service|store|tds).*\b(status|check|operational)", text_lower)
        )
        metadata["has_error_scenario"] = has_error
        metadata["has_date_validation"] = bool(re.search(r"date.*\b(valid|invalid|format|dd/mm/yyyy)", text_lower))
        metadata["has_price_calculation"] = has_calculation

        # --- Complexity ---
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        word_count = len(text.split())
        metadata["step_count"] = len(lines)
        metadata["word_count"] = word_count

        complexity_score = 0
        if word_count > 60:
            complexity_score += 3
        elif word_count > 35:
            complexity_score += 2
        elif word_count > 20:
            complexity_score += 1
        if len(detected_domains) > 2:
            complexity_score += 1
        if has_calculation:
            complexity_score += 1
        if has_flow:
            complexity_score += 1

        metadata["complexity"] = (
            "high" if complexity_score >= 4
            else "medium" if complexity_score >= 2
            else "low"
        )

        # --- Business Flow ---
        flow_patterns = {
            "search_and_book": r"search.*book",
            "price_and_confirm": r"(pric|calculat).*confirm",
            "validate_and_error": r"validat.*error",
            "add_and_checkout": r"add.*checkout",
            "select_and_save": r"select.*save",
        }
        metadata["business_flow"] = "single_operation"
        for flow_name, pattern in flow_patterns.items():
            if re.search(pattern, text_lower, re.DOTALL):
                metadata["business_flow"] = flow_name
                break

        return metadata

    # ================================================================
    # EMBEDDING GENERATION
    # ================================================================

    def generate_embedding(self, text: str, retry_count: int = 3) -> Optional[List[float]]:
        current_time = time.time()
        elapsed = current_time - self.last_api_call
        if elapsed < self.min_api_interval:
            time.sleep(self.min_api_interval - elapsed)

        for attempt in range(retry_count):
            try:
                self.last_api_call = time.time()
                embeddings = self.embedding_model.get_embeddings([text])
                if embeddings and len(embeddings) > 0:
                    return embeddings[0].values
            except Exception as e:
                if attempt < retry_count - 1:
                    time.sleep((attempt + 1) * 2)
                else:
                    print(f"\n❌ Embedding failed after {retry_count} attempts: {e}")
        return None

    # ================================================================
    # ID GENERATION
    # ================================================================

    @staticmethod
    def generate_id(text: str, row_index: int, source: str) -> str:
        prefix = "NEW_TC" if source == "new" else "OLD_TC"
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"{prefix}_{row_index:04d}_{content_hash}"

    # ================================================================
    # MAIN PROCESSING
    # ================================================================

    def process_excel_file(
        self,
        excel_path: str,
        batch_size: int = 50,
    ) -> Dict[str, Tuple[int, int]]:
        """
        Process Final_Sheet.xlsx and index both columns.

        Returns dict with 'new' and 'old' keys, each a (successful, failed) tuple.
        """
        start_time = time.time()
        collection_names = self._collection_names()

        # --- Read Excel ---
        print(f"\n📖 Reading Excel file: {excel_path}")
        df = pd.read_excel(excel_path)
        print(f"✅ Loaded {len(df)} rows")
        print(f"📋 Columns: {list(df.columns)}")

        new_count = df[self.NEW_TC_COLUMN].notna().sum()
        old_count = df[self.OLD_TC_COLUMN].notna().sum()
        print(f"   New Test Cases: {new_count}")
        print(f"   Old Test Cases: {old_count}")

        # --- Process New Test Cases ---
        print(f"\n{'='*60}")
        print(f"🆕 INDEXING NEW SHOPPING CART TEST CASES ({new_count})")
        print(f"   → Collection: {collection_names['new']}")
        print(f"{'='*60}")
        self._index_column(
            df=df,
            column=self.NEW_TC_COLUMN,
            source="new",
            collection_name=collection_names["new"],
            extract_fn=self.extract_new_tc_metadata,
            batch_size=batch_size,
        )

        # --- Process Old Test Cases ---
        print(f"\n{'='*60}")
        print(f"📜 INDEXING OLD BOOKING ENGINE TEST CASES ({old_count})")
        print(f"   → Collection: {collection_names['old']}")
        print(f"{'='*60}")
        self._index_column(
            df=df,
            column=self.OLD_TC_COLUMN,
            source="old",
            collection_name=collection_names["old"],
            extract_fn=self.extract_old_tc_metadata,
            batch_size=batch_size,
        )

        self.stats["processing_time"] = time.time() - start_time

        return {
            "new": (self.stats["new"]["successful"], self.stats["new"]["failed"]),
            "old": (self.stats["old"]["successful"], self.stats["old"]["failed"]),
        }

    def _index_column(
        self,
        df: pd.DataFrame,
        column: str,
        source: str,  # "new" | "old"
        collection_name: str,
        extract_fn,
        batch_size: int,
    ):
        """Index a single column into a Qdrant collection."""
        points: List[PointStruct] = []
        failed_indices: List[int] = []

        # Use offset IDs to avoid collisions in unified mode
        id_offset = 0 if source == "new" else 100_000

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Indexing [{source}]"):
            self.stats[source]["total"] += 1
            text = row[column]

            # Skip empty
            if pd.isna(text) or not str(text).strip():
                self.stats[source]["skipped"] += 1
                continue

            cleaned = self.clean_text(text)
            if not cleaned:
                self.stats[source]["skipped"] += 1
                continue

            # Embedding
            embedding = self.generate_embedding(cleaned)
            if embedding is None:
                failed_indices.append(idx)
                self.stats[source]["failed"] += 1
                continue

            # Metadata
            metadata = extract_fn(cleaned)
            tc_id = self.generate_id(cleaned, idx, source)

            # Payload
            payload = {
                # Core
                "test_case_id": tc_id,
                "test_case_text": cleaned,
                "original_text": str(text),
                "row_index": int(idx),
                "source": "new_shopping_cart" if source == "new" else "old_booking_engine",
                "indexed_at": datetime.now().isoformat(),
                # Domain
                "primary_domain": metadata.get("primary_domain", "general"),
                "domains": metadata.get("domains", ["general"]),
                # Test classification
                "test_type": metadata.get("test_type", "functional"),
                "test_category": metadata.get("test_category", "unknown"),
                "complexity": metadata.get("complexity", "low"),
                "word_count": metadata.get("word_count", 0),
                # Entities
                "entities": metadata.get("entities", []),
                # Error flag
                "has_error_scenario": metadata.get("has_error_scenario", False),
                "has_price_calculation": metadata.get("has_price_calculation", False),
            }

            # Source-specific payload fields
            if source == "new":
                payload.update({
                    "intent": metadata.get("intent", "other"),
                    "has_ui_check": metadata.get("has_ui_check", False),
                    "has_currency_check": metadata.get("has_currency_check", False),
                    "has_passenger_logic": metadata.get("has_passenger_logic", False),
                    "involves_insurance": metadata.get("involves_insurance", False),
                })
            else:
                payload.update({
                    "primary_actor": metadata.get("primary_actor", "system"),
                    "user_action_count": metadata.get("user_action_count", 0),
                    "system_action_count": metadata.get("system_action_count", 0),
                    "business_flow": metadata.get("business_flow", "single_operation"),
                    "step_count": metadata.get("step_count", 1),
                    "action_sequence": metadata.get("action_sequence", []),
                    "involves_cache": metadata.get("involves_cache", False),
                    "involves_database": metadata.get("involves_database", False),
                    "involves_api": metadata.get("involves_api", False),
                    "involves_service_check": metadata.get("involves_service_check", False),
                    "has_date_validation": metadata.get("has_date_validation", False),
                })

            point = PointStruct(
                id=int(idx) + id_offset,
                vector=embedding,
                payload=payload,
            )
            points.append(point)
            self.stats[source]["successful"] += 1

            # Batch upsert
            if len(points) >= batch_size:
                self._upsert_batch(points, collection_name)
                points = []

        # Remaining
        if points:
            self._upsert_batch(points, collection_name)

        if failed_indices:
            print(f"\n⚠️  Failed [{source}] rows: {failed_indices[:20]}{'...' if len(failed_indices) > 20 else ''}")

    def _upsert_batch(self, points: List[PointStruct], collection_name: str):
        try:
            self.qdrant_client.upsert(collection_name=collection_name, points=points)
        except Exception as e:
            print(f"\n❌ Upsert error ({collection_name}): {e}")
            raise

    # ================================================================
    # STATISTICS
    # ================================================================

    def print_statistics(self):
        print("\n" + "=" * 80)
        print("📊 INDEXING STATISTICS")
        print("=" * 80)

        for source, label in [("new", "New Shopping Cart"), ("old", "Old Booking Engine")]:
            s = self.stats[source]
            print(f"\n  🏷️  {label}")
            print(f"     Total Rows:           {s['total']}")
            print(f"     Successfully Indexed: {s['successful']} ✅")
            print(f"     Failed:               {s['failed']} ❌")
            print(f"     Skipped (empty):      {s['skipped']} ⏭️")

        total_ok = self.stats["new"]["successful"] + self.stats["old"]["successful"]
        print(f"\n  ⏱️  Total Processing Time: {self.stats['processing_time']:.2f}s")
        if total_ok > 0:
            avg = self.stats["processing_time"] / total_ok
            print(f"  ⏱️  Avg Time per TC:      {avg:.3f}s")
            cost = (total_ok * 300) / 1000 * 0.00002
            print(f"  💰 Estimated Cost:       ${cost:.4f}")
        print("=" * 80)

    # ================================================================
    # VERIFICATION
    # ================================================================

    def verify_and_sample(self):
        coll_names = self._collection_names()

        for source, coll_name in coll_names.items():
            try:
                info = self.qdrant_client.get_collection(coll_name)
                print(f"\n✅ Collection Verified: {coll_name}")
                print(f"   Vector Size:  {info.config.params.vectors.size}")
                print(f"   Total Points: {info.points_count}")
            except Exception as e:
                print(f"❌ Verification error ({coll_name}): {e}")
                continue

        # Sample searches
        print(f"\n🔍 Sample Cross-Search Tests:")
        queries = [
            "flight booking with price calculation",
            "hotel room validation and error handling",
            "payment deposit currency processing",
            "special request for passengers",
            "insurance protection plan quote",
        ]

        target_coll = list(set(coll_names.values()))[0]  # Use first/unified collection
        for query in queries:
            embedding = self.generate_embedding(query)
            if not embedding:
                continue
            results = self.qdrant_client.search(
                collection_name=target_coll,
                query_vector=embedding,
                limit=3,
            )
            print(f"\n  Query: '{query}'")
            for i, r in enumerate(results, 1):
                src = r.payload.get("source", "?")
                domain = r.payload.get("primary_domain", "?")
                ttype = r.payload.get("test_type", "?")
                print(f"  {i}. Score: {r.score:.4f} | Source: {src} | Domain: {domain} | Type: {ttype}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("🚀 DUAL TEST CASE INDEXING — New Shopping Cart + Old Booking Engine")
    print("=" * 80)

    # ========================
    # CONFIGURATION — UPDATE THESE
    # ========================

    # Qdrant
    QDRANT_URL = "https://b5030dd4-154d-457a-b473-bb09f2b0ecab.us-east4-0.gcp.cloud.qdrant.io"
    QDRANT_API_KEY = "REDACTED_TOKEN"
    COLLECTION_NAME = "test_case_comparisons"

    # Google Cloud — UPDATE THESE
    GCP_PROJECT_ID = "dg-assistant-451309"
    GCP_LOCATION = "us-central1"
    SERVICE_ACCOUNT_PATH = r"C:\Users\gudladhana.harshith\Desktop\TTC Comparsion\vertex_credentials.json"

    # Excel file — UPDATE THIS
    EXCEL_FILE_PATH = r"C:\Users\gudladhana.harshith\Desktop\TTC Comparsion\Final Sheet.xlsx"

    # Processing
    BATCH_SIZE = 50
    RECREATE_COLLECTIONS = True
    MODE = "unified"  # "unified" = single collection | "separate" = two collections

    # ========================
    # EXECUTION
    # ========================

    try:
        indexer = DualTestCaseIndexer(
            qdrant_url=QDRANT_URL,
            qdrant_api_key=QDRANT_API_KEY,
            gcp_project_id=GCP_PROJECT_ID,
            gcp_location=GCP_LOCATION,
            service_account_path=SERVICE_ACCOUNT_PATH,
            collection_name=COLLECTION_NAME,
            embedding_model="gemini-embedding-001",
            mode=MODE,
        )

        # Create collection(s)
        indexer.create_collections(recreate=RECREATE_COLLECTIONS)

        # Process and index both columns
        results = indexer.process_excel_file(
            excel_path=EXCEL_FILE_PATH,
            batch_size=BATCH_SIZE,
        )

        # Statistics
        indexer.print_statistics()

        # Verify
        indexer.verify_and_sample()

        new_ok, new_fail = results["new"]
        old_ok, old_fail = results["old"]

        print(f"\n✅ Indexing completed successfully!")
        print(f"   🆕 New Shopping Cart:    {new_ok} indexed, {new_fail} failed")
        print(f"   📜 Old Booking Engine:   {old_ok} indexed, {old_fail} failed")
        print(f"   📊 Total:                {new_ok + old_ok} test cases in Qdrant")
        print(f"   🔍 Ready for comparison queries!")

        return 0

    except Exception as e:
        print(f"\n❌ Indexing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())