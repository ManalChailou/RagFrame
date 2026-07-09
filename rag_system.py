import json
import logging
import os
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class CosmicRAGSystem:
    
    def __init__(self, 
                 knowledge_base_path: str = "rag_docs/cosmic_rag_update.jsonl",
                 model_name: str = "all-MiniLM-L6-v2",
                 cache_dir: str = "cache/"):
        
        self.knowledge_base_path = knowledge_base_path
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialiser le modèle d'embedding
        self.embedding_model = SentenceTransformer(model_name)
        
        # Charger ou créer les embeddings
        self.knowledge_chunks = []
        self.embeddings = None
        self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """Charge la base de connaissances et crée les embeddings"""
        
        # NEW in load_knowledge_base()
        kb_path = Path(self.knowledge_base_path)
        stamp = int(kb_path.stat().st_mtime) if kb_path.exists() else 0
        cache_version = "schema_v2"  # changes when embedded JSONL fields change
        cache_file = self.cache_dir / f"cosmic_embeddings_{kb_path.stem}_{self.model_name.replace('/', '_')}_{cache_version}_{stamp}.pkl"

        try:
            # Charger depuis le cache si disponible
            if cache_file.exists():
                logger.info("Loading embeddings from cache...")
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.knowledge_chunks = cache_data['chunks']
                    self.embeddings = cache_data['embeddings']
                logger.info(f"Loaded {len(self.knowledge_chunks)} chunks from cache")
                return
            
            # Charger et traiter la base de connaissances
            logger.info("Loading COSMIC knowledge base...")
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                for line in f:
                    chunk_data = json.loads(line.strip())
                    self.knowledge_chunks.append(chunk_data)
            
            logger.info(f"Loaded {len(self.knowledge_chunks)} knowledge chunks")
            
            # Créer les embeddings
            logger.info("Creating embeddings...")
            texts_to_embed = []
            for chunk in self.knowledge_chunks:
                # Combiner les métadonnées et le contenu pour un meilleur matching
                parts = [
                    chunk.get('id', ''),
                    chunk.get('app_domain', ''),
                    chunk.get('cosmic_component', ''),
                    chunk.get('functional_process', ''),
                    chunk.get('section', ''),
                    chunk.get('type', ''),
                    ' '.join(chunk.get('keywords', [])) if isinstance(chunk.get('keywords', []), list) else str(chunk.get('keywords', '')),
                    ' '.join(chunk.get('examples', [])) if isinstance(chunk.get('examples', []), list) else str(chunk.get('examples', '')),
                    chunk.get('content', '')
                ]
                text = ' '.join([str(p) for p in parts if p])
                texts_to_embed.append(text)
            
            self.embeddings = self.embedding_model.encode(texts_to_embed)
            
            # Sauver en cache
            logger.info("Saving embeddings to cache...")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'chunks': self.knowledge_chunks,
                    'embeddings': self.embeddings
                }, f)
            
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            raise
    
    def _chunk_matches_component(self, chunk: Dict, component_filter: Optional[str]) -> bool:
        """
        Match a retrieval filter against the JSONL schema.
        The current KB uses `cosmic_component`, not `domain`.
        Fallbacks are kept for compatibility with older KB files.
        """
        if not component_filter:
            return True

        wanted = component_filter.strip().lower()
        searchable_fields = [
            chunk.get("cosmic_component", ""),  # current JSONL field
            chunk.get("domain", ""),            # backward compatibility
            chunk.get("section", ""),
            chunk.get("type", ""),
        ]
        searchable = " ".join(str(v) for v in searchable_fields if v).lower()
        return wanted in searchable

    def _chunk_matches_app_domain(self, chunk: Dict, app_domain_filter: Optional[str]) -> bool:
        """
        Match application domain using the JSONL `app_domain` field.
        `general` chunks are always allowed because they contain common COSMIC rules.
        """
        if not app_domain_filter:
            return True

        wanted = app_domain_filter.strip().lower()
        value = str(chunk.get("app_domain", "") or "").lower()
        return wanted in value or "general" in value

    def _format_chunks_as_context(self, chunks: List[Dict]) -> str:
        """
        Return clean, prompt-ready RAG context.

        Important:
        - Do NOT inject technical metadata such as chunk id, app_domain,
          cosmic_component, type, or similarity score into the LLM prompt.
        - Keep only the human-readable knowledge/example content.
        - Chunk metadata remains available internally through logging and
          the retrieved chunk dictionaries.
        """
        context_parts = []

        for chunk in chunks:
            section = str(chunk.get("section", "") or "").strip()
            content = str(chunk.get("content", "") or "").strip()

            if not content:
                continue

            if section:
                context_parts.append(f"{section}:\n{content}")
            else:
                context_parts.append(content)

        return "\n\n".join(context_parts).strip() + ("\n\n" if context_parts else "")

    def retrieve_relevant_chunks(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.3,
        domain_filter: Optional[str] = None,
        app_domain_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve the most relevant chunks for a query.

        IMPORTANT:
        - `domain_filter` is kept as the public argument name for compatibility.
        - Internally it now matches the JSONL field `cosmic_component`.
        - `app_domain_filter` matches the JSONL field `app_domain` and keeps `general` chunks.
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            logger.warning("No embeddings available")
            return []

        try:
            query_embedding = self.embedding_model.encode([query])
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]

            # Filter by COSMIC component using the real JSONL field: cosmic_component
            if domain_filter:
                component_mask = np.array([
                    self._chunk_matches_component(chunk, domain_filter)
                    for chunk in self.knowledge_chunks
                ], dtype=float)
                similarities = similarities * component_mask

            # Filter by application domain using the real JSONL field: app_domain
            if app_domain_filter:
                app_mask = np.array([
                    self._chunk_matches_app_domain(chunk, app_domain_filter)
                    for chunk in self.knowledge_chunks
                ], dtype=float)
                similarities = similarities * app_mask

            top_indices = np.argsort(similarities)[::-1][:top_k]

            relevant_chunks = []
            for idx in top_indices:
                score = float(similarities[idx])
                if score >= min_similarity:
                    chunk = self.knowledge_chunks[idx].copy()
                    chunk["similarity_score"] = score
                    relevant_chunks.append(chunk)

            logger.info(
                "Retrieved %s chunks | component=%s | app_domain=%s | query=%s",
                len(relevant_chunks),
                domain_filter,
                app_domain_filter,
                query[:80]
            )
            return relevant_chunks

        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []

    def get_context_for_functional_users(self, requirements: List[str], app_domain: Optional[str] = None) -> str:
        """ Récupère le contexte pertinent pour l'identification des utilisateurs fonctionnels """

        query = f"functional users identification boundary interaction {' '.join(requirements[:2])}"
        chunks = self.retrieve_relevant_chunks(
            query, 
            top_k=3, 
            domain_filter="functional_users",
            app_domain_filter=app_domain
        )
        
        return self._format_chunks_as_context(chunks)
    
    def get_context_for_functional_processes(self, requirements: List[str], app_domain: Optional[str] = None) -> str:
        """ Récupère le contexte pertinent pour l'identification des processus fonctionnels """

        query = f"functional process triggering events data movements {' '.join(requirements[:2])}"
        chunks = self.retrieve_relevant_chunks(
            query, 
            top_k=4, 
            domain_filter="functional_processes",
            app_domain_filter=app_domain
        )
        
        return self._format_chunks_as_context(chunks)
    
    def get_context_for_data_groups(self, requirements: List[str], functional_processes: List[Dict], app_domain: Optional[str] = None) -> str:
        """ Récupère le contexte pertinent pour l'identification des groupes de données """

        fp_names = [fp.get('name', '') for fp in functional_processes]
        query = f"data groups object of interest data attributes {' '.join(fp_names)} {' '.join(requirements[:2])}"
        chunks = self.retrieve_relevant_chunks(
            query, 
            top_k=3, 
            domain_filter="data_groups",
            app_domain_filter=app_domain
        )
        
        return self._format_chunks_as_context(chunks)
    
    def get_context_for_sub_processes(self,requirements: List[str],functional_processes: List[Dict],data_groups: List[Dict],app_domain: Optional[str] = None) -> str:
        """
        Retrieve RAG context specifically for sub-process decomposition:
        - how to split FPs into steps
        - how steps map to elementary data movements (Entry, Exit, Read, Write)
        - domain-specific patterns/examples by app_domain
        """
        fp_names = [fp.get('name', '') for fp in functional_processes] if functional_processes else []
        dg_names = [dg.get('name', '') for dg in data_groups] if data_groups else []

        # Query that ties the three axes together (requirements, FP, DG)
        query = (
            "cosmic sub-process decomposition steps mapping to data movements "
            "Entry Exit Read Write validation forbidden patterns "
            f"{' '.join(fp_names)} {' '.join(dg_names)} {' '.join(requirements[:2])}"
        )

        # --- Retrieve both sub_processes and data_movements chunks ---
        subproc_chunks = self.retrieve_relevant_chunks(
            query,
            top_k=5,
            domain_filter="sub_processes",
            app_domain_filter=app_domain
        )

        datamove_chunks = self.retrieve_relevant_chunks(
            query,
            top_k=5,
            domain_filter="data_movements",
            app_domain_filter=app_domain
        )

        # --- Merge both types of context ---
        context = ""

        if subproc_chunks:
            context += "Sub-Processes Context:\n"
            context += self._format_chunks_as_context(subproc_chunks)

        if datamove_chunks:
            context += "Data Movements Context:\n"
            context += self._format_chunks_as_context(datamove_chunks)

        # --- Append concise movement rules (always useful) ---
        rules = self.retrieve_relevant_chunks(
            "Entry Exit Read Write rules guidelines forbidden patterns",
            top_k=4,
            domain_filter="data_movements",
            app_domain_filter=app_domain
        )

        if rules:
            rule_chunks = [r for r in rules if 'rules' in (r.get('type', '') or '').lower()]
            if rule_chunks:
                context += "Movement Rules:\n"
                context += self._format_chunks_as_context(rule_chunks)

        return context

    
    def get_context_for_data_movements(self, requirements: List[str], app_domain: Optional[str] = None) -> str:
        """ Récupère le contexte pertinent pour l'identification des mouvements de données """
        
        query = f"data movements Entry Exit Read Write boundary crossing {' '.join(requirements[:2])}"
        chunks = self.retrieve_relevant_chunks(
            query, 
            top_k=5, 
            domain_filter="data_movements",
            app_domain_filter=app_domain
        )
        
        context = self._format_chunks_as_context(chunks)

        # Ajouter des règles spécifiques
        rules_chunks = self.retrieve_relevant_chunks(
            "Entry Exit Read Write rules guidelines",
            top_k=4,
            domain_filter="data_movements",
            app_domain_filter=app_domain
        )

        if rules_chunks:
            rule_chunks = [chunk for chunk in rules_chunks if 'rules' in (chunk.get('type', '') or '').lower()]
            if rule_chunks:
                context += "Movement Rules:\n"
                context += self._format_chunks_as_context(rule_chunks)

        return context
    
    def get_validation_context(self) -> str:
        """ Récupère le contexte pour la validation des mesures """

        chunks = self.retrieve_relevant_chunks(
            "validation rules common mistakes quality assurance", 
            top_k=3, 
            domain_filter="quality_assurance"
        )
        
        context = "COSMIC Validation Context:\n\n"
        context += self._format_chunks_as_context(chunks)
        return context
    
    def search_specific_topic(self, topic: str, max_results: int = 5) -> List[Dict]:
        """ Recherche sur un sujet spécifique """
        return self.retrieve_relevant_chunks(topic, top_k=max_results)
    