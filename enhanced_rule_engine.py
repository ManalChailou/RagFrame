import json
import logging
import os
from typing import List, Dict, Tuple
from collections import defaultdict
import re

from models.cosmic_components import DataMovement, DataMovementType, SubProcess
from rag_system import CosmicRAGSystem

logger = logging.getLogger(__name__)

class EnhancedCosmicRuleEngine:

    def __init__(self, config_path: str = "config/cosmic_patterns.json"):
        self._load_patterns_and_rules(config_path)
        
        # Initialiser le système RAG
        try:
            self.rag_system = CosmicRAGSystem()
            logger.info("RAG system initialized in rule engine")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system in rule engine: {e}")
            self.rag_system = None

    def _load_patterns_and_rules(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pattern configuration file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Charger les patterns d'entités
        self.user_patterns = config["entity_patterns"].get("user", [])
        self.storage_patterns = config["entity_patterns"].get("storage", [])
        self.system_patterns = config["entity_patterns"].get("system", [])
        self.external_patterns = config["entity_patterns"].get("external", [])
        self.temporal_patterns = config["entity_patterns"].get("temporal", [])

        # Charger les règles de mouvement
        self.movement_rules = {}
        for movement_type_str, rule_list in config.get("movement_rules", {}).items():
            for verb, source, dest in rule_list:
                self.movement_rules[(verb, source, dest)] = DataMovementType[movement_type_str.upper()]

    def normalize_entity(self, entity: str) -> str:
        """Normalise les entités en utilisant les patterns de cosmic_patterns.json"""
        entity_lower = entity.lower().strip()
        if any(re.search(p, entity_lower) for p in self.external_patterns):
            return "External Application"
        if any(re.search(p, entity_lower) for p in self.temporal_patterns):
            return "Clock"
        if any(re.search(p, entity_lower) for p in self.storage_patterns):
            return "Storage"
        if any(re.search(p, entity_lower) for p in self.user_patterns):
            return "User"
        if any(re.search(p, entity_lower) for p in self.system_patterns):
            return "System"
        
        logger.debug(f"Entity '{entity}' not matched by any pattern, defaulting to System")
        return "System"

    def convert_sub_processes_to_movements(self, sub_processes: List[Dict]) -> List[Dict]:
        """Convertit les sous-processus en mouvements de données normalisés avec validation COSMIC"""
        movements = []

        print("\nSub processes to movements\n", sub_processes, "\n")

        for sp in sub_processes:
            # Supporte les deux formats : CamelCase et snake_case
            action = sp.get("ActionVerb") or sp.get("action_verb")
            data_group = sp.get("MovedDataGroup") or sp.get("moved_data_group") or sp.get("DataGroup") or sp.get("data_group")
            object_of_interest = (sp.get("ObjectOfInterest") or sp.get("object_of_interest") or "")
            source = sp.get("Source") or sp.get("source")
            destination = sp.get("Destination") or sp.get("destination")
            process_name = sp.get("process_name")

            if not all([action, data_group, source, destination]):
                logger.warning(f"Skipping incomplete sub-process: {sp}")
                continue

            #FILTRAGE PRÉCOCE des System→System
            norm_source = self.normalize_entity(source)
            norm_dest = self.normalize_entity(destination)

            print(f"\n[Normalized] src='{source}' -> {norm_source} | dst='{destination}' -> {norm_dest}\n")

            if norm_source == "System" and norm_dest == "System":
                logger.warning(f"🚫 EXCLUDED System→System sub-process: {action} {data_group} in {process_name}")
                continue

            movement_tuple = (action, data_group, source, destination)
            
            # Validation et décomposition si nécessaire
            if self.validate_movement_compliance(movement_tuple):
                # Mouvement valide, ajouter directement
                movement = {
                    "action": action,
                    "data_group": data_group,
                    "object_of_interest": object_of_interest,
                    "source": source,
                    "destination": destination,
                    "functional_process": process_name or ""
                }
                movements.append(movement)
            else:
                # Mouvement interdit, décomposer
                decomposed = self.decompose_forbidden_movements(movement_tuple)
                for decomposed_tuple in decomposed:
                    decomposed_action, decomposed_dg, decomposed_src, decomposed_dest = decomposed_tuple
                    movement = {
                        "action": decomposed_action,
                        "data_group": decomposed_dg,
                        "object_of_interest": object_of_interest,
                        "source": decomposed_src,
                        "destination": decomposed_dest,
                        "functional_process": process_name or ""
                    }
                    movements.append(movement)

        logger.info(f"Converted {len(movements)} movements (after COSMIC validation and decomposition)")
        return movements

    def apply_rule_1(self, functional_processes: List[Dict]) -> List[Dict]:
        """RU1: Un TEn ne peut déclencher qu'un seul processus fonctionnel"""
        logger.info("Applying RU1: Triggering Entry uniqueness (Enhanced)")
        
        seen_entries = {}
        merged_processes = []
        merge_count = 0
        
        for fp in functional_processes:
            te = fp.get("TriggeringEntry", "").strip()
            fu = fp.get("FunctionalUser", "").strip()
            
            if not te:
                logger.warning(f"Process {fp.get('name')} has no triggering entry")
                merged_processes.append(fp)
                continue
            
            # Seuls les processus avec même TEn ET même FU peuvent être fusionnés
            composite_key = f"{te.lower().strip()}#{fu.lower().strip()}"
            
            if composite_key in seen_entries:
                # Fusion seulement si même TEn ET même FU
                existing = seen_entries[composite_key]
                existing["name"] = f"{existing['name']}+{fp['name']}"
                existing["description"] = f"{existing.get('description', '')} | {fp.get('description', '')}"
                merge_count += 1
                logger.info(f"Merged process {fp['name']} with {existing['name']} (same TEn+FU: {te}#{fu})")
            else:
                seen_entries[composite_key] = fp
                merged_processes.append(fp)
        
        logger.info(f"RU1 completed: {merge_count} merges, {len(merged_processes)} final processes")
        return merged_processes
    
    def apply_rule_2(self, data_movements: List[Tuple]) -> List[Tuple]:
        """RU2: Éliminer les mouvements de données dupliqués"""
        logger.info("Applying RU2: Duplicate movement removal")
        
        initial_count = len(data_movements)
        unique_movements = list(set(data_movements))
        duplicates_removed = initial_count - len(unique_movements)
        
        logger.info(f"RU2 completed: {duplicates_removed} duplicates removed, {len(unique_movements)} unique movements")
        return unique_movements
    
    def apply_rule_3(self, movement_data: Tuple[str, str, str, str]) -> DataMovementType:
        """RU3: Classifier le type de mouvement de données avec support RAG"""
        action, data_group, source, dest = movement_data
        
        # Normalisation des entités
        norm_source = self.normalize_entity(source)
        norm_dest = self.normalize_entity(dest)
        
        # Normalisation de l'action
        action_norm = action.strip().title()
        
        # Construire la clé de recherche
        lookup_key = (action_norm, norm_source, norm_dest)
        
        # Recherche directe
        if lookup_key in self.movement_rules:
            result = self.movement_rules[lookup_key]
            logger.debug(f"Direct match: {lookup_key} -> {result.value}")
            return result
        
        # Recherche par patterns
        movement_type = self._classify_by_pattern(action_norm, norm_source, norm_dest)
        
        logger.debug(f"Pattern match: {lookup_key} -> {movement_type.value}")
        return movement_type
    
    def _classify_by_pattern(self, action: str, source: str, dest: str) -> DataMovementType:
        """Classification par patterns avec validation COSMIC stricte"""
        
        # Vérification de conformité avant classification
        if not self.validate_movement_compliance((action, "", source, dest)):
            logger.error(f"Attempting to classify forbidden movement: {source}→{dest}")
            return DataMovementType.ENTRY
        
        # Règles COSMIC strictes par direction
        if source == "User" and dest == "System":
            return DataMovementType.ENTRY
        elif source == "System" and dest == "User":
            return DataMovementType.EXIT
        elif source == "Storage" and dest == "System":
            return DataMovementType.READ
        elif source == "System" and dest == "Storage":
            return DataMovementType.WRITE
        elif source == "External Application" and dest == "System":
            return DataMovementType.ENTRY  # Données entrantes depuis externe
        elif source == "System" and dest == "External Application":
            return DataMovementType.EXIT   # Données sortantes vers externe
        elif source == "Clock" and dest == "System":
            return DataMovementType.ENTRY  # Événements temporels
        
        # Cas spéciaux basés sur l'action pour les mouvements valides uniquement
        action_lower = action.lower()
        
        # Patterns d'entrée (User→System)
        entry_verbs = ["input", "enter", "submit", "send", "request", "login", "create", "upload", "provide", "select"]
        if any(word in action_lower for word in entry_verbs) and source == "User" and dest == "System":
            return DataMovementType.ENTRY
            
        # Patterns de sortie (System→User)
        exit_verbs = ["output", "display", "show", "print", "view", "generate", "present", "notify", "report"]
        if any(word in action_lower for word in exit_verbs) and source == "System" and dest == "User":
            return DataMovementType.EXIT
            
        # Patterns de lecture (Storage→System)
        read_verbs = ["read", "retrieve", "load", "fetch", "get", "query", "access", "find", "search"]
        if any(word in action_lower for word in read_verbs) and source == "Storage" and dest == "System":
            return DataMovementType.READ
            
        # Patterns d'écriture (System→Storage)
        write_verbs = ["write", "store", "save", "update", "insert", "delete", "persist", "modify", "record", "archive"]
        if any(word in action_lower for word in write_verbs) and source == "System" and dest == "Storage":
            return DataMovementType.WRITE
        
        # Fallback sécurisé
        if source == "User" and dest == "System":
            return DataMovementType.ENTRY
        elif source == "System" and dest == "User":
            return DataMovementType.EXIT
        elif source == "Storage" and dest == "System":
            return DataMovementType.READ
        elif source == "System" and dest == "Storage":
            return DataMovementType.WRITE
        elif source == "External Application" and dest == "System":
            return DataMovementType.ENTRY
        elif source == "System" and dest == "External Application":
            return DataMovementType.EXIT
        elif source == "Clock" and dest == "System":
            return DataMovementType.ENTRY
        else:
            logger.error(f"Unable to classify movement: {action} from {source} to {dest}")
            return DataMovementType.ENTRY
    
    def apply_rule_4(self, movements: List[DataMovement], functional_processes: List[Dict]) -> List[DataMovement]:
        """RU4: Assurer qu'un processus fonctionnel a au moins un Entry"""
        logger.info("Applying RU4: Entry validation (Enhanced)")

        # Grouper les mouvements par processus
        movements_by_fp = defaultdict(list)
        for m in movements:
            movements_by_fp[m.functional_process].append(m)

        added_entries = 0

        for fp in functional_processes:
            fp_name = fp.get("name", "Unknown")
            fp_user = fp.get("FunctionalUser", "User")
            triggering_entry = fp.get("TriggeringEntry", "Default_Data")
            fp_movements = movements_by_fp.get(fp_name, [])

            # Vérifie s'il y a déjà un Entry
            has_entry = any(m.movement_type == DataMovementType.ENTRY for m in fp_movements)
            if not has_entry:
                # Gestion spéciale pour les processus automatiques
                if fp_user.lower() == "system":
                    # Processus automatique → Utilisateur fonctionnel "Timer"
                    default_entry = DataMovement(
                        action_verb="Trigger",
                        data_group="Time_Event",
                        object_of_interest="Timer",
                        source="Clock",
                        destination="System",
                        movement_type=DataMovementType.ENTRY,
                        functional_process=fp_name
                    )
                    logger.info(f"[RU4] Added Clocl Entry for automatic process {fp_name}")
                else:
                    # Processus normal
                    default_entry = DataMovement(
                        action_verb="Input",
                        data_group=triggering_entry,
                        object_of_interest="",
                        source=fp_user,
                        destination="System",
                        movement_type=DataMovementType.ENTRY,
                        functional_process=fp_name
                    )
                    logger.info(f"[RU4] Added default Entry for process {fp_name}")
                
                movements.append(default_entry)
                added_entries += 1

        logger.info(f"[RU4] Completed: {added_entries} Entry(s) added")
        return movements
    
    def validate_cosmic_rules(self, movements: List[DataMovement]) -> List[str]:
        """Validation des règles COSMIC"""
        errors = []
        
        # Vérifier que tous les mouvements ont un type valide
        for i, movement in enumerate(movements):
            if not isinstance(movement.movement_type, DataMovementType):
                errors.append(f"Movement {i}: Invalid movement type {movement.movement_type}")
        
        # Vérifier les interdictions COSMIC
        for i, movement in enumerate(movements):
            norm_source = self.normalize_entity(movement.source)
            norm_dest = self.normalize_entity(movement.destination)
            
            # Interdictions strictes
            if norm_source == "Storage" and norm_dest == "User":
                errors.append(f"Movement {i}: Direct Storage→User forbidden in COSMIC")
            
            if norm_source == "User" and norm_dest == "Storage":
                errors.append(f"Movement {i}: Direct User→Storage forbidden in COSMIC")
                
            if norm_source == "System" and norm_dest == "System":
                errors.append(f"Movement {i}: Internal System→System should be excluded from CFP")
        
        # Validation par processus
        movements_by_process = defaultdict(list)
        for movement in movements:
            movements_by_process[movement.functional_process].append(movement)
        
        for process_name, process_movements in movements_by_process.items():
            entry_count = sum(1 for m in process_movements if m.movement_type == DataMovementType.ENTRY)
            
            if entry_count == 0:
                errors.append(f"Process '{process_name}': No Entry movement found")
        
        # Statistiques globales
        entry_count = sum(1 for m in movements if m.movement_type == DataMovementType.ENTRY)
        exit_count = sum(1 for m in movements if m.movement_type == DataMovementType.EXIT)
        read_count = sum(1 for m in movements if m.movement_type == DataMovementType.READ)
        write_count = sum(1 for m in movements if m.movement_type == DataMovementType.WRITE)
        
        if entry_count == 0:
            errors.append("No Entry movements found - every system should have at least one Entry")
        
        logger.info(f"COSMIC Validation: {entry_count} Entry, {read_count} Read, {write_count} Write, {exit_count} Exit")
        
        return errors
    
    def get_rag_enhanced_validation(self, movements: List[DataMovement]) -> Dict:
        """Validation améliorée avec contexte RAG"""
        validation_result = {
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "cosmic_compliance": True
        }
        
        # Validation de base
        base_errors = self.validate_cosmic_rules(movements)
        validation_result["errors"].extend(base_errors)
        
        if base_errors:
            validation_result["cosmic_compliance"] = False
        
        # Enrichissement avec le RAG si disponible
        if self.rag_system:
            try:
                # Obtenir des suggestions de validation
                suggestions = self._get_rag_validation_suggestions(movements)
                validation_result["suggestions"].extend(suggestions)
                
            except Exception as e:
                logger.warning(f"RAG validation enhancement failed: {e}")
        
        return validation_result
    
    def _get_rag_validation_suggestions(self, movements: List[DataMovement]) -> List[str]:
        """Utilise le RAG pour obtenir des suggestions de validation"""
        if not self.rag_system:
            return []
        
        suggestions = []
        
        try:
            # Rechercher des conseils de validation dans la base de connaissances
            validation_chunks = self.rag_system.retrieve_relevant_chunks(
                "validation checkpoints common mistakes quality assurance", 
                top_k=3,
                domain_filter="quality_assurance"
            )
            
            for chunk in validation_chunks:
                content = chunk.get('content', '')
                if 'validation' in content.lower():
                    if 'triggering entry' in content.lower():
                        entry_count = sum(1 for m in movements if m.movement_type == DataMovementType.ENTRY)
                        if entry_count == 0:
                            suggestions.append("Consider adding triggering Entry movements - COSMIC requires each functional process to have a triggering Entry")
                    
                    if 'data group' in content.lower():
                        suggestions.append("Ensure each data movement moves exactly one data group as per COSMIC principles")
        
        except Exception as e:
            logger.debug(f"Error getting RAG validation suggestions: {e}")
        
        return suggestions
    
    def validate_movement_compliance(self, movement_data: Tuple[str, str, str, str]) -> bool:
        """Valide qu'un mouvement respecte les règles COSMIC fondamentales"""
        action, data_group, source, dest = movement_data
        
        # Normaliser les entités
        norm_source = self.normalize_entity(source)
        norm_dest = self.normalize_entity(dest)
        
        # Règles d'interdiction strictes COSMIC
        forbidden_combinations = [
            ("Storage", "User"),    # Storage ne peut jamais aller directement vers User
            ("User", "Storage"),    # User ne peut jamais aller directement vers Storage
            ("System", "System"),   # Mouvements internes exclus du CFP
            ("External Application", "External Application"),  # Externe vers externe interdit
            ("Storage", "External Application"),  # Storage vers externe interdit
            ("External Application", "Storage"),  # Externe vers storage interdit
        ]
        
        for forbidden_source, forbidden_dest in forbidden_combinations:
            if norm_source == forbidden_source and norm_dest == forbidden_dest:
                logger.warning(f"Forbidden movement: {norm_source}→{norm_dest} for action '{action}'")
                return False
        
        return True
    
    def decompose_forbidden_movements(self, movement_data: Tuple[str, str, str, str]) -> List[Tuple[str, str, str, str]]:
        """Décompose les mouvements interdits en séquences valides COSMIC"""
        action, data_group, source, dest = movement_data
        
        norm_source = self.normalize_entity(source)
        norm_dest = self.normalize_entity(dest)
        
        # Décomposition Storage→User en Read + Exit
        if norm_source == "Storage" and norm_dest == "User":
            return [
                ("Read", data_group, "Storage", "System"),
                ("Display", data_group, "System", dest)
            ]
        
        # Décomposition User→Storage en Entry + Write  
        if norm_source == "User" and norm_dest == "Storage":
            return [
                ("Input", data_group, source, "System"),
                ("Store", data_group, "System", "Storage")
            ]
        
        # Mouvements System→System sont supprimés
        if norm_source == "System" and norm_dest == "System":
            logger.info(f"Excluding internal movement: {action} {data_group} System→System")
            return []
        
        # Mouvement valide, retourner tel quel
        return [movement_data]
    
    def process_movements(self, raw_movements: List[Dict], functional_processes: List[Dict] = None) -> List[DataMovement]:
        """Pipeline complet de traitement des mouvements avec support RAG - Version finale corrigée"""
        logger.info(f"Starting enhanced movement processing: {len(raw_movements)} raw movements")
        
        if functional_processes is None:
            functional_processes = []
        
        try:
            # Étape 0: Appliquer RU1 sur les processus fonctionnels (corrigé)
            if functional_processes:
                functional_processes = self.apply_rule_1(functional_processes)
            
            # Étape 1: Filtrer et convertir les mouvements valides uniquement
            valid_movements = []
            excluded_count = 0
            
            for m in raw_movements:
                if not all(key in m for key in ["action", "data_group", "source", "destination"]):
                    logger.warning(f"Skipping incomplete movement: {m}")
                    continue
                
                # Logging pour debug
                logger.debug(f"Processing movement: {m['action']} {m['data_group']} {m['source']}→{m['destination']} in {m.get('process_name', 'Unknown')}")
                
                # Normaliser pour vérification
                norm_source = self.normalize_entity(m["source"])
                norm_dest = self.normalize_entity(m["destination"])
                
                # FILTRAGE STRICT des mouvements interdits
                if norm_source == "System" and norm_dest == "System":
                    logger.warning(f"🚫 EXCLUDED System→System: {m['action']} {m['data_group']} in {m.get('process_name', 'Unknown')}")
                    excluded_count += 1
                    continue
                
                # Vérification supplémentaire pour les sources/destinations textuelles
                if (m["source"].lower().strip() == "system" and m["destination"].lower().strip() == "system"):
                    logger.warning(f"🚫 EXCLUDED literal System→System: {m['action']} {m['data_group']} in {m.get('process_name', 'Unknown')}")
                    excluded_count += 1
                    continue
                
                if norm_source == "Storage" and norm_dest == "User":
                    logger.info(f"DECOMPOSING Storage→User: {m['action']} {m['data_group']}")
                    # Décomposer en Read + Exit
                    decomposed = [
                        {
                            "action": "Read",
                            "data_group": m["data_group"],
                            "object_of_interest": m.get("object_of_interest", ""),
                            "source": "Storage",
                            "destination": "System",
                            "functional_process": m.get("functional_process", "")
                        },
                        {
                            "action": "Display",
                            "data_group": m["data_group"],
                            "object_of_interest": m.get("object_of_interest", ""),
                            "source": "System", 
                            "destination": m["destination"],
                            "functional_process": m.get("functional_process", "")
                        }
                    ]
                    valid_movements.extend(decomposed)
                    continue
                
                if norm_source == "User" and norm_dest == "Storage":
                    logger.info(f"DECOMPOSING User→Storage: {m['action']} {m['data_group']}")
                    # Décomposer en Entry + Write
                    decomposed = [
                        {
                            "action": "Input",
                            "data_group": m["data_group"],
                            "object_of_interest": m.get("object_of_interest", ""),
                            "source": m["source"],
                            "destination": "System",
                            "functional_process": m.get("process_name", "")
                        },
                        {
                            "action": "Store",
                            "data_group": m["data_group"],
                            "object_of_interest": m.get("object_of_interest", ""),
                            "source": "System",
                            "destination": "Storage",
                            "functional_process": m.get("process_name", "")
                        }
                    ]
                    valid_movements.extend(decomposed)
                    continue
                
                # Mouvement valide, garder tel quel
                logger.debug(f"✅ VALID movement: {m['action']} {m['data_group']} {norm_source}→{norm_dest}")
                valid_movements.append(m)
            
            logger.info(f"Filtered movements: {len(valid_movements)} valid, {excluded_count} excluded")
            
            # Étape 2: Convertir en tuples pour traitement RU2/RU3
            movement_tuples = []
            for m in valid_movements:
                tuple_data = (m["action"], m["data_group"],m.get("object_of_interest", ""), m["source"], m["destination"], m.get("functional_process", ""))
                movement_tuples.append(tuple_data)
            
            # Étape 3: Appliquer RU2 (élimination des doublons)
            initial_count = len(movement_tuples)
            movement_tuples = self.apply_rule_2(movement_tuples)
            logger.info(f"RU2: Removed {initial_count - len(movement_tuples)} duplicates")
            
            # Étape 4: Appliquer RU3 (classification) et convertir en objets DataMovement
            movements = []
            for tuple_data in movement_tuples:
                action, dg,ooi, src, dst, fp_name = tuple_data
                movement_type = self.apply_rule_3((action, dg, src, dst))
                movement = DataMovement(
                    action_verb=action,
                    data_group=dg,
                    object_of_interest=ooi,
                    source=src,
                    destination=dst,
                    movement_type=movement_type,
                    functional_process=fp_name
                )
                movements.append(movement)
            
            # Étape 5: Appliquer RU4 (validation des Entry) - Version améliorée
            movements = self.apply_rule_4(movements, functional_processes)

            # Étape 6: Validation finale - Ne devrait plus avoir d'erreurs System→System
            validation_errors = self.validate_cosmic_rules(movements)
            if validation_errors:
                logger.warning(f"Validation errors found: {validation_errors}")
            else:
                logger.info("✅ All COSMIC validation rules passed!")
            
            # Étape 7: Vérification finale - Aucun mouvement System→System ne devrait subsister
            system_to_system_check = [m for m in movements 
                                    if self.normalize_entity(m.source) == "System" 
                                    and self.normalize_entity(m.destination) == "System"]
            
            if system_to_system_check:
                logger.error(f"⚠️  CRITICAL: {len(system_to_system_check)} System→System movements still present after filtering!")
                for m in system_to_system_check:
                    logger.error(f"   - {m.action_verb} {m.data_group} in {m.functional_process}")
            else:
                logger.info("✅ No System→System movements detected - COSMIC compliant!")
            
            logger.info(f"Enhanced movement processing completed: {len(movements)} final movements")
            return movements
            
        except Exception as e:
            logger.error(f"Error in enhanced movement processing: {e}")
            raise
    
    def get_cfp_summary(self, movements: List[DataMovement]) -> Dict:
        """Calcul CFP avec détails complets"""
        cfp_count = {
            DataMovementType.ENTRY.value: 0,
            DataMovementType.EXIT.value: 0,
            DataMovementType.READ.value: 0,
            DataMovementType.WRITE.value: 0
        }
        
        movements_detail = []
        movements_by_process = defaultdict(list)
        
        for movement in movements:
            cfp_count[movement.movement_type.value] += 1
            
            movement_detail = {
                "action": movement.action_verb,
                "data_group": movement.data_group,
                "object_of_interest": getattr(movement, "object_of_interest", ""),
                "source": movement.source,
                "destination": movement.destination,
                "type": movement.movement_type.value,
                "process": movement.functional_process
            }
            movements_detail.append(movement_detail)
            movements_by_process[movement.functional_process].append(movement_detail)
        
        total_cfp = sum(cfp_count.values())
        
        # Calcul CFP par processus
        cfp_by_process = {}
        for process_name, process_movements in movements_by_process.items():
            cfp_by_process[process_name] = len(process_movements)
        
        # Métriques de qualité RAG
        quality_metrics = {}
        if self.rag_system:
            quality_metrics = {
                "rag_enabled": True,
                "cosmic_compliance_score": self._calculate_compliance_score(movements)
            }
        
        return {
            "cfp_by_type": cfp_count,
            "total_cfp": total_cfp,
            "movement_count": len(movements),
            "movements_detail": movements_detail,
            "cfp_by_process": cfp_by_process,
            "quality_metrics": quality_metrics
        }
    
    def _calculate_compliance_score(self, movements: List[DataMovement]) -> float:
        """Calcule un score de conformité COSMIC - Version améliorée"""
        if not movements:
            return 0.0
        
        score = 1.0
        penalties = []
        
        # Grouper par processus
        movements_by_process = defaultdict(list)
        for movement in movements:
            movements_by_process[movement.functional_process].append(movement)
        
        for process_name, process_movements in movements_by_process.items():
            # Vérifier la présence d'un Entry (obligatoire)
            has_entry = any(m.movement_type == DataMovementType.ENTRY for m in process_movements)
            if not has_entry:
                penalties.append(0.3)  # Pénalité importante pour manque d'Entry
            
            # Vérifier l'équilibre des mouvements (éviter les processus déséquilibrés)
            if len(process_movements) > 10:
                penalties.append(0.1)  # Pénalité légère pour processus très complexes
        
        # Vérifier les mouvements interdits (pénalité très importante)
        forbidden_count = 0
        for movement in movements:
            norm_source = self.normalize_entity(movement.source)
            norm_dest = self.normalize_entity(movement.destination)
            
            if ((norm_source == "Storage" and norm_dest == "User") or 
                (norm_source == "User" and norm_dest == "Storage") or
                (norm_source == "System" and norm_dest == "System")):
                forbidden_count += 1
        
        if forbidden_count > 0:
            penalties.append(0.2 * forbidden_count)  # 0.2 par mouvement interdit
        
        # Vérifier la cohérence des data groups
        total_movements = len(movements)
        unique_data_groups = len(set(m.data_group for m in movements))
        
        if unique_data_groups > total_movements * 0.8:  # Trop de groupes différents
            penalties.append(0.1)
        
        # Appliquer les pénalités
        final_score = score - sum(penalties)
        return max(0.0, min(1.0, final_score))

    def generate_measurement_report(self, movements: List[DataMovement], functional_processes: List[Dict]) -> Dict:
        """Génère un rapport de mesure complet"""
        cfp_summary = self.get_cfp_summary(movements)
        
        report = {
            "total_cfp": cfp_summary["total_cfp"],
            "functional_processes_count": len(functional_processes),
            "data_movements_count": len(movements),
            "cfp_breakdown": cfp_summary["cfp_by_type"],
            "process_details": [],
            "quality_assessment": cfp_summary.get("quality_metrics", {}),
            "rag_insights": []
        }
        
        # Détails par processus
        for fp in functional_processes:
            fp_movements = [m for m in movements if m.functional_process == fp.get("name", "")]
            fp_cfp = len(fp_movements)
            
            report["process_details"].append({
                "process_name": fp.get("name", ""),
                "triggering_event": fp.get("TriggeringEvents", ""),
                "functional_user": fp.get("FunctionalUser", ""),
                "cfp": fp_cfp,
                "movements": [
                    {
                        "action": m.action_verb,
                        "type": m.movement_type.value,
                        "data_group": m.data_group,
                        "object_of_interest": getattr(m, "object_of_interest", "")
                    } for m in fp_movements
                ]
            })
        
        # Ajouter des insights RAG si disponible
        if self.rag_system:
            try:
                insights = self._generate_rag_insights(movements, functional_processes)
                report["rag_insights"] = insights
            except Exception as e:
                logger.debug(f"Failed to generate RAG insights: {e}")
        
        return report
    
    def _generate_rag_insights(self, movements: List[DataMovement], functional_processes: List[Dict]) -> List[str]:
        """Génère des insights basés sur la connaissance COSMIC via RAG - Version corrigée"""
        insights = []
        
        try:
            total_cfp = len(movements)
            process_count = len(functional_processes)
            
            if process_count > 0:
                avg_cfp_per_process = total_cfp / process_count
                
                if avg_cfp_per_process < 2:
                    insights.append("Low CFP per process detected - consider if functional processes are properly decomposed")
                elif avg_cfp_per_process > 8:
                    insights.append("High CFP per process - consider if granularity level is consistent across processes")
            
            # Analyser la distribution des types de mouvement
            if movements:
                movement_types = [m.movement_type for m in movements]
                entry_ratio = sum(1 for m in movement_types if m == DataMovementType.ENTRY) / len(movements)
                
                if entry_ratio > 0.5:
                    insights.append("High proportion of Entry movements - verify triggering events are properly identified")
                elif entry_ratio < 0.1:
                    insights.append("Low proportion of Entry movements - ensure all functional processes have triggering entries")
            
            # Détecter les processus automatiques
            automatic_processes = [fp for fp in functional_processes if fp.get("FunctionalUser", "").lower() == "system"]
            if automatic_processes:
                insights.append(f"Detected {len(automatic_processes)} automatic processes - ensure proper Clock functional user for time-based triggers")
            
            # Analyser les patterns de données
            data_groups = set(m.data_group for m in movements)
            if len(data_groups) > process_count * 2:
                insights.append("High number of data groups relative to processes - verify data group identification consistency")
            
            # Vérifier l'équilibre Entry/Exit
            entry_count = sum(1 for m in movements if m.movement_type == DataMovementType.ENTRY)
            exit_count = sum(1 for m in movements if m.movement_type == DataMovementType.EXIT)
            
            if entry_count > 0 and exit_count > 0:
                io_ratio = min(entry_count, exit_count) / max(entry_count, exit_count)
                if io_ratio < 0.5:
                    insights.append("Imbalanced Entry/Exit ratio - consider if all user interactions provide appropriate feedback")
            
            # Rechercher des recommandations spécifiques dans la base de connaissances RAG
            if self.rag_system:
                try:
                    measurement_chunks = self.rag_system.retrieve_relevant_chunks(
                        "best practices measurement quality functional size",
                        top_k=2,
                        domain_filter="quality_assurance"
                    )
                    
                    for chunk in measurement_chunks:
                        content = chunk.get('content', '')
                        if 'best practice' in content.lower():
                            insights.append(f"COSMIC Best Practice: Consider reviewing {chunk.get('section', 'measurement guidelines')}")
                
                except Exception as e:
                    logger.debug(f"RAG insights retrieval failed: {e}")
            
        except Exception as e:
            logger.debug(f"Error generating RAG insights: {e}")
        
        return insights