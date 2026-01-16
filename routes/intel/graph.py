"""
Entity Graph

Build and analyze entity relationship graphs.
Find clusters, echo chambers, and connections.
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass

from .entities import Entity, EntityStore, Relationship


@dataclass
class GraphNode:
    """Node in the entity graph."""
    entity_id: str
    text: str
    category: str
    confidence: float
    source_models: Set[str]
    is_echo_chamber: bool


@dataclass
class GraphEdge:
    """Edge between entities."""
    source_id: str
    target_id: str
    relation_type: str
    weight: float
    source_response_id: str


class EntityGraph:
    """
    Graph of entity relationships.

    Supports:
    - Co-occurrence detection (entities mentioned together)
    - Echo chamber detection (same model clusters)
    - Provenance chains
    """

    def __init__(self, store: EntityStore):
        self.store = store
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self._build()

    def _build(self):
        """Build graph from entity store."""
        entities = self.store.all()

        # Build nodes
        for e in entities:
            self.nodes[e.id] = GraphNode(
                entity_id=e.id,
                text=e.text,
                category=e.category,
                confidence=e.confidence,
                source_models=e.source_models,
                is_echo_chamber=e.is_echo_chamber
            )

        # Build edges from co-occurrence
        self._build_cooccurrence_edges(entities)

        # Build edges from explicit relationships
        for e in entities:
            for rel in e.relationships:
                self.edges.append(GraphEdge(
                    source_id=e.id,
                    target_id=rel.target_entity_id,
                    relation_type=rel.relation_type,
                    weight=rel.confidence,
                    source_response_id=rel.source_response_id
                ))

    def _build_cooccurrence_edges(self, entities: List[Entity]):
        """Build edges for entities that appear in the same response."""
        # Group entities by response
        response_entities: Dict[str, List[str]] = defaultdict(list)

        for e in entities:
            if e.originated_from:
                response_entities[e.originated_from.response_id].append(e.id)
            for conf in e.confirmed_by:
                response_entities[conf.response_id].append(e.id)

        # Count co-occurrences for edge weights
        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        pair_responses: Dict[Tuple[str, str], str] = {}

        for resp_id, ent_ids in response_entities.items():
            for i, eid1 in enumerate(ent_ids):
                for eid2 in ent_ids[i+1:]:
                    pair = tuple(sorted([eid1, eid2]))
                    pair_counts[pair] += 1
                    pair_responses[pair] = resp_id

        # Create edges with weight based on co-occurrence count
        max_count = max(pair_counts.values()) if pair_counts else 1
        for pair, count in pair_counts.items():
            # Weight: 0.2 (single occurrence) to 1.0 (max occurrences)
            weight = 0.2 + 0.8 * (count / max_count)
            self.edges.append(GraphEdge(
                source_id=pair[0],
                target_id=pair[1],
                relation_type="co_occurred",
                weight=weight,
                source_response_id=pair_responses[pair]
            ))

    def get_echo_chambers(self) -> List[List[str]]:
        """
        Find clusters of entities that are echo chambers.

        Returns list of entity ID clusters.
        """
        chambers = []

        # Group by source model
        model_entities: Dict[str, List[str]] = defaultdict(list)
        for node in self.nodes.values():
            if node.is_echo_chamber:
                for model in node.source_models:
                    model_entities[model].append(node.entity_id)

        # Clusters with multiple entities from same model
        for model, ent_ids in model_entities.items():
            if len(ent_ids) > 1:
                chambers.append({
                    "model": model,
                    "entities": ent_ids,
                    "count": len(ent_ids)
                })

        return chambers

    def get_connections(self, entity_id: str) -> List[Tuple[str, str, float]]:
        """
        Get all connections for an entity.

        Returns list of (target_id, relation_type, weight).
        """
        connections = []

        for edge in self.edges:
            if edge.source_id == entity_id:
                connections.append((edge.target_id, edge.relation_type, edge.weight))
            elif edge.target_id == entity_id:
                connections.append((edge.source_id, edge.relation_type, edge.weight))

        return connections

    def get_clusters(self, min_size: int = 2) -> List[Set[str]]:
        """
        Find connected clusters of entities.

        Returns list of entity ID sets.
        """
        # Build adjacency list
        adj: Dict[str, Set[str]] = defaultdict(set)
        for edge in self.edges:
            adj[edge.source_id].add(edge.target_id)
            adj[edge.target_id].add(edge.source_id)

        # Find connected components via BFS
        visited = set()
        clusters = []

        for node_id in self.nodes:
            if node_id in visited:
                continue

            # BFS from this node
            cluster = set()
            queue = [node_id]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                cluster.add(current)

                for neighbor in adj[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            if len(cluster) >= min_size:
                clusters.append(cluster)

        return clusters

    def get_model_distribution(self, entity_id: str) -> Dict[str, int]:
        """Get distribution of which models mentioned an entity."""
        entity = self.store.get(entity_id)
        if not entity:
            return {}

        dist = defaultdict(int)

        if entity.originated_from:
            dist[entity.originated_from.model] += 1

        for conf in entity.confirmed_by:
            dist[conf.model] += 1

        return dict(dist)

    def to_dict(self) -> dict:
        """Export graph to dict for visualization."""
        return {
            "nodes": [
                {
                    "id": n.entity_id,
                    "text": n.text,
                    "category": n.category,
                    "confidence": n.confidence,
                    "models": list(n.source_models),
                    "is_echo_chamber": n.is_echo_chamber
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "type": e.relation_type,
                    "weight": e.weight
                }
                for e in self.edges
            ],
            "echo_chambers": self.get_echo_chambers(),
            "clusters": [list(c) for c in self.get_clusters()]
        }


def analyze_graph(store: EntityStore) -> dict:
    """
    Run full graph analysis.

    Returns analysis summary.
    """
    graph = EntityGraph(store)

    echo_chambers = graph.get_echo_chambers()
    clusters = graph.get_clusters()

    # Find most connected entities
    connection_counts = {}
    for node_id in graph.nodes:
        connection_counts[node_id] = len(graph.get_connections(node_id))

    top_connected = sorted(
        connection_counts.items(),
        key=lambda x: -x[1]
    )[:10]

    return {
        "total_nodes": len(graph.nodes),
        "total_edges": len(graph.edges),
        "echo_chambers": echo_chambers,
        "num_clusters": len(clusters),
        "largest_cluster_size": max(len(c) for c in clusters) if clusters else 0,
        "top_connected": [
            {
                "entity_id": eid,
                "text": graph.nodes[eid].text if eid in graph.nodes else "?",
                "connections": count
            }
            for eid, count in top_connected
        ],
        "graph": graph.to_dict()
    }
