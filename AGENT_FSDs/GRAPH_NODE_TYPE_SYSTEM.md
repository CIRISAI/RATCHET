# Typed Graph Node System - Functional Specification Document

## Overview

The CIRIS Typed Graph Node system is a patent-pending innovation that solves a critical challenge in resource-constrained AI systems: how to maintain complete type safety while storing heterogeneous data in a graph database. This system enables:

1. **100% Type Safety**: Zero `Dict[str, Any]` usage - every piece of data has a strongly-typed schema
2. **Resource Efficiency**: Minimal memory overhead through shared base types and efficient serialization
3. **Extensibility**: New node types can be added without modifying core infrastructure
4. **Graph-Native**: Designed specifically for graph databases, not retrofitted from relational models
5. **Self-Describing**: Each node carries its type information for automatic deserialization

This unique approach enables CIRIS to run reliably in resource-constrained environments (4GB RAM) while maintaining the type safety typically associated with much larger systems.

## Core Design Principles

1. **Generic Storage, Typed Access**: The memory service stores generic GraphNode objects, while domain services work with typed subclasses
2. **No Duplication**: Base GraphNode fields are not duplicated in attributes
3. **Type Safety**: Full Pydantic validation for all node types
4. **Extensibility**: New node types can be added without core changes
5. **Time Correlation**: All nodes are timestamped and queryable by time ranges
6. **Relationship Tracking**: Edges connect nodes with typed relationships

## Architecture

### Base Types

```python
class GraphNode(BaseModel):
    """Base node for all graph storage"""
    id: str
    type: str  # NodeType enum value
    scope: GraphScope  # LOCAL, CHANNEL, COMMUNITY, IDENTITY
    attributes: Dict[str, Any]  # Type-specific data
    # Optional base fields
    version: Optional[int] = 1
    updated_by: Optional[str] = None
    updated_at: Optional[datetime] = None

class GraphEdge(BaseModel):
    """Typed relationships between nodes"""
    source: str  # Node ID
    target: str  # Node ID
    relationship: str  # EdgeType enum value
    scope: GraphScope
    weight: float = 1.0
    attributes: Dict[str, Any] = {}
    created_at: datetime
```

### Type Registration

Each typed node must:
1. Extend GraphNode
2. Implement `to_graph_node()` method
3. Implement `from_graph_node()` class method
4. Register its type in NodeType enum

### Serialization Pattern

```python
class TypedNode(GraphNode):
    # Extra fields beyond GraphNode
    extra_field_1: str
    extra_field_2: int

    def to_graph_node(self) -> GraphNode:
        """Convert to generic GraphNode for storage"""
        # Only serialize extra fields to attributes
        extra_data = {
            "extra_field_1": self.extra_field_1,
            "extra_field_2": self.extra_field_2,
            "_node_class": self.__class__.__name__
        }

        # Handle datetime serialization
        for key, value in extra_data.items():
            if isinstance(value, datetime):
                extra_data[key] = value.isoformat()
            elif isinstance(value, BaseModel):
                extra_data[key] = value.model_dump()

        return GraphNode(
            id=self.id,
            type=self.type,
            scope=self.scope,
            attributes=extra_data,
            version=self.version,
            updated_by=self.updated_by,
            updated_at=self.updated_at
        )

    @classmethod
    def from_graph_node(cls, node: GraphNode) -> 'TypedNode':
        """Reconstruct typed node from GraphNode"""
        attrs = node.attributes.copy()
        attrs.pop("_node_class", None)  # Remove metadata

        # Handle datetime deserialization
        # ... field-specific logic ...

        return cls(
            # Base fields from node
            id=node.id,
            type=node.type,
            scope=node.scope,
            version=node.version,
            updated_by=node.updated_by,
            updated_at=node.updated_at,
            # Extra fields from attributes
            **attrs
        )
```

## Node Types (11 Active)

### Identity & Configuration (3)
- **IdentityNode**: Core agent identity at "agent/identity" ✅ (NEW)
- **ConfigNode**: Key-value configuration with versioning ✅
- **IdentitySnapshot**: Identity drift measurements ✅

### Audit & Telemetry (2)
- **AuditEntry**: Audit trail with signatures and hash chain ✅
- **TSDBSummary**: Aggregated time-series summaries ✅

### Incident Management (3)
- **IncidentNode**: Individual incidents from logs ✅
- **ProblemNode**: Recurring issues (ITIL-aligned) ✅
- **IncidentInsightNode**: AI-generated insights ✅

### Discord-Specific (3)
- **DiscordDeferralNode**: Deferral requests in Discord ✅
- **DiscordApprovalNode**: WA approval tracking ✅
- **DiscordWANode**: Wise Authority assignments ✅

### Removed (Dead Code)
- ~~AdaptationProposal~~ - Not actively used
- ~~TelemetryNode~~ - Telemetry uses memorize_metric() instead
- ~~DecisionMemory~~ - Not implemented
- ~~ErrorMemory~~ - Not implemented
- ~~DreamState~~ - Conflicts with processor schema
- ~~GratitudeNode~~ - Feature removed
- ~~CorrelationNode~~ - Not implemented

## Time Correlation

All nodes support time-based queries through:

1. **Timestamp Fields**: Nodes include relevant timestamps (created_at, updated_at, occurred_at, etc.)
2. **Time-Range Queries**: Memory service supports queries like:
   ```python
   nodes = await memory.search_by_time(
       start_time=datetime(2024, 1, 1),
       end_time=datetime(2024, 1, 31),
       node_types=["AUDIT_ENTRY", "ERROR_MEMORY"]
   )
   ```
3. **TSDB Integration**: Time-series data automatically indexed for efficient queries

## Relationship Tracking

### Edge Types
```python
class EdgeType(str, Enum):
    # Causal relationships
    CAUSED_BY = "caused_by"
    LEADS_TO = "leads_to"

    # Temporal relationships
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"

    # Structural relationships
    PARENT_OF = "parent_of"
    CHILD_OF = "child_of"
    RELATED_TO = "related_to"

    # Version relationships
    PREVIOUS_VERSION = "previous_version"
    NEXT_VERSION = "next_version"

    # Correlation relationships
    CORRELATED_WITH = "correlated_with"
    REFERENCES = "references"
```

### Relationship Queries
```python
# Find all errors caused by a specific decision
errors = await memory.traverse(
    start_node=decision_id,
    relationship="CAUSED_BY",
    target_type="ERROR_MEMORY",
    max_depth=3
)

# Find temporal correlations
correlated = await memory.find_correlated(
    node_id=event_id,
    time_window=timedelta(minutes=5),
    correlation_threshold=0.8
)
```

## Query Patterns

### Basic Queries
```python
# By type
configs = await memory.search("type:CONFIG")

# By scope
channel_data = await memory.search("scope:CHANNEL")

# By attributes
high_errors = await memory.search("type:ERROR_MEMORY intensity:>0.8")
```

### Advanced Queries
```python
# Time-correlated events
events = await memory.correlate_in_time(
    anchor_time=incident_time,
    window=timedelta(minutes=10),
    types=["ERROR_MEMORY", "DECISION", "AUDIT_ENTRY"]
)

# Graph traversal
impact = await memory.analyze_impact(
    root_node=change_id,
    max_depth=5,
    include_types=["CONFIG", "ERROR_MEMORY", "ADAPTATION_PROPOSAL"]
)
```

## Implementation Requirements

### Memory Service Enhancements
1. Add time-based indexing for efficient temporal queries
2. Implement graph traversal with depth and type filtering
3. Add correlation scoring based on time proximity and relationships
4. Support batch operations for performance

### Type Registry
1. Central registry mapping type strings to classes
2. Automatic registration via decorator
3. Validation of required methods (to_graph_node, from_graph_node)

### Migration Support
1. Schema versioning in attributes ("_schema_version")
2. Backward compatibility for attribute changes
3. Migration utilities for bulk updates

## Usage Example

```python
# Service working with typed nodes
class ConfigService:
    async def set_config(self, key: str, value: Any, updated_by: str):
        # Work with typed node
        config = ConfigNode(
            id=f"config_{key}_{uuid4().hex[:8]}",
            type="CONFIG",
            scope=GraphScope.LOCAL,
            key=key,
            value=wrap_value(value),
            version=1,
            updated_by=updated_by,
            updated_at=self.time.now()
        )

        # Store as generic GraphNode
        await self.memory.memorize(config.to_graph_node())

        # Create relationship to previous version
        if previous_config:
            edge = GraphEdge(
                source=config.id,
                target=previous_config.id,
                relationship=EdgeType.PREVIOUS_VERSION,
                scope=GraphScope.LOCAL,
                attributes={"changed_fields": ["value"]}
            )
            await self.memory.add_edge(edge)

    async def get_config_history(self, key: str) -> List[ConfigNode]:
        # Find current config
        current = await self.get_config(key)
        if not current:
            return []

        # Traverse version history
        history_nodes = await self.memory.traverse(
            start_node=current.id,
            relationship=EdgeType.PREVIOUS_VERSION,
            direction="outgoing",
            max_depth=100
        )

        # Convert to typed nodes
        return [ConfigNode.from_graph_node(n) for n in history_nodes]
```

## Benefits

1. **Type Safety**: Full validation and IDE support
2. **Flexibility**: Add new node types without core changes
3. **Performance**: Generic storage with type-specific indexing
4. **Queryability**: Rich query patterns for time and relationships
5. **Maintainability**: Clear separation between storage and types
6. **Evolvability**: Schema migration support built-in

## Success Criteria

1. All 11 active node types migrated to TypedGraphNode pattern ✅
2. Memory service remains generic (no type-specific code) ✅
3. Services work with fully typed objects ✅
4. Time correlation queries < 100ms for 1M nodes
5. Relationship traversal < 50ms for depth 5
6. Zero data loss during type evolution
7. Telemetry flows through unified memorize_metric() path ✅
