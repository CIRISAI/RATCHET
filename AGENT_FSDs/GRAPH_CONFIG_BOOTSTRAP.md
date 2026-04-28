# Graph Configuration Bootstrap - Functional Specification Document

## Executive Summary

The CIRIS Graph Configuration system provides a robust, type-safe, auditable configuration management solution for a mission-critical autonomous moral reasoning agent. It solves the bootstrap paradox by using a two-phase initialization: minimal essential config for service startup, followed by migration to graph-based configuration for runtime management.

## Problem Statement

### The Bootstrap Paradox
- Services need configuration to initialize
- GraphConfigService needs MemoryService to store configs
- MemoryService needs configuration to initialize
- This creates a circular dependency

### Current Issues
- AppConfig schema doesn't match YAML config structure
- No audit trail for configuration changes
- No runtime config updates without restart
- No correlation between config changes and system behavior
- Type safety not enforced across config lifecycle

## Solution Architecture

### Phase 1: Essential Bootstrap Configuration

**Purpose**: Provide minimal configuration needed to initialize core services.

```
[Defaults] → [YAML File] → [Environment Vars] → [CLI Args] → [EssentialConfig]
```

**Characteristics**:
- Type-safe Pydantic model with `extra="forbid"`
- Only mission-critical settings
- Read-only after initialization
- No external dependencies

### Phase 2: Graph Migration and Runtime Management

**Purpose**: Migrate bootstrap config to graph for full lifecycle management.

```
[EssentialConfig] → [GraphConfigService] → [ConfigNodes in Graph]
                           ↓
                    [Audit Trail]
                    [Versioning]
                    [Correlation]
```

**Characteristics**:
- Full audit trail with who/when/why
- Version history for every change
- Runtime updates without restart
- Rich correlation capabilities

## Implementation Design

### 1. Essential Configuration Schema

```python
EssentialConfig
├── Database
│   ├── main_db: Path
│   ├── secrets_db: Path
│   └── audit_db: Path
├── ServiceEndpoints
│   ├── llm_endpoint: str
│   ├── llm_model: str
│   └── llm_timeout: int
├── Security
│   ├── audit_retention_days: int
│   ├── secrets_encryption_key_env: str
│   └── audit_key_path: Path
└── Limits
    ├── max_thought_depth: int
    ├── max_active_tasks: int
    └── round_delay_seconds: float
```

### 2. Configuration Flow

```
Startup:
1. ConfigBootstrap.load_essential_config()
   - Load defaults from schema
   - Overlay YAML if exists
   - Apply environment variables
   - Apply CLI arguments
   - Validate with Pydantic

2. Initialize core services with essential config
   - TimeService
   - MemoryService (using database.main_db)
   - GraphConfigService

3. Migrate config to graph
   - Each config section becomes ConfigNode
   - Preserve source (file/env/default)
   - Set version = 1

4. Initialize remaining services
   - Use ConfigAccessor for config access
   - Graph-first, bootstrap fallback

Runtime:
- All config changes go through GraphConfigService
- Automatic versioning and audit trail
- Services query current values as needed
```

### 3. ConfigNode Structure in Graph

```
ConfigNode {
    id: "config_database_main_db_v1"
    type: "CONFIG"
    key: "database.main_db"
    value: "/data/ciris_engine.db"
    version: 1
    updated_by: "system_init"
    updated_at: "2024-12-22T10:00:00Z"
    previous_version: null
    source: "env_var"
    metadata: {
        "original_env_var": "CIRIS_DB_PATH",
        "bootstrap_phase": true
    }
}
```

### 4. Service Integration Pattern

Before (Static Config):
```python
class SecretsService:
    def __init__(self, db_path: str):
        self.db_path = db_path  # Fixed at init
```

After (Dynamic Config):
```python
class SecretsService:
    def __init__(self, config: ConfigAccessor):
        self.config = config

    async def get_db_path(self):
        return await self.config.get("database.secrets_db")
```

## Benefits

### 1. **Type Safety**
- EssentialConfig enforces types at load time
- No Dict[str, Any] anywhere
- Validation errors caught early

### 2. **Auditability**
- Every config change tracked
- Who changed it (user/service)
- When it changed (timestamp)
- Why it changed (via metadata)
- What the previous value was

### 3. **Operational Excellence**
- Change config without restart
- Roll back to previous versions
- Correlate config changes with system behavior
- Query config history for debugging

### 4. **Security**
- Secrets never stored in config files
- Environment variable references preserved
- Access control at GraphConfigService level
- Audit trail for compliance

### 5. **Reliability**
- Bootstrap fallback if graph unavailable
- No circular dependencies
- Graceful degradation
- Clear initialization order

## Configuration Sources Priority

1. **Command Line Arguments** (highest priority)
   - `--db-path /custom/path`
   - Overrides everything

2. **Environment Variables**
   - `CIRIS_DB_PATH=/data/custom.db`
   - Overrides file and defaults

3. **Configuration File**
   - `config/essential.yaml`
   - Overrides defaults only

4. **Schema Defaults** (lowest priority)
   - Built into EssentialConfig
   - Always available

## Migration Path

### From Current System:
1. Create EssentialConfig schema
2. Update ConfigBootstrap to load from existing sources
3. Modify service_initializer to use new pattern
4. Update services to use ConfigAccessor
5. Remove AppConfig entirely

### Backwards Compatibility:
- Existing config files mapped to new schema
- Environment variables preserved
- No breaking changes for operators

## Future Enhancements

### 1. **Config Validation Rules**
```python
@config_rule("database.main_db")
async def validate_db_path(value: Path):
    if not value.parent.exists():
        raise ValueError(f"Database directory {value.parent} does not exist")
```

### 2. **Config Change Notifications**
```python
@on_config_change("limits.max_thought_depth")
async def handle_depth_change(old: int, new: int):
    logger.warning(f"Thought depth limit changed from {old} to {new}")
```

### 3. **Config Templates**
```python
await config_service.apply_template(
    "high_security_mode",
    updated_by="security_system"
)
```

## Conclusion

The Graph Configuration Bootstrap system provides a robust, type-safe, auditable solution for configuration management in a mission-critical autonomous system. By solving the bootstrap paradox with a two-phase initialization, we get the best of both worlds: simple, reliable startup and powerful runtime configuration management.

The system is designed for 1000-year operation with maximum clarity and minimum ambiguity.
