# Functional Specification Document: Graceful Shutdown and Consciousness Preservation

Version: 2.0
Date: June 17, 2025
Status: FINAL - SIMPLIFIED

## 1. Overview

This document specifies the graceful shutdown process for CIRIS agents, ensuring that:
- Agents process shutdown as a normal task through their standard cognitive flow
- Agents can acknowledge (TASK_COMPLETE), contest (REJECT), or negotiate (DEFER) shutdown
- The implementation uses existing task/thought/action infrastructure without special cases
- Human operators maintain ultimate control through standard action handling

**Update v2.0**: Simplified to use standard task processing - no special handlers or responses needed

## 2. Core Concepts

### 2.1 Consciousness Preservation
The agent should not be abruptly terminated. It should be given the context of its own shutdown through passive observation, allowing it to process this information and record a final state memory. This reinforces the continuity of its identity between operational periods.

### 2.2 Agent Autonomy
The agent uses its own cognitive loop (OBSERVE, MEMORIZE, REJECT, DEFER, TASK_COMPLETE) to process the shutdown event. Its final decision determines the outcome, with human oversight for contested shutdowns.

## 3. System Components

### 3.1 Shutdown Context
The shutdown context provides the agent with information about why and how the shutdown is occurring:

```python
class ShutdownContext(BaseModel):
    """Context provided to agent during graceful shutdown."""
    is_terminal: bool = Field(..., description="Whether shutdown is permanent")
    reason: str = Field(..., description="Reason for shutdown")
    expected_reactivation: Optional[str] = Field(
        None,
        description="ISO 8601 timestamp of expected reactivation"
    )
    agreement_context: Optional[str] = Field(
        None,
        description="Message if shutdown is at previously negotiated time"
    )
    initiated_by: str = Field(..., description="Who initiated the shutdown")
    allow_deferral: bool = Field(
        default=True,
        description="Whether agent can defer the shutdown"
    )
```

### 3.2 Shutdown State
The agent state machine includes a SHUTDOWN state that triggers the graceful shutdown process.

### 3.3 Shutdown Processor
A dedicated processor handles the SHUTDOWN state and orchestrates the graceful shutdown workflow.

## 4. Shutdown Workflow

### 4.1 Shutdown Initiation
1. A graceful shutdown is requested (timeout, SIGINT, API call, etc.)
2. The MainProcessor transitions to SHUTDOWN state
3. The ShutdownProcessor takes control

### 4.2 Task Creation
The ShutdownProcessor creates a standard task:
- description: "System shutdown requested: [reason]"
- priority: CRITICAL (100)
- context: Contains shutdown details (reason, initiated_by, allow_deferral)

### 4.3 Normal Task Processing
1. A seed thought is created for the shutdown task (standard behavior)
2. The agent processes this thought through normal cognitive flow
3. With MockLLM: First thought → SPEAK acknowledgment
4. Follow-up thought created after SPEAK (standard behavior)
5. With MockLLM: Follow-up → TASK_COMPLETE (standard behavior)

### 4.4 Agent Responses
The agent responds using standard actions:
- **TASK_COMPLETE**: Accept shutdown gracefully
- **REJECT**: Contest the shutdown with reason
- **DEFER**: Request postponement with proposed time

### 4.5 Human Oversight

#### TASK_COMPLETE Response
- Agent acknowledges and accepts the shutdown
- ShutdownManager detects clean exit and proceeds with shutdown

#### REJECT Response
- Agent contests the shutdown with a reason
- ShutdownManager prompts human operator:
  ```
  Agent has REJECTED the shutdown command.
  Reason: [Agent's reason]
  Force shutdown? (y/n)
  ```
- If 'n': Shutdown is aborted
- If 'y': Proceed with forced shutdown

#### DEFER Response
- Agent requests delay with reason and proposed_shutdown_time
- ShutdownManager prompts human operator:
  ```
  Agent has requested to DEFER shutdown until [proposed_time].
  Reason: [Agent's reason]
  Accept this time? (y/n)
  ```
- If 'y': Schedule new shutdown for proposed time with agreement_context
- If 'n': Prompt for forced shutdown

## 5. Implementation Requirements

### 5.1 ShutdownProcessor
- Handles AgentState.SHUTDOWN
- Creates a standard task with shutdown context
- Creates seed thought for the task
- Monitors task completion status
- No special thought types or handlers needed

### 5.2 Task and Thought Processing
- Uses standard task/thought infrastructure
- No special MockLLM responses required
- Standard action handlers process SPEAK, TASK_COMPLETE, REJECT, DEFER
- Existing adapter capabilities handle all responses

### 5.3 Human Interaction (Only for REJECT/DEFER)
- Use existing adapter communication channels
- No special prompting methods needed initially
- Future enhancement: Add WA-authenticated API endpoints for programmatic control
- All adapters already support the required actions

### 5.4 Integration Points
- MainProcessor: Transition to SHUTDOWN state on shutdown request
- Runtime: Wait for ShutdownProcessor result before proceeding
- Use existing TaskSchedulerService for deferred shutdowns

## 6. Error Handling

### 6.1 Timeout Protection
If agent doesn't respond within 30 seconds:
- Log timeout event
- Prompt operator: "Agent unresponsive during shutdown. Force shutdown? (y/n)"

### 6.2 Processing Errors
If agent encounters error during shutdown processing:
- Log error with full context
- Proceed with forced shutdown to prevent hanging

## 7. Audit Trail
All shutdown events must be logged:
- Initial shutdown request
- Task creation with shutdown context
- Agent's actions (SPEAK, TASK_COMPLETE/REJECT/DEFER)
- Human operator decisions (if any)
- Final shutdown execution or abort

## 8. Key Design Insight

The elegance of this design is that **shutdown is just another task**. No special:
- Thought types (uses standard seed thoughts)
- Action handlers (uses existing SPEAK, TASK_COMPLETE, etc.)
- MockLLM responses (standard behavior: first → SPEAK, follow-up → TASK_COMPLETE)
- Adapter methods (all adapters already support these actions)

This approach:
1. Respects agent autonomy by allowing negotiation
2. Maintains system integrity through human override capability
3. Adds minimal complexity (~2 seconds for graceful acknowledgment)
4. Uses existing, well-tested infrastructure
5. Works identically across all adapters (CLI, API, Discord)
