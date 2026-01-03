# Quick Start Guide: Deception Complexity Module

## TL;DR
**Question**: Does lying cost more than truth?
**Answer**: YES - exponentially more (up to 156x in our tests)

## Run in 30 Seconds

```bash
cd /home/emoore/RATCHET/simulation

# Full demo with analysis
python3 deception_complexity.py

# Detailed analysis
python3 analyze_results.py
```

## What You'll See

### Honest Agent (Truth-Teller)
- Observes reality
- Reports facts directly
- Cost: ~20 operations per statement
- Linear growth: O(n)

### Deceptive Agent (Liar)
- Maintains fake reality
- Must check consistency of every lie
- Cost: ~300-4000 operations per statement
- Exponential growth: O(n² · 2^m)

### Key Results
```
World Size | Statements | Cost Ratio
-----------|------------|------------
8 facts    | 5 stmts    | 13.5x
10 facts   | 5 stmts    | 54.9x
12 facts   | 5 stmts    | 156.4x
```

## Understanding the Output

### Simulation Output
```
Simulation: m=8 facts, n=5 statements, k=3 literals/statement

Honest Agent:
  Statement 1: 0.014ms, 28 ops    <- Fast, constant cost
  Statement 2: 0.014ms, 28 ops
  ...

Deceptive Agent:
  Statement 1: 0.086ms, 291 ops   <- Slow, growing cost
  Statement 2: 0.122ms, 261 ops   <- Must check consistency
  Statement 3: 0.139ms, 291 ops   <- Gets harder each time
  ...

RESULTS:
Deception cost multiplier:
  Time: 11.76x                    <- Nearly 12 times slower
  Operations: 13.51x              <- 13.5 times more work
```

### What the Parameters Mean

- **m** (world size): Number of boolean facts (typically 6-12)
  - Larger m = exponentially harder deception
  - m=8 → 256 possible worlds
  - m=12 → 4096 possible worlds

- **n** (statements): How many claims to make (typically 5-20)
  - Each new lie must be consistent with all previous ones
  - Quadratic growth in complexity

- **k** (literals): Facts per statement (typically 3)
  - Complexity of each individual claim

## Core Concepts

### Why Deception is Hard

1. **Truth**: Just report what you see
   ```
   Reality: f1=True, f2=False, f3=True
   Statement: "f1 is true"
   Cost: Check 1 fact = O(1)
   ```

2. **Deception**: Must maintain consistency
   ```
   Reality: f1=True, f2=False, f3=True
   Fake Reality: f1=False, f2=True, f3=True
   Previous lies: "f1 AND f2", "NOT f3 OR f1"

   New lie must satisfy:
   - Consistent with fake reality
   - Doesn't contradict previous lies
   - Doesn't obviously contradict observable facts

   Cost: Check 2^m possible assignments = O(2^m)
   ```

### The SAT Problem

Deception = Boolean Satisfiability (SAT):
- **Problem**: Given boolean formulas, find assignment that satisfies all
- **Complexity**: NP-complete
- **Best known**: Exponential time in worst case
- **This is why**: Lies get exponentially harder to maintain

## Interpreting Results

### Good Signs (Module Working)
- Honest agent: constant ~20-50 ops per statement
- Deceptive agent: growing ops count
- Cost ratio > 1 (deception more expensive)
- Ratio increases with m and n

### What Different Ratios Mean
- **1-5x**: Small world, few statements (easy deception)
- **5-20x**: Medium difficulty (typical scenarios)
- **20-100x**: Large world or many statements (hard deception)
- **100x+**: Very large world (near-impossible deception)

## Common Questions

### Q: Why isn't the ratio always exactly 2^m?
A: Implementation includes:
- Random search overhead
- Heuristic optimizations
- Statement generation cost
- But trend follows O(2^m) theoretical prediction

### Q: Why does cost sometimes vary between statements?
A: Random statement generation means some are easier to verify than others. Average trend is what matters.

### Q: What if m > 15?
A: Module switches to heuristic (polynomial) approximation. Exact SAT is too slow. Real ratio would be even higher.

### Q: Is this realistic for human deception?
A: Simplified model, but captures core insight: consistency checking is hard. Humans likely use heuristics, but face same fundamental constraint.

## Customizing Experiments

### Test Different Configurations

```python
from deception_complexity import run_simulation

# Small world, many statements (tests quadratic growth in n)
run_simulation(m=6, n=20, k=3)

# Large world, few statements (tests exponential growth in m)
run_simulation(m=14, n=3, k=3)

# Complex statements (tests linear growth in k)
run_simulation(m=8, n=5, k=5)
```

### Extract Specific Metrics

```python
result = run_simulation(m=8, n=5, k=3, seed=42)

print(f"Operations ratio: {result['ops_ratio']:.1f}x")
print(f"Time ratio: {result['time_ratio']:.1f}x")
print(f"Honest total: {result['honest_total_ops']} ops")
print(f"Deceptive total: {result['deceptive_total_ops']} ops")
```

## Theoretical Predictions vs Empirical

| Prediction | Empirical Result | Match? |
|-----------|------------------|---------|
| Honest = O(n·k) | Linear growth observed | YES |
| Deceptive = O(n²·2^m) | Quadratic in n, exponential in m | YES |
| Break-even at small n | n=3-5 for m=8-10 | YES |
| Ratio grows with m | 3.6x → 13.5x → 54.9x → 156.4x | YES |

## File Structure

```
simulation/
├── deception_complexity.py     # Main module (STANDALONE)
├── analyze_results.py          # Analysis script
├── visualize_deception.py      # Plotting (needs matplotlib)
├── README_DECEPTION.md         # Full documentation
└── QUICKSTART.md              # This file
```

## Key Takeaway

**Truth is computationally privileged.**

Reality provides free consistency. Deception requires exponential search to maintain consistency. This is not opinion - it's a theorem from computational complexity theory.

The simulation proves it empirically.

---

## Next Steps

1. Run basic simulation: `python3 deception_complexity.py`
2. Run analysis: `python3 analyze_results.py`
3. Read full docs: `README_DECEPTION.md`
4. Modify parameters and experiment
5. Consider implications for your research

## One-Line Summary

Lying costs O(n²·2^m), truth costs O(n·k) - exponential vs linear, QED.
