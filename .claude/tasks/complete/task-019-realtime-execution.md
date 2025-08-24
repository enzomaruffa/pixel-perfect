# Task 019: Real-time Pipeline Execution

## Objective
Implement real-time pipeline execution that processes images as users modify operations and parameters, providing immediate visual feedback with intelligent caching and performance optimization.

## Requirements
1. Execute pipeline operations in real-time as parameters change
2. Implement intelligent caching to avoid redundant processing
3. Add partial pipeline execution (up to selected step)
4. Create execution progress indicators and performance monitoring
5. Handle execution errors gracefully with user feedback

## File Structure to Extend
```
src/ui/components/
├── pipeline_executor.py     # Main pipeline execution controller
├── execution_manager.py     # Execution state and caching management
├── progress_indicators.py   # Progress bars and status displays
└── error_handling.py        # Error display and recovery

src/ui/utils/
├── execution_cache.py       # Smart caching for pipeline steps
├── performance_monitor.py   # Execution time and memory monitoring
└── image_processor.py       # Optimized image processing utilities
```

## Core Features

### Real-time Pipeline Execution
- Execute full pipeline automatically when parameters change
- Debounced execution to avoid excessive processing
- Cancel previous executions when new changes occur
- Smart execution that only recomputes changed operations

### Step-by-Step Execution
- Execute pipeline up to any selected operation
- Show intermediate results at each step
- Allow users to preview effects of individual operations
- Step-through debugging mode for complex pipelines

### Intelligent Caching System
- Cache results of expensive operations
- Invalidate cache only for affected operations
- Share cache between similar parameter combinations
- Memory-aware cache management with LRU eviction

### Performance Optimization
- Background execution using Streamlit's async capabilities
- Progress indicators for long-running operations
- Memory usage monitoring and optimization
- Execution time profiling and bottleneck identification

## Technical Implementation

### Pipeline Execution Controller
```python
class PipelineExecutor:
    def __init__(self):
        self.cache = ExecutionCache()
        self.current_execution = None

    def execute_pipeline(self, image, operations, execute_up_to=None):
        """Execute pipeline with caching and progress tracking"""
        if self.current_execution:
            self.current_execution.cancel()

        self.current_execution = self._execute_async(
            image, operations, execute_up_to
        )
        return self.current_execution

    async def _execute_async(self, image, operations, execute_up_to):
        """Async pipeline execution with progress updates"""
        results = []
        current_image = image

        for i, operation_config in enumerate(operations):
            if execute_up_to and i >= execute_up_to:
                break

            # Check cache first
            cache_key = self._generate_cache_key(current_image, operation_config)
            cached_result = self.cache.get(cache_key)

            if cached_result:
                current_image, context = cached_result
                results.append({
                    'step': i,
                    'operation': operation_config['name'],
                    'image': current_image,
                    'context': context,
                    'cached': True
                })
            else:
                # Execute operation
                with st.spinner(f"Executing {operation_config['name']}..."):
                    current_image, context = await self._execute_operation(
                        current_image, operation_config
                    )

                # Cache result
                self.cache.set(cache_key, (current_image, context))

                results.append({
                    'step': i,
                    'operation': operation_config['name'],
                    'image': current_image,
                    'context': context,
                    'cached': False
                })

        return results
```

### Execution State Management
```python
def manage_execution_state():
    """Manage pipeline execution state in Streamlit session"""
    if 'execution_results' not in st.session_state:
        st.session_state.execution_results = []

    if 'execution_in_progress' not in st.session_state:
        st.session_state.execution_in_progress = False

    if 'execution_error' not in st.session_state:
        st.session_state.execution_error = None

def trigger_pipeline_execution():
    """Trigger pipeline re-execution when parameters change"""
    if should_execute_pipeline():
        execute_pipeline_async()

def should_execute_pipeline():
    """Determine if pipeline should be re-executed"""
    return (
        st.session_state.original_image is not None
        and len(st.session_state.pipeline_operations) > 0
        and not st.session_state.execution_in_progress
    )
```

### Smart Caching Implementation
```python
class ExecutionCache:
    def __init__(self, max_size=100, max_memory_mb=1000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.current_memory = 0

    def get(self, key):
        """Get cached result with LRU tracking"""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None

    def set(self, key, value):
        """Cache result with memory management"""
        if self._should_cache(key, value):
            self._ensure_space_available(value)
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.current_memory += self._estimate_memory(value)

    def _generate_cache_key(self, image, operation_config):
        """Generate unique cache key for operation and image state"""
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
        operation_hash = hashlib.md5(
            json.dumps(operation_config, sort_keys=True).encode()
        ).hexdigest()
        return f"{image_hash}_{operation_hash}"

    def _should_cache(self, key, value):
        """Determine if result should be cached based on size and cost"""
        memory_estimate = self._estimate_memory(value)
        return memory_estimate < self.max_memory_mb * 0.1  # Don't cache huge results
```

### Progress and Status Display
```python
def show_execution_progress():
    """Display execution progress and status"""
    if st.session_state.execution_in_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, result in enumerate(st.session_state.current_results):
            progress = (i + 1) / len(st.session_state.pipeline_operations)
            progress_bar.progress(progress)

            cache_status = "💾 Cached" if result['cached'] else "⚡ Processed"
            status_text.text(f"Step {i+1}: {result['operation']} {cache_status}")

    # Show execution statistics
    if st.session_state.execution_results:
        with st.expander("Execution Details"):
            show_execution_statistics()

def show_execution_statistics():
    """Display execution time and performance metrics"""
    results = st.session_state.execution_results

    col1, col2, col3 = st.columns(3)

    with col1:
        total_time = sum(r.get('execution_time', 0) for r in results)
        st.metric("Total Time", f"{total_time:.2f}s")

    with col2:
        cache_hits = sum(1 for r in results if r.get('cached', False))
        cache_rate = (cache_hits / len(results)) * 100 if results else 0
        st.metric("Cache Hit Rate", f"{cache_rate:.1f}%")

    with col3:
        memory_usage = estimate_memory_usage(results)
        st.metric("Memory Usage", f"{memory_usage:.1f} MB")
```

## Step-by-Step Execution Interface

### Pipeline Step Visualization
```python
def render_pipeline_steps():
    """Display pipeline steps with execution controls"""
    st.subheader("Pipeline Steps")

    for i, operation in enumerate(st.session_state.pipeline_operations):
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.write(f"**Step {i+1}**: {operation['name']}")

            with col2:
                if st.button(f"▶️ Run to here", key=f"run_to_{i}"):
                    execute_pipeline_up_to(i)

            with col3:
                if st.button(f"👁️ Preview", key=f"preview_{i}"):
                    show_step_preview(i)
```

### Step Preview System
- Show result of each individual operation
- Before/after comparison for each step
- Parameter sensitivity analysis
- Step execution time and memory usage

## Error Handling and Recovery

### Graceful Error Handling
```python
def handle_execution_error(error, operation_step):
    """Handle pipeline execution errors gracefully"""
    error_info = {
        'step': operation_step,
        'operation': st.session_state.pipeline_operations[operation_step]['name'],
        'error': str(error),
        'timestamp': time.time()
    }

    st.session_state.execution_error = error_info
    st.session_state.execution_in_progress = False

    # Show user-friendly error message
    st.error(f"❌ Error in step {operation_step + 1} ({error_info['operation']})")
    st.write(f"**Error details**: {error_info['error']}")

    # Suggest fixes
    suggest_error_fixes(error, operation_step)

def suggest_error_fixes(error, operation_step):
    """Provide helpful suggestions for common errors"""
    suggestions = []

    if "memory" in str(error).lower():
        suggestions.append("💡 Try reducing image size or simplifying parameters")

    if "validation" in str(error).lower():
        suggestions.append("💡 Check parameter values are within valid ranges")

    if suggestions:
        st.info("**Suggestions:**")
        for suggestion in suggestions:
            st.write(suggestion)
```

### Recovery Options
- Retry execution with modified parameters
- Skip problematic operations
- Reset pipeline to last working state
- Rollback to previous parameter values

## Performance Monitoring

### Real-time Performance Metrics
- Execution time per operation
- Memory usage tracking
- Cache hit/miss ratios
- Processing throughput (pixels/second)

### Performance Optimization Alerts
- Warn about slow operations
- Suggest parameter optimizations
- Recommend caching strategies
- Memory usage warnings

## Integration Points

### With Parameter Configuration (Task 018)
- Trigger execution when parameters change
- Debounce parameter changes to avoid excessive execution
- Validate parameters before execution

### With Image Display (Task 020)
- Display execution results in real-time
- Show intermediate step results
- Update comparison views automatically

### With Pipeline Management
- Handle pipeline modifications during execution
- Maintain execution state across UI changes
- Synchronize execution with pipeline operations

## Success Criteria
- [ ] Pipeline executes automatically when parameters change
- [ ] Step-by-step execution shows intermediate results
- [ ] Intelligent caching improves performance significantly
- [ ] Execution progress is clearly communicated to users
- [ ] Errors are handled gracefully with helpful messages
- [ ] Performance metrics are displayed and monitored
- [ ] Memory usage is controlled and optimized
- [ ] Background execution doesn't block UI interactions

## Test Cases to Implement
- **Real-time Execution Test**: Parameters changes trigger appropriate execution
- **Caching Effectiveness Test**: Cache improves performance measurably
- **Step Execution Test**: Partial pipeline execution works correctly
- **Error Handling Test**: Various error scenarios handled gracefully
- **Performance Test**: Execution performance meets acceptable standards
- **Memory Management Test**: Memory usage stays within bounds
- **Cancellation Test**: Execution can be cancelled and restarted

## Dependencies
- Builds on: Task 018 (Parameter Configuration Interface)
- Required: Async execution capabilities in Streamlit
- Blocks: Task 020 (Real-time Image Display and Comparison)

## Notes
- Focus on responsive user experience
- Optimize for common use cases and workflows
- Balance between immediacy and performance
- Ensure execution state is always clear to users
- Make caching transparent but effective
