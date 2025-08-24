# Task 022: Streamlit UI Deployment and Production Setup

## Objective
Set up production deployment configuration for the Streamlit UI application, including performance optimizations, security considerations, and deployment to cloud platforms.

## Requirements
1. Configure Streamlit for production deployment
2. Set up deployment to Streamlit Community Cloud
3. Implement performance optimizations for production
4. Add error tracking and analytics
5. Create deployment documentation and CI/CD pipeline

## File Structure to Create
```
.streamlit/
├── config.toml              # Streamlit configuration
└── secrets.toml              # Production secrets (git-ignored)

deployment/
├── streamlit_cloud.toml      # Streamlit Cloud configuration
├── requirements.txt          # Production dependencies
├── Dockerfile               # Container deployment option
└── docker-compose.yml       # Local development with Docker

docs/
├── deployment.md            # Deployment guide
└── production_setup.md      # Production configuration guide

.github/workflows/
└── deploy.yml               # CI/CD pipeline
```

## Core Features

### Production Configuration
- Optimized Streamlit settings for performance
- Memory management and resource limits
- Caching configuration for production workloads
- Security headers and CORS settings

### Cloud Deployment
- Streamlit Community Cloud setup
- Alternative deployment options (Docker, cloud providers)
- Environment variable management
- SSL/HTTPS configuration

### Performance Optimization
- Image processing optimization for cloud deployment
- Memory-efficient caching strategies
- Lazy loading and resource management
- CDN integration for static assets

### Monitoring and Analytics
- Error tracking and logging
- Performance monitoring
- Usage analytics and metrics
- Health checks and uptime monitoring

## Technical Implementation

### Streamlit Configuration
```toml
# .streamlit/config.toml
[global]
developmentMode = false
logLevel = "info"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 200
maxMessageSize = 200

[browser]
gatherUsageStats = false
showErrorDetails = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[cache]
maxAge = 3600
allowCaching = true
ttl = 24*60*60  # 24 hours
persist = "disk"
```

### Production Dependencies
```toml
# Update pyproject.toml for production
[project]
dependencies = [
    # Existing dependencies...
    "streamlit>=1.28.0",
    "streamlit-authenticator>=0.2.3",  # Optional authentication
    "sentry-sdk>=1.32.0",              # Error tracking
    "prometheus-client>=0.17.0",        # Metrics
]

[project.optional-dependencies]
production = [
    "gunicorn>=21.2.0",
    "nginx-python>=1.1.0",
    "redis>=5.0.0",
]
```

### Deployment Scripts
```python
# deployment/setup_production.py
import streamlit as st
import logging
import sentry_sdk
from sentry_sdk.integrations.streamlit import StreamlitIntegration

def setup_production_environment():
    """Configure production environment settings"""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )

    # Setup error tracking
    if st.secrets.get("SENTRY_DSN"):
        sentry_sdk.init(
            dsn=st.secrets["SENTRY_DSN"],
            integrations=[StreamlitIntegration()],
            traces_sample_rate=0.1,
            environment="production"
        )

    # Configure performance monitoring
    setup_performance_monitoring()

def setup_performance_monitoring():
    """Setup performance monitoring and metrics"""
    try:
        from prometheus_client import Counter, Histogram, start_http_server

        # Metrics
        REQUEST_COUNT = Counter('streamlit_requests_total', 'Total app requests')
        REQUEST_LATENCY = Histogram('streamlit_request_duration_seconds', 'Request latency')

        # Start metrics server
        start_http_server(8000)

    except ImportError:
        logging.warning("Prometheus client not available, skipping metrics setup")
```

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen

# Copy application code
COPY src/ ./src/
COPY .streamlit/ ./.streamlit/

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit app
CMD ["uv", "run", "streamlit", "run", "src/ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### CI/CD Pipeline
```yaml
# .github/workflows/deploy.yml
name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install uv
        uv sync

    - name: Run tests
      run: |
        uv run pytest tests/ -v

    - name: Run linting
      run: |
        uv run ruff check src/
        uv run basedpyright src/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - name: Trigger Streamlit Cloud deployment
      run: |
        curl -X POST \
        -H "Authorization: Bearer ${{ secrets.STREAMLIT_CLOUD_TOKEN }}" \
        -H "Content-Type: application/json" \
        -d '{"app_id": "${{ secrets.STREAMLIT_APP_ID }}"}' \
        "https://share.streamlit.io/api/v1/apps/redeploy"
```

## Performance Optimizations

### Memory Management
```python
def setup_memory_optimization():
    """Configure memory optimization settings"""
    import gc

    # Configure garbage collection
    gc.set_threshold(700, 10, 10)

    # Set up memory monitoring
    @st.cache_resource
    def get_memory_monitor():
        import psutil
        return psutil.Process()

    def check_memory_usage():
        monitor = get_memory_monitor()
        memory_mb = monitor.memory_info().rss / 1024 / 1024

        if memory_mb > 500:  # 500MB threshold
            st.warning(f"High memory usage: {memory_mb:.1f}MB")
            gc.collect()
```

### Caching Strategy
```python
# Production caching configuration
@st.cache_data(ttl=3600, max_entries=100)
def cache_processed_image(image_bytes, operations_hash):
    """Cache processed images with TTL and size limits"""
    pass

@st.cache_resource
def get_operation_registry():
    """Cache expensive operation initialization"""
    pass

def clear_cache_on_memory_pressure():
    """Clear cache when memory usage is high"""
    if get_memory_usage() > MEMORY_THRESHOLD:
        st.cache_data.clear()
        st.cache_resource.clear()
```

### Image Processing Optimization
```python
def optimize_for_production():
    """Apply production-specific optimizations"""

    # Limit maximum image size
    MAX_IMAGE_SIZE = (2048, 2048)

    # Use memory-efficient image processing
    def process_image_efficiently(image):
        # Resize if too large
        if image.width > MAX_IMAGE_SIZE[0] or image.height > MAX_IMAGE_SIZE[1]:
            image.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)

        # Convert to RGB if necessary to reduce memory
        if image.mode not in ('RGB', 'RGBA'):
            image = image.convert('RGB')

        return image
```

## Security Considerations

### Input Validation
```python
def validate_uploaded_image(uploaded_file):
    """Validate uploaded images for security"""

    # File size check
    if uploaded_file.size > 50 * 1024 * 1024:  # 50MB limit
        raise ValueError("File too large")

    # File type validation
    allowed_types = ['image/png', 'image/jpeg', 'image/webp']
    if uploaded_file.type not in allowed_types:
        raise ValueError("Unsupported file type")

    # Image validation
    try:
        from PIL import Image
        image = Image.open(uploaded_file)
        image.verify()
        return True
    except Exception:
        raise ValueError("Invalid image file")
```

### Rate Limiting
```python
def setup_rate_limiting():
    """Implement basic rate limiting"""

    @st.cache_data(ttl=60)
    def get_request_count(client_ip):
        return 0

    def check_rate_limit(client_ip, max_requests=30):
        count = get_request_count(client_ip)
        if count > max_requests:
            st.error("Rate limit exceeded. Please wait before making more requests.")
            st.stop()

        # Increment counter
        st.cache_data.clear()  # Simple implementation
```

## Monitoring and Analytics

### Error Tracking
```python
def setup_error_tracking():
    """Configure error tracking and logging"""

    def log_error(error, context=None):
        logging.error(f"Application error: {error}", extra={
            'context': context,
            'user_agent': st.context.headers.get('user-agent'),
            'timestamp': datetime.now().isoformat()
        })

        # Send to Sentry if configured
        if 'sentry_sdk' in globals():
            sentry_sdk.capture_exception(error)
```

### Usage Analytics
```python
def track_usage_analytics():
    """Track application usage (privacy-compliant)"""

    # Track general usage patterns
    analytics_data = {
        'session_id': st.session_state.get('session_id'),
        'page_view': True,
        'operation_count': len(st.session_state.get('pipeline_operations', [])),
        'timestamp': datetime.now().isoformat()
    }

    # Store analytics (implement based on needs)
    store_analytics(analytics_data)
```

## Deployment Options

### Streamlit Community Cloud
1. Connect GitHub repository
2. Configure build settings
3. Set up environment variables
4. Configure custom domain (optional)

### Alternative Platforms
- **Heroku**: Container deployment with Procfile
- **AWS ECS**: Container service deployment
- **Google Cloud Run**: Serverless container platform
- **Azure Container Instances**: Simple container deployment

## Success Criteria
- [ ] Streamlit app deploys successfully to production
- [ ] Performance optimizations reduce memory usage and load times
- [ ] Error tracking captures and reports issues effectively
- [ ] Security measures protect against common vulnerabilities
- [ ] CI/CD pipeline automates testing and deployment
- [ ] Monitoring provides visibility into app health and usage
- [ ] Documentation enables easy deployment replication
- [ ] App handles production load without performance degradation

## Test Cases to Implement
- **Deployment Test**: App deploys and runs in production environment
- **Performance Test**: App maintains performance under load
- **Security Test**: Input validation and rate limiting work correctly
- **Monitoring Test**: Error tracking and analytics capture data
- **CI/CD Test**: Pipeline successfully deploys updates
- **Scaling Test**: App handles concurrent users appropriately

## Dependencies
- Builds on: All Streamlit UI tasks (016-021)
- Required: Cloud platform access, monitoring tools
- Completes: Full production-ready Streamlit application

## Notes
- Prioritize security and performance from the start
- Make deployment process reproducible and documented
- Consider costs of cloud services and optimize accordingly
- Plan for scaling as user base grows
- Ensure compliance with data privacy requirements
