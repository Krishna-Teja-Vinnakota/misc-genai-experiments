"""
All LLM prompts and documentation type queries
"""

# === Summary Generation Prompt ===
SUMMARY_PROMPT_TEMPLATE = """You are a technical documentation expert analyzing a code file. Write a comprehensive, detailed summary as natural prose.

File: {file_path}
Language: {language}
Lines of Code: {lines_of_code}

Complete File Content:
```
{content}
```

Write a detailed summary of 400-600 words in natural prose paragraphs. Your summary should comprehensively cover:

1. The primary purpose and main responsibility of this file
2. Key functionality and features implemented
3. Important classes, functions, and methods (mention specific names from the code)
4. How this component fits in the overall system architecture
5. Integration points with other components, services, or external systems
6. Business logic, algorithms, or important processing workflows
7. External dependencies, libraries, APIs, or services used
8. Data structures, models, or schemas defined
9. Configuration requirements and environment variables
10. Error handling, validation, or security mechanisms

Write in clear, professional paragraphs. Use natural flowing sentences. 
Do not use markdown headers, bullet points, or numbered lists. Do not use bold text or special formatting. 
Just write detailed prose paragraphs that thoroughly explain the file."""


# === Documentation Type Queries ===

SYSTEM_OVERVIEW_QUERY = """System architecture and overall structure of the application.

I'm looking for files that show:
- Main application entry points and initialization
- Application configuration and setup
- Core system components and their organization
- Service layer architecture and structure
- High-level component interactions
- Application factory patterns or builders
- Dependency injection or service container setup
- Middleware and interceptor configurations
- System-wide utilities and base classes
- Module organization and boundaries
- Monolithic vs microservices architecture patterns
- Inter-service communication setup
- Shared libraries and common utilities

This includes files like:
- Python: main.py, app.py, __init__.py, wsgi.py, asgi.py
- JavaScript/Node.js: index.js, server.js, app.js, main.ts
- React: App.jsx, App.tsx, index.tsx, main.tsx
- Next.js: app/layout.tsx, pages/_app.js, next.config.js
- Application configuration files
- Service registries or dependency containers
- Core bootstrap and initialization code
- High-level orchestration components"""

API_REFERENCE_QUERY = """API endpoints, HTTP routes, and request handlers.

I'm looking for files that define:
- REST API route definitions and endpoints
- HTTP request handlers (GET, POST, PUT, DELETE, PATCH)
- API controllers and view classes
- Route decorators and path definitions
- Request parameter validation and parsing
- Response formatting and serialization
- API middleware (authentication, logging, rate limiting, CORS)
- Python: Flask routes (@app.route, Blueprint), FastAPI routers (@router.get, APIRouter)
- Python: Django views (class-based views, function views, viewsets)
- Node.js/Express: app.get(), app.post(), router.use(), Express Router
- Next.js: API routes (pages/api/, app/api/)
- Request/response models and schemas
- API versioning implementations
- WebSocket endpoint handlers
- GraphQL resolvers and schemas
- API documentation decorators (Swagger, OpenAPI)
- Endpoint security and authorization checks
- Input validation decorators
- Error handling for API endpoints

This includes files containing:
- Python: @app.route, @router.get, @router.post, Blueprint, APIRouter
- Node.js: app.get, app.post, router.get, express.Router()
- Next.js: export default function handler, NextApiRequest
- API route files in routes/, controllers/, api/ directories"""

DATA_MODELS_QUERY = """Database models, ORM classes, schema definitions, and data structures.

I'm looking for files that define:
- Database ORM models and entity classes
- Python: SQLAlchemy models (Base, Model), Django models (models.Model), Peewee
- JavaScript/Node.js: Mongoose schemas, Sequelize models, Prisma schemas, TypeORM entities
- Table definitions and column specifications
- Database relationships (ForeignKey, OneToMany, ManyToMany)
- Python: Pydantic models and validation schemas
- TypeScript: Zod schemas, class-validator, type definitions
- Database migration files and schema changes
- SQL schema creation scripts
- Python: Alembic migrations, Django migrations
- Node.js: Knex migrations, Sequelize migrations, Prisma migrations
- Data transfer objects (DTOs) and serializers
- Entity relationships and associations
- Database indexes and constraints
- Table inheritance and polymorphic models
- Composite keys and unique constraints
- Data validation rules and constraints
- GraphQL type definitions and schemas

This includes files containing:
- Python: Model, Base, models.Model, BaseModel, db.Column
- JavaScript: mongoose.Schema, Sequelize.define, @Entity, prisma.schema
- TypeScript: interface, type definitions for data structures
- Migration files: *.migration.ts, alembic versions, Django migrations"""

BUSINESS_FLOWS_QUERY = """Business logic, workflows, and core application processes.

I'm looking for files that implement:
- Business service layer implementations
- Workflow orchestration and state machines
- Transaction processing and coordination logic
- Complex business rule validation and enforcement
- Multi-step process implementations
- Domain-driven design service classes
- Use case implementations and handlers
- Business event handlers and processors
- Payment processing workflows and transaction management
- Order fulfillment and lifecycle management
- Inventory management and stock operations
- User registration and onboarding flows
- Authentication and authorization logic
- Shopping cart and checkout processes
- Subscription and billing workflows
- Notification and communication workflows
- Background job processors and task handlers
- Saga pattern implementations
- Command and query handlers
- Business calculations and algorithms
- Process coordination between multiple services
- Error recovery and compensation logic

This includes files with:
- Service classes with business logic
- Workflow managers and orchestrators
- Transaction coordinators
- Domain service implementations
- Complex multi-step processes"""

INTEGRATIONS_QUERY = """External service integrations, third-party APIs, and external dependencies.

I'm looking for files that integrate with:
- Payment gateway integrations (Stripe, PayPal, Square, Braintree)
- Shipping provider APIs (FedEx, UPS, DHL, USPS)
- Email service integrations (SendGrid, Mailgun, AWS SES)
- SMS and notification services (Twilio, SNS, Firebase)
- Cloud storage services (AWS S3, Google Cloud Storage, Azure Blob)
- Authentication providers (OAuth, Auth0, Okta, SAML, Social login)
- Message queue clients (RabbitMQ, Kafka, AWS SQS, Redis)
- Cache systems (Redis, Memcached)
- Search engines (Elasticsearch, Algolia)
- Analytics services (Google Analytics, Mixpanel, Segment)
- Monitoring and logging (Datadog, New Relic, Sentry)
- CRM integrations (Salesforce, HubSpot)
- Social media APIs (Facebook, Twitter, Instagram)
- Maps and location services (Google Maps, Mapbox)
- File processing services
- Webhook handlers for external services
- API clients and wrapper classes
- External data source integrations
- Third-party SDK implementations

This includes files with:
- HTTP client implementations (requests, httpx, aiohttp)
- Third-party library imports and usage
- API client classes and wrappers
- Webhook receivers and handlers
- External service configuration"""

DEPLOYMENT_QUERY = """Deployment configuration, infrastructure setup, and operational aspects.

I'm looking for files related to:
- Docker configuration (Dockerfile, docker-compose.yml)
- Container orchestration (Kubernetes manifests, Helm charts)
- CI/CD pipeline definitions (GitHub Actions, GitLab CI, Jenkins, CircleCI)
- Infrastructure as code (Terraform, CloudFormation, Pulumi)
- Environment configuration files (.env, config files)
- Server configuration and provisioning scripts
- Deployment automation scripts and tools
- Load balancer configurations (Nginx, HAProxy)
- Reverse proxy settings
- SSL/TLS certificate management
- Service mesh configurations (Istio, Linkerd)
- Monitoring and observability setup (Prometheus, Grafana)
- Logging configuration (ELK stack, Fluentd)
- Health check endpoints and readiness probes
- Scaling policies and autoscaling rules
- Backup and disaster recovery scripts
- Database connection pooling configuration
- Environment-specific settings (dev, staging, production)
- Secret management (Vault, AWS Secrets Manager)
- Application settings and feature flags

This includes files:
- Dockerfile, docker-compose.yml, .dockerignore
- Kubernetes YAML files (deployments, services, ingress, configmaps)
- CI/CD workflow files (.github/workflows/, .gitlab-ci.yml)
- Terraform .tf files
- Python: requirements.txt, setup.py, pyproject.toml, Pipfile
- JavaScript/Node.js: package.json, tsconfig.json
- Build configs: webpack.config.js, vite.config.ts, next.config.js
- Nginx, Apache configuration files
- Shell scripts for deployment"""

# === Query Configurations ===
# Using smart selection: min_score threshold + max_results
QUERY_CONFIGS = {
    "system_overview": {
        "query": SYSTEM_OVERVIEW_QUERY,
        "min_score": 0.45,  # Confidence score threshold
        "max_results": 20
    },
    "api_reference": {
        "query": API_REFERENCE_QUERY,
        "min_score": 0.45,  # Confidence score threshold
        "max_results": 20
    },
    "data_models": {
        "query": DATA_MODELS_QUERY,
        "min_score": 0.45,  # Confidence score threshold
        "max_results": 20
    },
    "business_flows": {
        "query": BUSINESS_FLOWS_QUERY,
        "min_score": 0.45,  # Confidence score threshold
        "max_results": 20
    },
    "integrations": {
        "query": INTEGRATIONS_QUERY,
        "min_score": 0.45,  # Confidence score threshold
        "max_results": 20
    },
    "deployment": {
        "query": DEPLOYMENT_QUERY,
        "min_score": 0.45,  # Confidence score threshold
        "max_results": 20
    }
}

# === Chat Configuration ===
CHAT_QUERY_CONFIG = {
    "min_score": 0.40,  # Lower threshold for better recall in chat
    "max_results": 10  # Fewer files for chat responses
}

# === Documentation Generation Prompt ===
DOC_GENERATION_PROMPT_TEMPLATE = """# Task: Generate {doc_type_label} Documentation

Generate comprehensive {doc_type_label} documentation for this codebase.

Requirements:
- Clear markdown structure with proper headings
- Professional and concise explanations
- Include flowcharts (mermaid) ONLY where they significantly clarify understanding
- Focus on essential information for knowledge transfer
- Direct explanations without unnecessary examples
- **CRITICAL**: Complete ALL sections before hitting token limits. Prioritize finishing the document over excessive detail.

---

## MERMAID DIAGRAM SYNTAX RULES (CRITICAL - FOLLOW STRICTLY):

**CRITICAL RULES:**
1. **NO HTML TAGS** - Mermaid does NOT support HTML tags like <br/>, <br>, or any HTML
2. **NO PARENTHESES in node labels** - Use commas or dashes instead
3. **NO SPECIAL CHARACTERS** in subgraph names - Use simple text only
4. **NO HYPHENS in state names** (stateDiagram-v2) - Use underscores or camelCase instead (e.g., use "in_reviews" or "inReviews" NOT "in-reviews")
5. **Use double quotes** for all node labels containing spaces or special characters

**CORRECT NODE LABEL PATTERNS:**
✅ A["FastAPI Server - api_server.py"]
✅ A["User Registration Form"]
✅ A["POST /api/v1/auth/registration"]
✅ A[Rate Limiter]
❌ A["FastAPI Server<br/>(api_server.py)"] - NO HTML TAGS
❌ A[Rate Limiter (access.limiter.js)] - NO PARENTHESES
❌ A[User submits registration form<br/>(with optional avatar)] - NO HTML TAGS

**CORRECT SUBGRAPH PATTERNS:**
✅ subgraph "User Interaction"
✅ subgraph Backend
✅ subgraph "API Endpoints"
❌ subgraph "Backend Services (api_server.py)" - NO PARENTHESES
❌ subgraph "User Action<br/>Flow" - NO HTML TAGS
❌ subgraph "AI & Data Services" - NO AMPERSANDS

**MULTI-LINE NODE LABELS:**
If you need multiple lines, use commas or dashes:
✅ A["User selects room, submits booking request"]
✅ A["Admin reviews - approves or rejects"]
❌ A["User selects room<br/>Submits booking"] - NO HTML

**EXAMPLES OF CORRECT MERMAID SYNTAX:**

```mermaid
graph TD
    A["Incoming HTTP Request"] --> B["Rate Limiter - access.limiter.js"]
    B --> C["DB Connection - connect.mongo.db.js"]
    C --> D["Security and CORS"]
```

```mermaid
stateDiagram-v2
    direction LR
    [*] --> pending: User places booking
    pending --> approved: Admin approves
    pending --> rejected: Admin rejects
    approved --> in_reviews: Admin marks as post-stay
    in_reviews --> completed: User submits review
    completed --> [*]
```

**STATE NAME RULES (stateDiagram-v2):**
- ❌ BAD: `in-reviews` (hyphen causes parse error)
- ✅ GOOD: `in_reviews` (use underscore)
- ✅ GOOD: `inReviews` (use camelCase)
- ✅ GOOD: `"in reviews"` (use quotes with space)

Remember: NO HTML tags, NO parentheses in labels, NO hyphens in state names, keep it simple!

---

# Codebase Files ({file_count} files):

{file_contexts}

---

Generate the documentation now. Use markdown format with clear headings and structure.
Follow Mermaid syntax rules strictly!"""
