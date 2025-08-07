#  AI Email Analysis

Automatically analyze Gmail emails using **free AI models** and trigger smart workflows. Detects urgency, sentiment, and categories to route emails intelligently with Slack alerts for urgent messages.

**100% Free** - Uses local Hugging Face models, no API costs.

##  Quick Start

```bash
# 1. Clone repository
git clone https://github.com/yourusername/ai-email-analysis.git
cd ai-email-analysis

# 2. Start services (requires Docker)
docker-compose up --build

# 3. Access interfaces
# - Email Analysis API: http://localhost:5001
# - n8n Workflows: http://localhost:5678
```

##  What You Need

- [Docker Desktop](https://docker.com/get-started)
- Gmail account (for email integration)
- Slack workspace (optional, for alerts)

##  Test API

```bash
curl -X POST http://localhost:5001/analyze_email \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "URGENT: Server down!",
    "body": "Production system crashed, need help!"
  }'
```

Returns urgency level, sentiment, and category classification.

## 🔧 Setup

1. **Start services** with docker-compose
2. **Configure Gmail** trigger in n8n (port 5678)
3. **Add Slack** credentials for urgent alerts
4. **Import workflow** from n8n interface

First run downloads ~2GB of AI models automatically.