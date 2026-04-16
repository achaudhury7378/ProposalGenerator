# AI-Powered Proposal Generator

## Overview

An intelligent multi-agent proposal generation system powered by Microsoft Semantic Kernel and OpenAI APIs. Specialized AI agents collaboratively research, analyze, and develop comprehensive business proposals with market insights, financial projections, technical solutions, and risk assessments.

## Tech Stack

- **Framework**: Microsoft Semantic Kernel (Python)
- **AI Models**: 
  - **o3-mini** - Main orchestrator and synthesizer (faster reasoning)
  - **gpt-4o-mini** - Research agents (web search and analysis)
- **Search Integration**: Tavily API for web research
- **Configuration**: ConfigParser-based prompt management

## Agent Architecture

| Agent | Role | Responsibility |
|-------|------|-----------------|
| **Main Agent** | Orchestrator | Leads proposal creation and synthesis |
| **Market Analyst** | Research | Industry trends, competitive analysis, TAM/SAM/SOM |
| **Financial Planner** | Finance | Budgets, revenue models, ROI, break-even analysis |
| **Solution Designer** | Technical | Architecture, features, implementation roadmap |
| **Client Engagement** | Strategy | Value proposition, customer needs, testimonials |
| **Risk Manager** | Compliance | Risk assessment, security, regulatory compliance |
| **Synthesizer** | Integration | Combines all outputs into final proposal |

## Key Features

- **Web-Based Research**: Autonomous agents search for real-time market and financial data
- **Structured Analysis**: Extracts and validates market size, competitors, pricing, customer segments
- **Multi-Model Orchestration**: Combines fast reasoning (o3) with detailed analysis (gpt-4o)
- **Configurable Prompts**: All agent instructions defined in `prompts.cfg` for easy customization
- **Risk & Compliance**: Built-in assessment of security, regulations, and business risks

## Getting Started

### Prerequisites
- Python 3.9+
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)
- Tavily API key for web search integration

### Installation
```bash
pip install semantic-kernel
pip install configparser
```

### Configuration
1. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```
2. Update agent prompts in `prompts.cfg` as needed

### Running the System
```bash
python proposalmain.py
```

The system will orchestrate all agents, conduct research, and generate a comprehensive proposal.

## Workflow

1. **Initiation** - Main Agent receives business problem and context
2. **Research Phase** - Market, Financial, and Solution agents conduct parallel web research
3. **Analysis** - Each agent processes findings against configurable criteria
4. **Integration** - Client Engagement and Risk Manager assess strategy and compliance
5. **Synthesis** - Final agent compiles all insights into structured proposal
6. **Output** - Complete proposal delivered with citations and recommendations

## Project Structure
```
├── proposalmain.py      # Main entry point
├── prompts.cfg          # Agent role prompts and instructions
├── README.md            # This file
├── LICENSE              # Project license
└── augment/
    ├── servers/         # API integrations (web search)
    └── tools/           # Utility functions
```

## Environment Variables
- `OPENAI_API_KEY` - Your OpenAI API key for model access
- `TAVILY_API_KEY` - Tavily API key for web search (if using integrated search)

## Customization

Edit `prompts.cfg` to modify:
- Agent system prompts and instructions
- Crawler prompts for specialized data extraction
- Role descriptions for each specialist agent

Each section corresponds to an agent that can be updated independently.

## License

See LICENSE file for details.
