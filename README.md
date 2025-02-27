# Stock Potential Identifier System

An advanced stock analysis system that identifies potential trading opportunities based on machine learning predictions, technical indicators, fundamentals, and alternative data.

## Features

- **Multi-layered Architecture**: From data acquisition to decision generation
- **Free Data Sources**: Uses Yahoo Finance, Alpha Vantage, SEC EDGAR, and other free APIs
- **Comprehensive Feature Engineering**: Technical, fundamental, and alternative data features
- **Advanced ML Models**: XGBoost, LightGBM, LSTM, Prophet
- **Adaptive Model Fusion**: Dynamic weighting based on performance and market regime
- **Risk-Aware Signal Generation**: Volatility-adjusted position sizing and risk metrics
- **Portfolio-Aware Decisions**: Ranks opportunities based on both signal quality and portfolio fit

## Architecture

```
┌───────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Acquisition │     │ Feature Pipeline │     │ Model Hierarchy │
│  ---------------  │────▶│ --------------- │────▶│ --------------- │
│  - Market Data    │     │ - Technical     │     │ - Time Series   │
│  - Fundamentals   │     │ - Fundamental   │     │ - Ensemble ML   │
│  - Alt Data       │     │ - Alternative   │     │ - Deep Learning │
└───────────────────┘     └─────────────────┘     └────────┬────────┘
                                                           │
┌───────────────────┐     ┌─────────────────┐     ┌───────▼────────┐
│  Decision Engine  │◀────│ Signal Generator │◀────│ Fusion Layer   │
│  ---------------  │     │ --------------- │     │ ------------- │
│  - Time Horizons  │     │ - Opportunity   │     │ - Probability │
│  - Confidence     │     │ - Risk Analysis │     │ - Consensus   │
└───────────────────┘     └─────────────────┘     └───────────────┘
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose (for containerized deployment)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-potential-identifier.git
   cd stock-potential-identifier
   ```

2. Build and run using Docker:
   ```bash
   docker-compose up -d
   ```

3. Or install locally:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Edit the configuration in `config/config.yaml` to:
- Add your preferred stock universe
- Configure data sources and API keys
- Adjust model parameters
- Set signal quality thresholds

### API Keys

For full functionality, set the following environment variables:
- `ALPHA_VANTAGE_API_KEY` - Get from [Alpha Vantage](https://www.alphavantage.co/)
- `FMP_API_KEY` - Get from [Financial Modeling Prep](https://financialmodelingprep.com/)

These can be added to a `.env` file in the project root:
```
ALPHA_VANTAGE_API_KEY=your_key_here
FMP_API_KEY=your_key_here
```

## Usage

### Running the System

Start the system with default settings:
```bash
python main.py
```

Train models only, without running the full system:
```bash
python main.py --train-only
```

Use a custom configuration file:
```bash
python main.py --config path/to/custom_config.yaml
```

### Accessing Reports

Decision reports are saved to the `reports` directory in multiple formats (JSON, CSV, HTML).

### Using the Jupyter Interface

Access the Jupyter interface for interactive analysis:
1. With Docker: Open http://localhost:8888 in your browser
2. Run `jupyter notebook` in the project directory if installed locally

## Improving the System

Some ideas for enhancing the system:
- Add more alternative data sources
- Implement backtesting module
- Add reinforcement learning models
- Create a web-based dashboard
- Set up email/SMS alerts for signals

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Technical indicators powered by pandas-ta
- Machine learning capabilities from scikit-learn, XGBoost, LightGBM
- Deep learning powered by TensorFlow