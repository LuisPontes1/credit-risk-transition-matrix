# Credit Risk Transition Matrix Library

A professional Python library for credit risk analysis using transition matrices. Designed for data scientists, risk analysts, and financial institutions to analyze credit portfolio behavior, predict defaults, and generate risk reports.

## 🎯 Overview

This library provides comprehensive tools for:
- **Transition Matrix Calculation**: Calculate migration probabilities between risk buckets
- **Professional Visualizations**: Generate publication-ready heatmaps and charts
- **Risk Analytics**: PD (Probability of Default) calculations and portfolio metrics
- **Segmented Analysis**: Compare different portfolio segments
- **Model Validation**: Backtesting and performance metrics

## 🚀 Quick Start

### Installation

**Development Version (Current):**
```bash
# Clone the repository
git clone https://github.com/LuisPontes1/credit-risk-transition-matrix.git
cd credit-risk-transition-matrix

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

**PyPI Release (Coming Soon):**
```bash
pip install credit-risk-transition-matrix
```

### Basic Usage

```python
from credit_risk_transition_matrix import TransitionMatrixLearner

# Initialize learner with custom buckets
learner = TransitionMatrixLearner(
    buckets=[0, 15, 30, 60, 90, 120, 180, 240, 360],
    time_horizon=12
)

# Fit on your data
learner.fit(
    df_panel,
    id_col="id_contrato",
    time_col="data_ref",
    bucket_col="dias_atraso"
)

# Generate professional heatmaps
learner.plot_heatmaps(modes=["global"], save_dir="./outputs/")

# Calculate PD metrics
pd_results = learner.calculate_pd(time_horizon=12)
```

## 📊 Key Features

### 1. Flexible Bucket Definitions
- Support for custom risk buckets (days past due)
- Default configuration: [0, 15, 30, 60, 90, 120, 180, 240, 360+ days]
- Automatic bucket assignment and validation

### 2. Professional Visualizations
- **Global Heatmaps**: Publication-ready transition matrices
- **Segmented Analysis**: Compare different portfolio segments
- **Flow Diagrams**: Visualize transition patterns
- **Trend Analysis**: Time series of migration patterns

### 3. Advanced Analytics
- **PD Calculations**: 1-year and lifetime default probabilities
- **Portfolio Metrics**: Migration rates and concentration measures
- **Stress Testing**: Scenario analysis capabilities
- **Model Validation**: Backtesting and performance metrics

### 4. Data Requirements

Your input data should have the following structure:

| Column | Type | Description |
|--------|------|-------------|
| `id_contrato` | string | Unique contract identifier |
| `data_ref` | datetime | Reference date for observation |
| `dias_atraso` | int | Days past due (bucket value) |
| `segmento` | string (optional) | Portfolio segment for grouped analysis |
| `valor_exposicao` | float (optional) | Exposure amount for weighted analysis |

## 📈 Advanced Usage

### Segmented Analysis

```python
# Analyze by portfolio segments
learner.fit(
    df_panel,
    id_col="id_contrato", 
    time_col="data_ref",
    bucket_col="dias_atraso",
    segment_col="produto_tipo"  # Segment by product type
)

# Generate segment comparison
learner.plot_heatmaps(modes=["global", "segmented"], save_dir="./analysis/")

# Get matrices for all segments
matrices = learner.transform(modes=['global', 'segmented'])
```

### Predictive Analysis

```python
# Predict future state distributions
current_state = "30-59"  # Current bucket
future_distribution = learner.predict_transitions(current_state, n_periods=6)

# Calculate portfolio-level PD
pd_metrics = learner.calculate_pd(
    time_horizon=12,
    default_buckets=["240-359", "360+"]
)
```

### Model Validation

```python
# Validate on out-of-time data
validation_results = learner.validate_model(
    test_data=df_test,
    metrics=['accuracy', 'auc']
)
```

## 🛠️ Development Setup

### For Contributors

```bash
# Clone repository
git clone https://github.com/LuisPontes1/credit-risk-transition-matrix.git
cd credit-risk-transition-matrix

# Install development dependencies
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/ -v

# Run example
python examples/basic_usage.py

# Format code
black src/ tests/ examples/
```

### Project Status

- ✅ **Phase 1 Complete**: Core functionality and basic visualizations
- 🔄 **Phase 2 In Progress**: Advanced analytics and validation framework
- ⏭️ **Phase 3 Planned**: Performance optimization and interactive features
- 🏢 **Phase 4 Planned**: Enterprise features and regulatory compliance

## 📚 Documentation

### API Reference
- [TransitionMatrixLearner API](docs/api/transition_matrix_learner.md)
- [Visualization Functions](docs/api/visualization.md)
- [Risk Metrics](docs/api/risk_metrics.md)

### Tutorials
- [Getting Started Tutorial](docs/tutorials/getting_started.ipynb)
- [Advanced Analysis Examples](docs/tutorials/advanced_analysis.ipynb)
- [Regulatory Compliance](docs/tutorials/ifrs9_compliance.ipynb)

### Mathematical Background
- [Transition Matrix Theory](docs/theory/transition_matrices.md)
- [PD Calculation Methods](docs/theory/pd_calculations.md)

## 🏦 Regulatory Compliance

This library supports calculations required for:
- **IFRS 9** Expected Credit Loss models
- **CECL** Current Expected Credit Loss methodology
- **Basel** regulatory capital calculations
- Audit trail and model documentation

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution
- Additional risk metrics
- New visualization types
- Performance optimizations
- Documentation improvements
- Bug fixes and testing

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Links

- **PyPI**: https://pypi.org/project/credit-risk-transition-matrix/
- **Documentation**: https://credit-risk-transition-matrix.readthedocs.io/
- **Source Code**: https://github.com/LuisPontes1/credit-risk-transition-matrix
- **Issue Tracker**: https://github.com/LuisPontes1/credit-risk-transition-matrix/issues

## 📞 Support

- **Documentation**: Check our comprehensive docs
- **Issues**: Report bugs via GitHub issues
- **Discussions**: Use GitHub discussions for questions
- **Email**: For commercial support inquiries

## 🎯 Roadmap

### Phase 1 (Current) ✅
- Core TransitionMatrixLearner class
- Basic visualization capabilities
- Global and segmented analysis

### Phase 2 (Next)
- Advanced analytics and validation framework
- Interactive Plotly visualizations
- Performance optimizations

### Phase 3 (Future)
- Web API endpoints
- Real-time monitoring capabilities
- Advanced stress testing tools

### Phase 4 (Long-term)
- Machine learning integration
- Automated model selection
- Cloud deployment options

---

**Made with ❤️ for the credit risk community**
Ferramenta para análise de risco de crédito através de matrizes de transição - diagnóstico rápido de inadimplência, geração de heatmaps e calibração de provisões
