# Network Anomaly Detection System

Unsupervised machine learning system for zero-day attack detection achieving 98.8% ROC-AUC on UNSW-NB15 dataset.

## Problem

Traditional intrusion detection systems require labeled attack data, making them ineffective against novel (zero-day) threats. This system detects anomalies by learning normal network behavior patterns without prior attack examples.

## Solution

Trained three unsupervised models on legitimate traffic only, comparing their ability to detect attacks as anomalies:
- PCA Reconstruction Error
- Bayesian Gaussian Mixture Models
- Isolation Forest

## Results

**Test Set Performance:**
- ROC-AUC: 98.8%
- Macro F1-Score: 94.0%
- Anomaly Recall: 99.97% (only 22 of 7,118 attacks missed)
- False Positive Rate: 4.8% (1,432 of 30,000 normal samples)

**Model Comparison:**

| Model | ROC-AUC | Macro-F1 | Speed | 
|-------|---------|----------|-------|
| PCA Reconstruction | 98.8% | 95.0% | Fastest |
| Bayesian GMM | 99.0% | 94.0% | Slow |
| Isolation Forest | 98.5% | 94.0% | Fast |

**Winner:** PCA Reconstruction - best balance of accuracy, speed, and interpretability.

## Technical Approach
```
UNSW-NB15 Network Flows (700K samples, 49 features)
    ↓
Manual Data Cleaning:
  • Convert IPs to integers
  • Fix variable types
  • Remove duplicate rows
    ↓
Preprocessing Pipeline:
  • Handle missing values (IPs, service types)
  • OneHotEncode categorical features (proto, service, state)
  • StandardScale numerical features
    ↓
Train only on Normal Traffic
    ↓
PCA Dimensionality Reduction (95% variance)
    ↓
Reconstruction Error = |Input - Reconstructed|
    ↓
Threshold Optimization (F1-maximization)
    ↓
High Error → Anomaly | Low Error → Normal
```

## Key Findings

**Attack Indicators (by reconstruction error):**

1. **TTL Anomalies** (sttl, dttl, ct_state_ttl)
   - Spoofed packets, unusual routing paths

2. **TCP Handshake Irregularities** (synack, ackdat)
   - Scanning, incomplete sessions, flooding

3. **IP Anomalies** (srcip, dstip, is_sm_ips_ports)
   - Probing from unexpected network locations

4. **Payload Patterns** (Sload, sbytes, Sintpkt, dmeansz)
   - Data exfiltration, tunneling, bot behavior

5. **Service Irregularities** (service_Unknown, ct_flw_http_mthd)
   - Exploitation tools, malformed requests

## Dataset

**UNSW-NB15** - Network Intrusion Detection Dataset (2015)
- 175,341 network flow records
- 49 features (flow stats, TCP flags, timing, payload)
- 9 attack categories: Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms
- Source: https://research.unsw.edu.au/projects/unsw-nb15-dataset

## Installation
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Usage
```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load('pca_anomaly_detection.joblib')

# Predict on network flow data (49 features matching UNSW-NB15 format)
flow_data = pd.read_csv('network_flows.csv')
predictions = model.predict(flow_data)
# Output: 0 = normal, 1 = anomaly
```

## Project Structure
```
unsw-anomaly-detection/
├── unsw.csv                        # UNSW-NB15 dataset
├── anomaly_detection.ipynb         # Full analysis notebook
├── pca_anomaly_detection.joblib    # Trained model
├── requirements.txt
└── README.md
```

## Why This Matters

**Zero-day detection capability:**
- No labeled attack data required
- Detects novel attack patterns
- Complements supervised intrusion detection

**Production considerations:**
- 4.8% false positive rate acceptable for security monitoring
- Fast inference (PCA is efficient)
- Interpretable (reconstruction error shows which features are anomalous)

## Technologies

Python • scikit-learn • PCA • Bayesian Gaussian Mixture • Isolation Forest • pandas • NumPy • matplotlib

## Contact

**Rafiou Diallo**  
rafioudiallo12@gmail.com  
[GitHub](https://github.com/D-Rafiou) | [LinkedIn](https://www.linkedin.com/in/rafiou-diallo-004522260/)
